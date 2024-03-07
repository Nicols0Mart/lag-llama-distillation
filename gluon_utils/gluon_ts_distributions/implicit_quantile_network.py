# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

from functools import partial
from typing import Dict, Optional, Tuple, Callable

import torch
import torch.nn as nn
import torch.distributions as td
import torch.nn.functional as F
from torch.distributions import Distribution, Beta, constraints, Gumbel

from gluonts.core.component import validated
from gluonts.torch.distributions import DistributionOutput

# from gluonts.torch.distributions.distribution_output import DistributionOutput
from gluonts.torch.modules.lambda_layer import LambdaLayer


SCALE_OFFSET = 1e-6

class QuantileLayer(nn.Module):
    r"""
    Implicit Quantile Layer from the paper ``IQN for Distributional
    Reinforcement Learning`` (https://arxiv.org/abs/1806.06923) by
    Dabney et al. 2018.
    """

    def __init__(self, num_output: int, cos_embedding_dim: int = 128):
        super().__init__()

        self.output_layer = nn.Sequential(
            nn.Linear(cos_embedding_dim, cos_embedding_dim),
            nn.PReLU(),
            nn.Linear(cos_embedding_dim, num_output),
        )

        self.register_buffer("integers", torch.arange(0, cos_embedding_dim))

    def forward(self, tau: torch.Tensor) -> torch.Tensor:  # tau: [B, T]
        cos_emb_tau = torch.cos(tau.unsqueeze(-1) * self.integers * torch.pi)
        return self.output_layer(cos_emb_tau)


class ImplicitQuantileModule(nn.Module):
    r"""
    Implicit Quantile Network from the paper ``IQN for Distributional
    Reinforcement Learning`` (https://arxiv.org/abs/1806.06923) by
    Dabney et al. 2018.
    """

    def __init__(
        self,
        in_features: int,
        args_dim: Dict[str, int],
        domain_map: Callable[..., Tuple[torch.Tensor]],
        concentration1: float = 1.0,
        concentration0: float = 1.0,
        output_domain_map=None,
        cos_embedding_dim: int = 64,
    ):
        super().__init__()
        self.output_domain_map = output_domain_map
        self.domain_map = domain_map
        self.beta = Beta(concentration1=concentration1, concentration0=concentration0)

        self.quantile_layer = QuantileLayer(
            in_features, cos_embedding_dim=cos_embedding_dim
        )
        self.output_layer = nn.Sequential(
            nn.Linear(in_features, in_features), nn.PReLU()
        )

        self.proj = nn.ModuleList(
            [nn.Linear(in_features, dim) for dim in args_dim.values()]
        )

    def forward(self, inputs: torch.Tensor):
        if self.training:
            taus = self.beta.sample(sample_shape=inputs.shape[:-1]).to(inputs.device)
        else:
            taus = torch.rand(size=inputs.shape[:-1], device=inputs.device)

        emb_taus = self.quantile_layer(taus)
        emb_inputs = inputs * (1.0 + emb_taus)

        emb_outputs = self.output_layer(emb_inputs)
        outputs = [proj(emb_outputs).squeeze(-1) for proj in self.proj]
        if self.output_domain_map is not None:
            outputs = [self.output_domain_map(output) for output in outputs]
        return (*self.domain_map(*outputs), taus)


class ImplicitQuantileNetwork(Distribution):
    r"""
    Distribution class for the Implicit Quantile from which
    we can sample or calculate the quantile loss.

    Parameters
    ----------
    outputs
        Outputs from the Implicit Quantile Network.
    taus
        Tensor random numbers from the Beta or Uniform distribution for the
        corresponding outputs.
    """

    arg_constraints: Dict[str, constraints.Constraint] = {}

    def __init__(self, outputs: torch.Tensor, taus: torch.Tensor, validate_args=None):
        self.taus = taus
        self.outputs = outputs

        super().__init__(batch_shape=outputs.shape, validate_args=validate_args)

    @torch.no_grad()
    def sample(self, sample_shape=torch.Size()) -> torch.Tensor:
        return self.outputs

    def quantile_loss(self, value: torch.Tensor) -> torch.Tensor:
        # penalize by tau for under-predicting
        # and by 1-tau for over-predicting
        return (self.taus - (value < self.outputs).float()) * (value - self.outputs)


class ImplicitQuantileNetworkOutput(DistributionOutput):
    r"""
    DistributionOutput class for the IQN from the paper
    ``Probabilistic Time Series Forecasting with Implicit Quantile Networks``
    (https://arxiv.org/abs/2107.03743) by Gouttes et al. 2021.

    Parameters
    ----------
    output_domain
        Optional domain mapping of the output. Can be "positive", "unit"
        or None.
    concentration1
        Alpha parameter of the Beta distribution when sampling the taus
        during training.
    concentration0
        Beta parameter of the Beta distribution when sampling the taus
        during training.
    cos_embedding_dim
        The embedding dimension for the taus embedding layer of IQN.
        Default is 64.
    """

    distr_cls = ImplicitQuantileNetwork
    args_dim = {"quantile_function": 1}

    @validated()
    def __init__(
        self,
        output_domain: Optional[str] = None,
        concentration1: float = 1.0,
        concentration0: float = 1.0,
        cos_embedding_dim: int = 64,
    ) -> None:
        super().__init__()

        self.concentration1 = concentration1
        self.concentration0 = concentration0
        self.cos_embedding_dim = cos_embedding_dim

        if output_domain in ["positive", "unit"]:
            output_domain_map_func = {
                "positive": F.softplus,
                "unit": partial(F.softmax, dim=-1),
            }
            self.output_domain_map = output_domain_map_func[output_domain]
        else:
            self.output_domain_map = None

    def get_args_proj(self, in_features: int) -> nn.Module:
        return ImplicitQuantileModule(
            in_features=in_features,
            args_dim=self.args_dim,
            output_domain_map=self.output_domain_map,
            domain_map=LambdaLayer(self.domain_map),
            concentration1=self.concentration1,
            concentration0=self.concentration0,
            cos_embedding_dim=self.cos_embedding_dim,
        )

    @classmethod
    def domain_map(cls, *args):
        return args

    def distribution(self, distr_args, loc=0, scale=None) -> ImplicitQuantileNetwork:
        (outputs, taus) = distr_args
        if scale is not None:
            outputs = outputs * scale
        if loc is not None:
            outputs = outputs + loc
        return self.distr_cls(outputs=outputs, taus=taus)

    @property
    def event_shape(self):
        return ()

    def loss(
        self,
        target: torch.Tensor,
        distr_args: Tuple[torch.Tensor, ...],
        loc: Optional[torch.Tensor] = None,
        scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        distribution = self.distribution(distr_args, loc=loc, scale=scale)
        return distribution.quantile_loss(target)

def inverse_softplus(tensor: torch.Tensor):
    return tensor + torch.log(1 - torch.exp(-tensor))


class GaussianDistributionOutput(nn.Module):
    def __init__(
        self,
        network,
        dist_dim,
        use_tied_cov=False,
        use_trainable_cov=True,
        sigma=None,
        max_scale=1.0,
        scale_nonlinearity="softplus",
    ):
        """A Gaussian distribution on top of a neural network.

        Args:
            network (nn.Module):
                A torch module that outputs the parameters of the
                Gaussian distribution.
            dist_dim ([type]):
                The dimension of the Gaussian distribution.
            use_tied_cov (bool, optional):
                Whether to use a tied covariance matrix.
                Defaults to False.
            use_trainable_cov (bool, optional):
                True if the covariance matrix is to be learned.
                Defaults to True. If False, the covariance matrix is set to I.
            sigma (float, optional):
                Initial value of scale.
            max_scale (float, optional):
                Maximum scale when using sigmoid non-linearity.
            scale_nonlinearity (str, optional):
                Which non-linearity to use for scale -- sigmoid or softplus.
                Defaults to softplus.
        """
        super().__init__()
        self.dist_dim = dist_dim
        self.network = network
        self.use_tied_cov = use_tied_cov
        self.use_trainable_cov = use_trainable_cov
        self.max_scale = max_scale
        self.scale_nonlinearity = scale_nonlinearity
        if not self.use_trainable_cov:
            assert (
                sigma is not None
            ), "sigma cannot be None for non-trainable covariance!"
            self.sigma = sigma
        if self.use_trainable_cov and self.use_tied_cov:
            self.usigma = nn.Parameter(
                inverse_softplus(
                    torch.full([1, dist_dim], sigma if sigma is not None else 1e-1)
                )
            )

    def forward(self, tensor):
        """The forward pass.

        Args:
            tensor (torch.Tensor):
                The input tensor.

        Returns:
            out_dist:
                The Gaussian distribution with parameters obtained by passing
                the input tensor through self.network.
        """
        args_tensor = self.network(tensor)
        mean_tensor = args_tensor[..., :self.dist_dim]
        if self.use_trainable_cov:
            if self.use_tied_cov:
                if self.scale_nonlinearity == "sigmoid":
                    scale_tensor = self.max_scale * torch.sigmoid(self.usigma)
                else:
                    scale_tensor = F.softplus(self.usigma)
                out_dist = td.normal.Normal(mean_tensor, scale_tensor + SCALE_OFFSET)
                out_dist = td.independent.Independent(out_dist, 1)
            else:
                if self.scale_nonlinearity == "sigmoid":
                    scale_tensor = self.max_scale * torch.sigmoid(
                        args_tensor[..., self.dist_dim :]
                    )
                else:
                    scale_tensor = F.softplus(args_tensor[..., self.dist_dim :])

                out_dist = td.normal.Normal(mean_tensor, scale_tensor + SCALE_OFFSET)
                out_dist = td.independent.Independent(out_dist, 1)
        else:
            out_dist = td.normal.Normal(mean_tensor, self.sigma)
            out_dist = td.independent.Independent(out_dist, 1)
        return out_dist



class PtArgProj(nn.Module):
    r"""
    A PyTorch module that can be used to project from a dense layer
    to PyTorch distribution arguments.

    Parameters
    ----------
    in_features
        Size of the incoming features.
    dim_args
        Dictionary with string key and int value
        dimension of each arguments that will be passed to the domain
        map, the names are not used.
    domain_map
        Function returning a tuple containing one tensor
        a function or a nn.Module. This will be called with num_args
        arguments and should return a tuple of outputs that will be
        used when calling the distribution constructor.
    """

    def __init__(
        self,
        in_features: int,
        args_dim: Dict[str, int],
        domain_map: Callable[..., Tuple[torch.Tensor]],
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.args_dim = args_dim
        self.proj = nn.ModuleList(
            [nn.Linear(in_features, dim) for dim in args_dim.values()]
        )
        self.domain_map = domain_map

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        params_unbounded = [proj(x) for proj in self.proj]

        return self.domain_map(*params_unbounded)

class GumbelDistributionOutput(DistributionOutput):
    args_dim: Dict[str, int] = {"loc": 1, "scale": 1}
    distr_cls: type = td.gumbel.Gumbel

    @validated()
    def __init__(self, scale: Optional[float] = None) -> None:
        self.scale = scale

    def get_args_proj(self, in_features: int) -> nn.Module:
        return PtArgProj(in_features, self.args_dim, LambdaLayer(self.domain_map))

    @classmethod
    def domain_map(cls, loc, scale):
        return loc.squeeze(-1), F.softplus(scale).squeeze(-1)

    def distribution(
        self, distr_args, loc: Optional[torch.Tensor] = None, scale: Optional[torch.Tensor] = None
    ) -> Distribution:
        loc, scale = distr_args
        if torch.isnan(scale).any():
            scale = torch.nan_to_num(scale, nan=1e-6)
        if torch.isnan(loc).any():
            loc = torch.nan_to_num(loc, nan=1e-6)
        return self.distr_cls(loc, scale)
    
    @property
    def event_shape(self) -> Tuple:
        return ()

iqn = ImplicitQuantileNetworkOutput()
