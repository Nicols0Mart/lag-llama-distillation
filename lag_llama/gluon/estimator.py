from collections import namedtuple
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple
from gluonts.dataset.loader import TrainDataLoader
from gluonts.itertools import Cached
from gluonts.torch.batchify import batchify
import numpy as np
import pytorch_lightning as pl
import torch
from gluonts.core.component import validated
from gluonts.dataset.common import Dataset
from gluonts.dataset.field_names import FieldName
from gluonts.torch.modules.lambda_layer import LambdaLayer
from gluonts.dataset.loader import as_stacked_batches
from gluonts.dataset.stat import calculate_dataset_statistics
from gluonts.itertools import Cyclic
from gluonts.time_feature import (
    get_lags_for_frequency,
    time_features_from_frequency_str,
)
from gluonts.evaluation import make_evaluation_predictions, Evaluator

from gluonts.torch.model.estimator import PyTorchLightningEstimator
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.torch.modules.loss import DistributionLoss, NegativeLogLikelihood
from gluonts.transform import (
    AddObservedValuesIndicator,
    AddTimeFeatures,
    Chain,
    DummyValueImputation,
    ExpectedNumInstanceSampler,
    InstanceSampler,
    AsNumpyArray,
    InstanceSplitter,
    TestSplitSampler,
    Transformation,
    ValidationSplitSampler,
    VstackFeatures
)
from peft import LoraConfig, get_peft_model

from gluonts.torch.model.deepar import DeepAREstimator
from gluonts.torch.distributions import StudentTOutput, NormalOutput
from gluon_utils.gluon_ts_distributions.implicit_quantile_network import (
    ImplicitQuantileNetworkOutput, GumbelDistributionOutput, PtArgProj
)
from gluonts.time_feature import TimeFeature
from gluonts.dataset.field_names import FieldName
from gluonts.transform import (
    AddObservedValuesIndicator,
    AddTimeFeatures,
    Chain,
    ExpectedNumInstanceSampler,
    TestSplitSampler,
    Transformation,
    ValidationSplitSampler,
)
from torch.nn import functional as F
from gluonts.core.component import validated
from gluonts.dataset.common import DataEntry
from gluonts.transform.sampler import InstanceSampler
from typing import Optional, Type
from gluonts.transform._base import FlatMapTransformation

from torch.distributions import (
    AffineTransform,
    Distribution,
    TransformedDistribution,
    StudentT,
    Independent
)
from gluonts.torch.distributions import DistributionOutput
from lag_llama.gluon.lightning_module import LagLlamaLightningModule, LagLlamaDALightningModule
from gluonts.core.component import validated
from gluonts.dataset.common import Dataset
from gluonts.env import env
from gluonts.itertools import Cached
from gluonts.model import Estimator, Predictor
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.transform import Transformation
import torch.nn as nn
from typing import NamedTuple
from lag_llama.model.lora import lora


class TrainOutput(NamedTuple):
    transformation: Transformation
    trained_net: nn.Module
    trainer: pl.Trainer
    predictor: PyTorchPredictor

PREDICTION_INPUT_NAMES = [
    #TODO: add lpls features
    "feat_static_real",
    "past_target",
    "past_observed_values",
]
TRAINING_INPUT_NAMES = PREDICTION_INPUT_NAMES + [
    "future_target",
    "future_observed_values",
]


class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)


class ValidSplitSampler(InstanceSampler):
    """
    Sampler used for prediction.

    Always selects the last time point for splitting i.e. the forecast point
    for the time series.
    """

    allow_empty_interval: bool = False

    def __call__(self, ts: np.ndarray, offset=None) -> np.ndarray:
        a, b = self._get_bounds(ts)
        assert self.allow_empty_interval or a <= b
        if offset is None:
            return np.array(list(range(a, b+1))) if a <= b else np.array([], dtype=int)
        else:
            return np.array(list(range(b+offset, b+1))) if a <= b else np.array([], dtype=int)




class CustomInstanceSplitter(FlatMapTransformation):

    @validated()
    def __init__(
        self,
        target_field: str,
        is_pad_field: str,
        start_field: str,
        forecast_start_field: str,
        instance_sampler: InstanceSampler,
        past_length: int,
        future_length: int,
        lead_time: int = 0,
        output_NTC: bool = True,
        time_series_fields: List[str] = [],
        dummy_value: float = 0.0,
        backtest_offset: int = -100,
    ) -> None:
        super().__init__()

        assert future_length > 0, "The value of `future_length` should be > 0"

        self.instance_sampler = instance_sampler
        self.past_length = past_length
        self.future_length = future_length
        self.lead_time = lead_time
        self.output_NTC = output_NTC
        self.ts_fields = time_series_fields
        self.target_field = target_field
        self.is_pad_field = is_pad_field
        self.start_field = start_field
        self.forecast_start_field = forecast_start_field
        self.dummy_value = dummy_value
        self.backtest_offset = backtest_offset
        

    def _past(self, col_name):
        return f"past_{col_name}"

    def _future(self, col_name):
        return f"future_{col_name}"

    def flatmap_transform(
        self, data: DataEntry, is_train: bool
    ) -> Iterator[DataEntry]:
        future_length = self.future_length
        lt = self.lead_time
        slice_cols = self.ts_fields + [self.target_field]
        target = data[self.target_field]
        if self.instance_sampler.__class__.__name__ == "ValidSplitSampler":
            sampled_indices = self.instance_sampler(target, offset=self.backtest_offset)
        else:
            sampled_indices = self.instance_sampler(target)

        for i in sampled_indices:
            pad_length = max(self.past_length - i, 0)
            d = data.copy()
            for ts_field in slice_cols:
                if i > self.past_length:
                    # truncate to past_length
                    past_piece = d[ts_field][..., i - self.past_length : i]
                elif i < self.past_length:
                    pad_block = (
                        np.ones(
                            d[ts_field].shape[:-1] + (pad_length,),
                            dtype=d[ts_field].dtype,
                        )
                        * self.dummy_value
                    )
                    past_piece = np.concatenate(
                        [pad_block, d[ts_field][..., :i]], axis=-1
                    )
                else:
                    past_piece = d[ts_field][..., :i]
                d[self._past(ts_field)] = past_piece
                d[self._future(ts_field)] = d[ts_field][
                    ..., i + lt : i + lt + future_length
                ]
                del d[ts_field]
            pad_indicator = np.zeros(self.past_length, dtype=target.dtype)
            if pad_length > 0:
                pad_indicator[:pad_length] = 1

            if self.output_NTC:
                for ts_field in slice_cols:
                    d[self._past(ts_field)] = d[
                        self._past(ts_field)
                    ].transpose()
                    d[self._future(ts_field)] = d[
                        self._future(ts_field)
                    ].transpose()

            d[self._past(self.is_pad_field)] = pad_indicator
            d[self.forecast_start_field] = d[self.start_field] + i + lt
            yield d




class AffineTransformed(TransformedDistribution):
    def __init__(self, base_distribution: Distribution, loc=None, scale=None, event_dim=0):
        self.scale = 1.0 if scale is None else scale
        self.loc = 0.0 if loc is None else loc

        super().__init__(base_distribution, [AffineTransform(loc=self.loc, scale=self.scale, event_dim=event_dim)])

    @property
    def mean(self):
        """
        Returns the mean of the distribution.
        """
        return self.base_dist.mean * self.scale + self.loc

    @property
    def variance(self):
        """
        Returns the variance of the distribution.
        """
        return self.base_dist.variance * self.scale**2

    @property
    def stddev(self):
        """
        Returns the standard deviation of the distribution.
        """
        return self.variance.sqrt()
    
    
    def entropy(self):
        """
        Computes the entropy of the affine-transformed distribution.

        For an affine transformation Y = scale * X + loc,
        the entropy H(Y) = H(X) + log|scale|

        For multi-dimensional distributions, if scale is a vector,
        H(Y) = H(X) + sum(log|scale_i|) over all dimensions i.
        """
        # Compute the entropy of the base distribution
        base_entropy = self.base_dist.entropy()

        # Ensure scale is a tensor for consistency
        if isinstance(self.scale, (float, int)):
            scale_tensor = torch.tensor(self.scale)
        else:
            scale_tensor = torch.as_tensor(self.scale)

        # Compute log|scale|
        log_abs_scale = torch.log(torch.abs(scale_tensor))

        # If the distribution is multi-dimensional, sum the log|scale| across event dimensions
        # Assuming scale is applied per event dimension
        if log_abs_scale.dim() > 0:
            entropy_adjustment = log_abs_scale.sum()
        else:
            entropy_adjustment = log_abs_scale

        # Total entropy is base entropy plus the adjustment
        total_entropy = base_entropy + entropy_adjustment

        return total_entropy


class HFDistributionOutput:
    distr_cls: type
    in_features: int
    args_dim: Dict[str, int]

    def __init__(self, dim: int = 1) -> None:
        self.dim = dim
        self.args_dim = {k: dim * self.args_dim[k] for k in self.args_dim}

    def _base_distribution(self, distr_args):
        if self.dim == 1:
            return self.distr_cls(*distr_args)
        else:
            return Independent(self.distr_cls(*distr_args), 1)

    def distribution(
        self,
        distr_args,
        loc: Optional[torch.Tensor] = None,
        scale: Optional[torch.Tensor] = None,
    ) -> Distribution:
        distr = self._base_distribution(distr_args)
        if loc is None and scale is None:
            return distr
        else:
            return AffineTransformed(distr, loc=loc, scale=scale, event_dim=3)

    @property
    def event_shape(self) -> Tuple:
        r"""
        Shape of each individual event contemplated by the distributions that this object constructs.
        """
        return () if self.dim == 1 else (self.dim,)

    @property
    def event_dim(self) -> int:
        r"""
        Number of event dimensions, i.e., length of the `event_shape` tuple, of the distributions that this object
        constructs.
        """
        return len(self.event_shape)

    @property
    def value_in_support(self) -> float:
        r"""
        A float that will have a valid numeric value when computing the log-loss of the corresponding distribution. By
        default 0.0. This value will be used when padding data series.
        """
        return 0.0

    def get_args_proj(self, in_features: int) -> nn.Module:
        r"""
        Return the parameter projection layer that maps the input to the appropriate parameters of the distribution.
        """
        return PtArgProj(
            in_features=in_features,
            args_dim={key: 1 for key in self.args_dim.keys()},
            domain_map=LambdaLayer(self.domain_map),
        )

    def domain_map(self, *args: torch.Tensor):
        r"""
        Converts arguments to the right shape and domain. The domain depends on the type of distribution, while the
        correct shape is obtained by reshaping the trailing axis in such a way that the returned tensors define a
        distribution of the right event_shape.
        """
        raise NotImplementedError()

    @staticmethod
    def squareplus(x: torch.Tensor) -> torch.Tensor:
        r"""
        Helper to map inputs to the positive orthant by applying the square-plus operation. Reference:
        https://twitter.com/jon_barron/status/1387167648669048833
        """
        return (x + torch.sqrt(torch.square(x) + 4.0)) / 2.0

class MutivariateStudentTOutput(HFDistributionOutput):
    args_dim: Dict[str, int] = {"df": 1, "loc": 1, "scale": 1}
    distr_cls: type = StudentT

    def _base_distribution(self, distr_args):
        return super()._base_distribution(distr_args)

    @classmethod
    def domain_map(
        cls, df: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor
    ):
        epsilon = torch.finfo(scale.dtype).eps
        scale = F.softplus(scale).clamp_min(epsilon)
        df = 2.0 + F.softplus(df)
        return df.squeeze(-1), loc.squeeze(-1), scale.squeeze(-1)

    @property
    def event_shape(self) -> Tuple:
        return (3,)

    def distribution(
        self,
        distr_args,
        loc: Optional[torch.Tensor] = None,
        scale: Optional[torch.Tensor] = None,
    ) -> Distribution:
        distr = self._base_distribution(distr_args)
        if loc is None and scale is None:
            return distr
        else:
            return AffineTransformed(distr, loc=loc, scale=scale, event_dim=self.event_dim)

class LagLlamaEstimator(PyTorchLightningEstimator):
    """
    An estimator training a ConvTSMixer model for forecasting.

    This class is uses the model defined in ``ConvTSMixerModel``,
    and wraps it into a ``ConvTSMixerLightningModule`` for training
    purposes: training is performed using PyTorch Lightning's ``pl.Trainer``
    class.

    Parameters
    ----------
    prediction_length
        Length of the prediction horizon.
    context_length
        Number of time steps prior to prediction time that the model
        takes as inputs (default: ``10 * prediction_length``).
    lr
        Learning rate (default: ``1e-3``).
    weight_decay
        Weight decay regularization parameter (default: ``1e-8``).
    distr_output
        Distribution to use to evaluate observations and sample predictions
        (default: StudentTOutput()).
    loss
        Loss to be optimized during training
        (default: ``NegativeLogLikelihood()``).
    batch_norm
        Whether to apply batch normalization.
    batch_size
        The size of the batches to be used for training (default: 32).
    num_batches_per_epoch
        Number of batches to be processed in each training epoch
            (default: 50).
    trainer_kwargs
        Additional arguments to provide to ``pl.Trainer`` for construction.
    train_sampler
        Controls the sampling of windows during training.
    validation_sampler
        Controls the sampling of windows during validation.
    """

    @validated()
    def __init__(
        self,
        prediction_length: int,
        context_length: Optional[int] = None,
        input_size: int = 1,
        n_layer: int = 1,
        n_embd_per_head: int = 32,
        n_head: int = 4,
        max_context_length: int = 2048,
        rope_scaling=None,
        scaling: Optional[str] = "mean",
        lr: float = 1e-3,
        weight_decay: float = 1e-8,
        # Augmentations arguments
        aug_prob: float = 0.5,
        freq_mask_rate: float = 0.5,
        freq_mixing_rate: float = 0.25,
        jitter_prob: float = 0.0,
        jitter_sigma: float = 0.03,
        scaling_prob: float = 0.0,
        scaling_sigma: float = 0.1,
        rotation_prob: float = 0.0,
        permutation_prob: float = 0.0,
        permutation_max_segments: int = 5,
        permutation_seg_mode: str = "equal",
        magnitude_warp_prob: float = 0.0,
        magnitude_warp_sigma: float = 0.2,
        magnitude_warp_knot: int = 4,
        time_warp_prob: float = 0.0,
        time_warp_sigma: float = 0.2,
        time_warp_knot: int = 4,
        window_slice_prob: float = 0.0,
        window_slice_reduce_ratio: float = 0.9,
        window_warp_prob: float = 0.0,
        window_warp_window_ratio: float = 0.1,
        window_warp_scales: list = [0.5, 2.0],
        # Continuning model arguments
        distr_output: str = "studentT",
        loss: DistributionLoss = NegativeLogLikelihood(),
        num_parallel_samples: int = 100,
        batch_size: int = 32,
        num_batches_per_epoch: int = 50,
        trainer_kwargs: Optional[Dict[str, Any]] = None,
        train_sampler: Optional[InstanceSampler] = None,
        validation_sampler: Optional[InstanceSampler] = None,
        time_feat: bool = False,
        dropout: float = 0.0,
        lags_seq: list = ["Q", "M", "W", "D", "H", "T", "S"],
        data_id_to_name_map: dict = {},
        use_cosine_annealing_lr: bool = False,
        cosine_annealing_lr_args: dict = {},
        track_loss_per_series: bool = False,
        ckpt_path: Optional[str] = None,
        use_feat_dynamic_real=True,
        mistral=False,
    ) -> None:
        default_trainer_kwargs = {"max_epochs": 100}
        if trainer_kwargs is not None:
            default_trainer_kwargs.update(trainer_kwargs)
        super().__init__(trainer_kwargs=default_trainer_kwargs)

        self.scaling = scaling
        self.mistral=mistral
        self.input_size = input_size
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.max_context_length = max_context_length

        lag_indices = []
        for freq in lags_seq:
            lag_indices.extend(
                get_lags_for_frequency(freq_str=freq, num_default_lags=1)
            )
        self.use_feat_dynamic_real = use_feat_dynamic_real

        if len(lag_indices):
            self.lags_seq = sorted(set(lag_indices))
            self.lags_seq = [lag_index - 1 for lag_index in self.lags_seq] # len 83, max: 1092
        else:
            self.lags_seq = []

        self.n_head = n_head
        self.n_layer = n_layer
        self.n_embd_per_head = n_embd_per_head
        self.rope_scaling = rope_scaling

        self.lr = lr
        self.weight_decay = weight_decay
        if distr_output == "studentT":
            distro_output = MutivariateStudentTOutput(dim=3)
        elif distr_output == "iqn":
            distro_output = ImplicitQuantileNetworkOutput()
        elif distr_output == "gumbel":
            distro_output = GumbelDistributionOutput()
        self.distr_output = distro_output
        self.num_parallel_samples = num_parallel_samples
        self.loss = loss
        self.batch_size = batch_size # 32
        self.num_batches_per_epoch = num_batches_per_epoch # 50

        self.train_sampler = train_sampler or ExpectedNumInstanceSampler(
            num_instances=1.0, min_future=prediction_length
        )
        self.validation_sampler = validation_sampler or ValidSplitSampler(min_future=self.prediction_length)

        self.aug_prob = aug_prob
        self.freq_mask_rate = freq_mask_rate
        self.freq_mixing_rate = freq_mixing_rate
        self.jitter_prob = jitter_prob
        self.jitter_sigma = jitter_sigma
        self.scaling_prob = scaling_prob
        self.scaling_sigma = scaling_sigma
        self.rotation_prob = rotation_prob
        self.permutation_prob = permutation_prob
        self.permutation_max_segments = permutation_max_segments
        self.permutation_seg_mode = permutation_seg_mode
        self.magnitude_warp_prob = magnitude_warp_prob
        self.magnitude_warp_sigma = magnitude_warp_sigma
        self.magnitude_warp_knot = magnitude_warp_knot
        self.time_warp_prob = time_warp_prob
        self.time_warp_sigma = time_warp_sigma
        self.time_warp_knot = time_warp_knot
        self.window_slice_prob = window_slice_prob
        self.window_slice_reduce_ratio = window_slice_reduce_ratio
        self.window_warp_prob = window_warp_prob
        self.window_warp_window_ratio = window_warp_window_ratio
        self.window_warp_scales = window_warp_scales
        self.track_loss_per_series = track_loss_per_series

        self.time_feat = time_feat
        self.dropout = dropout
        self.data_id_to_name_map = data_id_to_name_map
        self.ckpt_path = ckpt_path

        self.use_cosine_annealing_lr = use_cosine_annealing_lr
        self.cosine_annealing_lr_args = cosine_annealing_lr_args
        # self.transformation = self.create_transformation()

    def train(
        self,
        training_data: Dataset,
        validation_data: Optional[Dataset] = None,
        shuffle_buffer_length: Optional[int] = None,
        cache_data: bool = False,
        ckpt_path: Optional[str] = None,
        **kwargs,
    ) -> PyTorchPredictor:
        use_lora = kwargs.get("use_lora", True)
        return self.train_model(
            training_data,
            validation_data,
            shuffle_buffer_length=shuffle_buffer_length,
            cache_data=cache_data,
            ckpt_path=ckpt_path,
            use_lora=use_lora,
        ).predictor
    
    def train_with_module(
        self,
        training_data: Dataset,
        validation_data: Optional[Dataset] = None,
        shuffle_buffer_length: Optional[int] = None,
        cache_data: bool = False,
        ckpt_path: Optional[str] = None,
        **kwargs,
    ) -> LagLlamaLightningModule:
        use_lora = kwargs.get("use_lora", True)
        return self.train_model(
            training_data,
            validation_data,
            shuffle_buffer_length=shuffle_buffer_length,
            cache_data=cache_data,
            ckpt_path=ckpt_path,
            use_lora=use_lora,
        ).trained_net

    @classmethod
    def derive_auto_fields(cls, train_iter):
        stats = calculate_dataset_statistics(train_iter)

        return {
            "num_feat_dynamic_real": stats.num_feat_dynamic_real,
            "num_feat_static_real": stats.feat_static_real,
            "num_feat_static_cat": len(stats.feat_static_cat),
            "cardinality": [len(cats) for cats in stats.feat_static_cat],
        }

    def create_transformation(self) -> Transformation:
        if self.time_feat:
            return Chain(
                [
                    AddTimeFeatures(
                        start_field=FieldName.START,
                        target_field=FieldName.TARGET,
                        output_field=FieldName.FEAT_TIME,
                        time_features=time_features_from_frequency_str("5T"),
                        pred_length=self.prediction_length,
                    ),
                    # VstackFeatures(
                    #     output_field=FieldName.FEAT_TIME,
                    #     input_fields=[FieldName.FEAT_TIME] + [FieldName.FEAT_DYNAMIC_REAL]
                    # ),
                    # FilterTransformation(lambda x: sum(abs(x[FieldName.TARGET])) > 0),
                    AddObservedValuesIndicator(
                        target_field=FieldName.TARGET,
                        output_field=FieldName.OBSERVED_VALUES,
                        imputation_method=DummyValueImputation(0.0),
                    ),
                    AsNumpyArray(
                        field=FieldName.FEAT_STATIC_REAL,
                        expected_ndim=1,
                        # dtype=int,
                    ),
                    AsNumpyArray(
                        field=FieldName.FEAT_DYNAMIC_REAL,
                        expected_ndim=2,
                        # dtype=int,
                    ),
                ]
            )
        else:
            return Chain(
                [
                    AddObservedValuesIndicator(
                        target_field=FieldName.TARGET,
                        output_field=FieldName.OBSERVED_VALUES,
                        imputation_method=DummyValueImputation(0.0),
                    ),
                ]
            )

    def create_lightning_module(self, use_kv_cache: bool = False, use_lora=True) -> pl.LightningModule:
        model_kwargs = {
            "input_size": self.input_size,
            "context_length": self.context_length,
            "max_context_length": self.max_context_length,
            "lags_seq": self.lags_seq,
            "n_layer": self.n_layer,
            "n_embd_per_head": self.n_embd_per_head,
            "n_head": self.n_head,
            "scaling": self.scaling,
            "distr_output": self.distr_output,
            "num_parallel_samples": self.num_parallel_samples,
            "rope_scaling": self.rope_scaling,
            "time_feat": self.time_feat,
            "dropout": self.dropout,
        }
        if self.ckpt_path is not None and use_lora:
            with lora(r=8, alpha=16, dropout=0.05, enabled=True):
                module = LagLlamaLightningModule.load_from_checkpoint(
                    checkpoint_path=self.ckpt_path,
                    strict=False,
                    loss=self.loss,
                    lr=self.lr,
                    weight_decay=self.weight_decay,
                    context_length=self.context_length,
                    prediction_length=self.prediction_length,
                    model_kwargs=model_kwargs,
                    use_feat_dynamic_real=self.use_feat_dynamic_real,
                    # Augmentations
                    aug_prob=self.aug_prob,
                    freq_mask_rate=self.freq_mask_rate,
                    freq_mixing_rate=self.freq_mixing_rate,
                    jitter_prob=self.jitter_prob,
                    jitter_sigma=self.jitter_sigma,
                    scaling_prob=self.scaling_prob,
                    scaling_sigma=self.scaling_sigma,
                    rotation_prob=self.rotation_prob,
                    permutation_prob=self.permutation_prob,
                    permutation_max_segments=self.permutation_max_segments,
                    permutation_seg_mode=self.permutation_seg_mode,
                    magnitude_warp_prob=self.magnitude_warp_prob,
                    magnitude_warp_sigma=self.magnitude_warp_sigma,
                    magnitude_warp_knot=self.magnitude_warp_knot,
                    time_warp_prob=self.time_warp_prob,
                    time_warp_sigma=self.time_warp_sigma,
                    time_warp_knot=self.time_warp_knot,
                    window_slice_prob=self.window_slice_prob,
                    window_slice_reduce_ratio=self.window_slice_reduce_ratio,
                    window_warp_prob=self.window_warp_prob,
                    window_warp_window_ratio=self.window_warp_window_ratio,
                    window_warp_scales=self.window_warp_scales,
                    use_kv_cache=use_kv_cache,
                    data_id_to_name_map=self.data_id_to_name_map,
                    use_cosine_annealing_lr=self.use_cosine_annealing_lr,
                    cosine_annealing_lr_args=self.cosine_annealing_lr_args,
                    track_loss_per_series=self.track_loss_per_series,
                    mistral=self.mistral,
                )
                module.print_trainable_parameters(module.model)
                return module
        elif self.ckpt_path is not None and not use_lora:
            module = LagLlamaLightningModule.load_from_checkpoint(
                    checkpoint_path=self.ckpt_path,
                    strict=False,
                    loss=self.loss,
                    lr=self.lr,
                    weight_decay=self.weight_decay,
                    context_length=self.context_length,
                    prediction_length=self.prediction_length,
                    model_kwargs=model_kwargs,
                    use_feat_dynamic_real=self.use_feat_dynamic_real,
                    # Augmentations
                    aug_prob=self.aug_prob,
                    freq_mask_rate=self.freq_mask_rate,
                    freq_mixing_rate=self.freq_mixing_rate,
                    jitter_prob=self.jitter_prob,
                    jitter_sigma=self.jitter_sigma,
                    scaling_prob=self.scaling_prob,
                    scaling_sigma=self.scaling_sigma,
                    rotation_prob=self.rotation_prob,
                    permutation_prob=self.permutation_prob,
                    permutation_max_segments=self.permutation_max_segments,
                    permutation_seg_mode=self.permutation_seg_mode,
                    magnitude_warp_prob=self.magnitude_warp_prob,
                    magnitude_warp_sigma=self.magnitude_warp_sigma,
                    magnitude_warp_knot=self.magnitude_warp_knot,
                    time_warp_prob=self.time_warp_prob,
                    time_warp_sigma=self.time_warp_sigma,
                    time_warp_knot=self.time_warp_knot,
                    window_slice_prob=self.window_slice_prob,
                    window_slice_reduce_ratio=self.window_slice_reduce_ratio,
                    window_warp_prob=self.window_warp_prob,
                    window_warp_window_ratio=self.window_warp_window_ratio,
                    window_warp_scales=self.window_warp_scales,
                    use_kv_cache=use_kv_cache,
                    data_id_to_name_map=self.data_id_to_name_map,
                    use_cosine_annealing_lr=self.use_cosine_annealing_lr,
                    cosine_annealing_lr_args=self.cosine_annealing_lr_args,
                    track_loss_per_series=self.track_loss_per_series,
                    mistral=self.mistral,
                )
            # for param in module.model.parameters():
            #     param.requires_grad = False
            # last_transformer = module.model.transformer.h[-2:]
            # for param in last_transformer.parameters():
            #     param.requires_grad = True
            return module
        else:
            return LagLlamaLightningModule(
                loss=self.loss,
                lr=self.lr,
                weight_decay=self.weight_decay,
                context_length=self.context_length,
                prediction_length=self.prediction_length,
                model_kwargs=model_kwargs,
                # Augmentations
                aug_prob=self.aug_prob,
                freq_mask_rate=self.freq_mask_rate,
                freq_mixing_rate=self.freq_mixing_rate,
                jitter_prob=self.jitter_prob,
                use_feat_dynamic_real=self.use_feat_dynamic_real,
                jitter_sigma=self.jitter_sigma,
                scaling_prob=self.scaling_prob,
                scaling_sigma=self.scaling_sigma,
                rotation_prob=self.rotation_prob,
                permutation_prob=self.permutation_prob,
                permutation_max_segments=self.permutation_max_segments,
                permutation_seg_mode=self.permutation_seg_mode,
                magnitude_warp_prob=self.magnitude_warp_prob,
                magnitude_warp_sigma=self.magnitude_warp_sigma,
                magnitude_warp_knot=self.magnitude_warp_knot,
                time_warp_prob=self.time_warp_prob,
                time_warp_sigma=self.time_warp_sigma,
                time_warp_knot=self.time_warp_knot,
                window_slice_prob=self.window_slice_prob,
                window_slice_reduce_ratio=self.window_slice_reduce_ratio,
                window_warp_prob=self.window_warp_prob,
                window_warp_window_ratio=self.window_warp_window_ratio,
                window_warp_scales=self.window_warp_scales,
                use_kv_cache=use_kv_cache,
                data_id_to_name_map=self.data_id_to_name_map,
                use_cosine_annealing_lr=self.use_cosine_annealing_lr,
                cosine_annealing_lr_args=self.cosine_annealing_lr_args,
                track_loss_per_series=self.track_loss_per_series,
                mistral=self.mistral,
            )

    def _create_instance_splitter(self, module: LagLlamaLightningModule, mode: str):
        assert mode in {"training", "validation", "test"}

        instance_sampler = {
            "training": self.train_sampler,
            "validation": self.validation_sampler,
            "test": self.validation_sampler #TestSplitSampler(),
        }[mode]

        return CustomInstanceSplitter(
            target_field=FieldName.TARGET,
            is_pad_field=FieldName.IS_PAD,
            start_field=FieldName.START,
            forecast_start_field=FieldName.FORECAST_START,
            instance_sampler=instance_sampler,
            past_length=self.context_length + max(self.lags_seq),
            future_length=self.prediction_length,
            time_series_fields=[FieldName.FEAT_TIME, FieldName.OBSERVED_VALUES]
            if self.time_feat
            else [FieldName.OBSERVED_VALUES],
            dummy_value=self.distr_output.value_in_support,
        )

    def create_training_data_loader(
        self,
        data: Dataset,
        module: LagLlamaLightningModule,
        shuffle_buffer_length: Optional[int] = None,
        **kwargs,
    ) -> Iterable:
        data = Cyclic(data).stream()
        instances = self._create_instance_splitter(module, "training").apply(
            data, is_train=True
        )
        if self.time_feat:
            return as_stacked_batches(
                instances,
                batch_size=self.batch_size,
                shuffle_buffer_length=shuffle_buffer_length,
                field_names=TRAINING_INPUT_NAMES
                + ["past_time_feat", "future_time_feat", "data_id", "item_id", "feat_static_real", ],
                # + ["past_time_feat", "future_time_feat"],
                # + ["past_time_feat", "future_time_feat"],
                output_type=torch.tensor,
                num_batches_per_epoch=self.num_batches_per_epoch,
            )

        else:
            return as_stacked_batches(
                instances,
                batch_size=self.batch_size,
                shuffle_buffer_length=shuffle_buffer_length,
                # field_names=TRAINING_INPUT_NAMES,
                field_names=TRAINING_INPUT_NAMES + ["data_id", "item_id"],
                # field_names=TRAINING_INPUT_NAMES + [],
                output_type=torch.tensor,
                num_batches_per_epoch=self.num_batches_per_epoch,
            )

    def create_validation_data_loader(
        self,
        data: Dataset,
        module: LagLlamaLightningModule,
        **kwargs,
    ) -> Iterable:
        instances = self._create_instance_splitter(module, "validation").apply(
            data, is_train=True
        )
        if self.time_feat:
            return as_stacked_batches(
                instances,
                batch_size=self.batch_size,
                field_names=TRAINING_INPUT_NAMES
                + ["past_time_feat", "future_time_feat", "data_id", "item_id"],
                # + ["past_time_feat", "future_time_feat"],
                output_type=torch.tensor,
            )
        else:
            return as_stacked_batches(
                instances,
                batch_size=self.batch_size,
                field_names=TRAINING_INPUT_NAMES + ["data_id", "item_id"],
                # field_names=TRAINING_INPUT_NAMES,
                output_type=torch.tensor,
            )
    
    def create_trainer_dl(self, dataset, module):
        # instances = self._create_instance_splitter(module, "training").apply(
        #     dataset, is_train=True
        # )
        if self.time_feat:
            # return as_stacked_batches(
            #     instances,
            #     batch_size=self.batch_size,
            #     field_names=TRAINING_INPUT_NAMES
            #     + ["past_time_feat", "future_time_feat", "data_id", "item_id"],
            #     # + ["past_time_feat", "future_time_feat"],
            #     output_type=torch.tensor,
            # )
            data_loader = TrainDataLoader(
    # We cache the dataset, to make training faster
                Cached(dataset),
                batch_size=self.batch_size,
                stack_fn=batchify,
                transform=self.create_transformation(),
                # num_batches_per_epoch=100,
            )
            return data_loader

    def train_model(
        self,
        training_data: Dataset,
        validation_data: Optional[Dataset] = None,
        from_predictor: Optional[PyTorchPredictor] = None,
        shuffle_buffer_length: Optional[int] = None,
        cache_data: bool = False,
        ckpt_path: Optional[str] = None,
        **kwargs,
    ) -> TrainOutput:
        transformation = self.create_transformation()

        with env._let(max_idle_transforms=min(len(training_data), 100)):
            transformed_training_data = transformation.apply(
                training_data, is_train=True
            )
            if cache_data:
                transformed_training_data = Cached(transformed_training_data)

            training_network = self.create_lightning_module(use_lora=kwargs.get("use_lora", True))

            training_data_loader = self.create_training_data_loader(
                transformed_training_data,
                training_network,
                shuffle_buffer_length=shuffle_buffer_length,
            )

        validation_data_loader = None

        if validation_data is not None:
            with env._let(max_idle_transforms=max(len(validation_data), 100)):
                transformed_validation_data = transformation.apply(
                    validation_data, is_train=True
                )
                if cache_data:
                    transformed_validation_data = Cached(
                        transformed_validation_data
                    )

                validation_data_loader = self.create_validation_data_loader(
                    transformed_validation_data,
                    training_network,
                )

        if from_predictor is not None:
            training_network.load_state_dict(
                from_predictor.network.state_dict()
            )

        monitor = "train_loss" if validation_data is None else "val_loss"
        checkpoint = pl.callbacks.ModelCheckpoint(
            monitor=monitor, mode="min", verbose=True
        )

        custom_callbacks = self.trainer_kwargs.get("callbacks", [])
        callbacks = [checkpoint] + custom_callbacks
        trainer_kwargs = {**self.trainer_kwargs, "callbacks": callbacks, "precision": "bf16-mixed"}
        trainer_kwargs["accelerator"] = "gpu"
        trainer_kwargs["devices"] = [1]
        trainer = pl.Trainer(**trainer_kwargs)
        training_network.strict_loading = False
        trainer.fit(
            model=training_network,
            train_dataloaders=training_data_loader,
            val_dataloaders=validation_data_loader,
            ckpt_path=ckpt_path,
        )

        # logger.info(f"Loading best model from {checkpoint.best_model_path}")
        if hasattr(checkpoint, "best_model_path"):
            best_model = training_network.__class__.load_from_checkpoint(
                checkpoint.best_model_path,
                strict=False,
            )
        else:
            best_model = training_network

        return TrainOutput(
            transformation=transformation,
            trained_net=best_model,
            trainer=trainer,
            predictor=self.create_predictor(transformation, best_model),
        )


    def create_predictor(
        self,
        transformation: Transformation,
        module: LagLlamaLightningModule,
    ) -> PyTorchPredictor:
        prediction_splitter = self._create_instance_splitter(module, "test")
        if self.time_feat:
            return PyTorchPredictor(
                input_transform=transformation + prediction_splitter,
                input_names=PREDICTION_INPUT_NAMES
                + ["past_time_feat", "future_time_feat"],
                prediction_net=module,
                batch_size=self.batch_size,
                prediction_length=self.prediction_length,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
        else:
            return PyTorchPredictor(
                input_transform=transformation + prediction_splitter,
                input_names=PREDICTION_INPUT_NAMES,
                prediction_net=module,
                batch_size=self.batch_size,
                prediction_length=self.prediction_length,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
    
    def create_test_dataloader(self,
        module,
        freq,
        data,
        batch_size: int,
        **kwargs,
        ):
        ##+ ["past_time_feat", "future_time_feat", "data_id", "item_id"]
        transformation = self.create_transformation()
        transformed_data = transformation.apply(data, is_train=False)

        # We create a test Instance splitter to sample the very last
        # context window from the dataset provided.
        instance_sampler = self._create_instance_splitter(module, "test")

        # We apply the transformations in test mode
        testing_instances = instance_sampler.apply(transformed_data, is_train=False)
        
        return as_stacked_batches(
            testing_instances,
            batch_size=batch_size,
            output_type=torch.tensor,
            field_names=TRAINING_INPUT_NAMES
                + ["past_time_feat", "future_time_feat", "feat_static_real"],
        )


class DomainAdaptationTrainer(pl.Trainer):

    def _fit_impl(self, model: pl.LightningModule, train_dataloaders: Any | pl.LightningDataModule | None = None, val_dataloaders: Any | None = None, datamodule: pl.LightningDataModule | None = None, ckpt_path: str | Path | None = None) -> None:
        return super()._fit_impl(model, train_dataloaders, val_dataloaders, datamodule, ckpt_path)

# class LagLlamaDomainAdaptiationEstimator(LagLlamaEstimator):
    

#     @classmethod
#     def derive_auto_fields(cls, train_iter):
#         stats = calculate_dataset_statistics(train_iter)

#         return {
#             "num_feat_dynamic_real": stats.num_feat_dynamic_real,
#             "num_feat_static_cat": len(stats.feat_static_cat),
#             "cardinality": [len(cats) for cats in stats.feat_static_cat],
#         }

#     def create_transformation(self) -> Transformation:
#         if self.time_feat:
#             return Chain(
#                 [
#                     AddTimeFeatures(
#                         start_field=FieldName.START,
#                         target_field=FieldName.TARGET,
#                         output_field=FieldName.FEAT_TIME,
#                         time_features=time_features_from_frequency_str("5T"),
#                         pred_length=self.prediction_length,
#                     ),
#                     # VstackFeatures(
#                     #     output_field=FieldName.FEAT_TIME,
#                     #     input_fields=[FieldName.FEAT_TIME] + [FieldName.FEAT_DYNAMIC_REAL]
#                     # ),
#                     # FilterTransformation(lambda x: sum(abs(x[FieldName.TARGET])) > 0),
#                     AddObservedValuesIndicator(
#                         target_field=FieldName.TARGET,
#                         output_field=FieldName.OBSERVED_VALUES,
#                         imputation_method=DummyValueImputation(0.0),
#                     ),
                    
#                 ]
#             )
#         else:
#             return Chain(
#                 [
#                     AddObservedValuesIndicator(
#                         target_field=FieldName.TARGET,
#                         output_field=FieldName.OBSERVED_VALUES,
#                         imputation_method=DummyValueImputation(0.0),
#                     ),
#                 ]
#             )

#     def create_lightning_module(self, use_kv_cache: bool = False) -> pl.LightningModule:
#         model_kwargs = {
#             "input_size": self.input_size,
#             "context_length": self.context_length,
#             "max_context_length": self.max_context_length,
#             "lags_seq": self.lags_seq,
#             "n_layer": self.n_layer,
#             "n_embd_per_head": self.n_embd_per_head,
#             "n_head": self.n_head,
#             "scaling": self.scaling,
#             "distr_output": self.distr_output,
#             "num_parallel_samples": self.num_parallel_samples,
#             "rope_scaling": self.rope_scaling,
#             "time_feat": self.time_feat,
#             "dropout": self.dropout,
#         }
#         if self.ckpt_path is not None:
#             with lora(r=4, alpha=16, dropout=0.05, enabled=True):
#                 module = LagLlamaLightningModule.load_from_checkpoint(
#                     checkpoint_path=self.ckpt_path,
#                     strict=False,
#                     loss=self.loss,
#                     lr=self.lr,
#                     weight_decay=self.weight_decay,
#                     context_length=self.context_length,
#                     prediction_length=self.prediction_length,
#                     model_kwargs=model_kwargs,
#                     # Augmentations
#                     aug_prob=self.aug_prob,
#                     freq_mask_rate=self.freq_mask_rate,
#                     freq_mixing_rate=self.freq_mixing_rate,
#                     jitter_prob=self.jitter_prob,
#                     jitter_sigma=self.jitter_sigma,
#                     scaling_prob=self.scaling_prob,
#                     scaling_sigma=self.scaling_sigma,
#                     rotation_prob=self.rotation_prob,
#                     permutation_prob=self.permutation_prob,
#                     permutation_max_segments=self.permutation_max_segments,
#                     permutation_seg_mode=self.permutation_seg_mode,
#                     magnitude_warp_prob=self.magnitude_warp_prob,
#                     magnitude_warp_sigma=self.magnitude_warp_sigma,
#                     magnitude_warp_knot=self.magnitude_warp_knot,
#                     time_warp_prob=self.time_warp_prob,
#                     time_warp_sigma=self.time_warp_sigma,
#                     time_warp_knot=self.time_warp_knot,
#                     window_slice_prob=self.window_slice_prob,
#                     window_slice_reduce_ratio=self.window_slice_reduce_ratio,
#                     window_warp_prob=self.window_warp_prob,
#                     window_warp_window_ratio=self.window_warp_window_ratio,
#                     window_warp_scales=self.window_warp_scales,
#                     use_kv_cache=use_kv_cache,
#                     data_id_to_name_map=self.data_id_to_name_map,
#                     use_cosine_annealing_lr=self.use_cosine_annealing_lr,
#                     cosine_annealing_lr_args=self.cosine_annealing_lr_args,
#                     track_loss_per_series=self.track_loss_per_series,
#                 )
#                 module.print_trainable_parameters(module.model)
#                 return module
#         else:
#             return LagLlamaDALightningModule(
#                 loss=self.loss,
#                 lr=self.lr,
#                 weight_decay=self.weight_decay,
#                 context_length=self.context_length,
#                 prediction_length=self.prediction_length,
#                 model_kwargs=model_kwargs,
#                 # Augmentations
#                 aug_prob=self.aug_prob,
#                 freq_mask_rate=self.freq_mask_rate,
#                 freq_mixing_rate=self.freq_mixing_rate,
#                 jitter_prob=self.jitter_prob,
#                 jitter_sigma=self.jitter_sigma,
#                 scaling_prob=self.scaling_prob,
#                 scaling_sigma=self.scaling_sigma,
#                 rotation_prob=self.rotation_prob,
#                 permutation_prob=self.permutation_prob,
#                 permutation_max_segments=self.permutation_max_segments,
#                 permutation_seg_mode=self.permutation_seg_mode,
#                 magnitude_warp_prob=self.magnitude_warp_prob,
#                 magnitude_warp_sigma=self.magnitude_warp_sigma,
#                 magnitude_warp_knot=self.magnitude_warp_knot,
#                 time_warp_prob=self.time_warp_prob,
#                 time_warp_sigma=self.time_warp_sigma,
#                 time_warp_knot=self.time_warp_knot,
#                 window_slice_prob=self.window_slice_prob,
#                 window_slice_reduce_ratio=self.window_slice_reduce_ratio,
#                 window_warp_prob=self.window_warp_prob,
#                 window_warp_window_ratio=self.window_warp_window_ratio,
#                 window_warp_scales=self.window_warp_scales,
#                 use_kv_cache=use_kv_cache,
#                 data_id_to_name_map=self.data_id_to_name_map,
#                 use_cosine_annealing_lr=self.use_cosine_annealing_lr,
#                 cosine_annealing_lr_args=self.cosine_annealing_lr_args,
#                 track_loss_per_series=self.track_loss_per_series,
#             )

#     def _create_instance_splitter(self, module: LagLlamaLightningModule, mode: str):
#         assert mode in ["training", "validation", "test"]

#         instance_sampler = {
#             "training": self.train_sampler,
#             "validation": self.validation_sampler,
#             "test": TestSplitSampler(),
#         }[mode]

#         return InstanceSplitter(
#             target_field=FieldName.TARGET,
#             is_pad_field=FieldName.IS_PAD,
#             start_field=FieldName.START,
#             forecast_start_field=FieldName.FORECAST_START,
#             instance_sampler=instance_sampler,
#             past_length=self.context_length + max(self.lags_seq),
#             future_length=self.prediction_length,
#             time_series_fields=[FieldName.FEAT_TIME, FieldName.OBSERVED_VALUES]
#             if self.time_feat
#             else [FieldName.OBSERVED_VALUES],
#             dummy_value=self.distr_output.value_in_support,
#         )

#     def create_training_data_loader(
#         self,
#         data: List[Dataset],
#         module: LagLlamaLightningModule,
#         shuffle_buffer_length: Optional[int] = None,
#         **kwargs,
#     ) -> Iterable:
#         data = Cyclic(data).stream()
#         instances = self._create_instance_splitter(module, "training").apply(
#             data, is_train=True
#         )
#         if self.time_feat:
#             return as_stacked_batches(
#                 instances,
#                 batch_size=self.batch_size,
#                 shuffle_buffer_length=shuffle_buffer_length,
#                 field_names=TRAINING_INPUT_NAMES
#                 + ["past_time_feat", "future_time_feat", "data_id", "item_id"],
#                 # + ["past_time_feat", "future_time_feat"],
#                 # + ["past_time_feat", "future_time_feat"],
#                 output_type=torch.tensor,
#                 num_batches_per_epoch=self.num_batches_per_epoch,
#             )

#         else:
#             return as_stacked_batches(
#                 instances,
#                 batch_size=self.batch_size,
#                 shuffle_buffer_length=shuffle_buffer_length,
#                 # field_names=TRAINING_INPUT_NAMES,
#                 field_names=TRAINING_INPUT_NAMES + ["data_id", "item_id"],
#                 # field_names=TRAINING_INPUT_NAMES + [],
#                 output_type=torch.tensor,
#                 num_batches_per_epoch=self.num_batches_per_epoch,
#             )

#     def create_validation_data_loader(
#         self,
#         data: Dataset,
#         module: LagLlamaLightningModule,
#         **kwargs,
#     ) -> Iterable:
#         instances = self._create_instance_splitter(module, "validation").apply(
#             data, is_train=True
#         )
#         if self.time_feat:
#             return as_stacked_batches(
#                 instances,
#                 batch_size=self.batch_size,
#                 field_names=TRAINING_INPUT_NAMES
#                 + ["past_time_feat", "future_time_feat", "data_id", "item_id"],
#                 # + ["past_time_feat", "future_time_feat"],
#                 output_type=torch.tensor,
#             )
#         else:
#             return as_stacked_batches(
#                 instances,
#                 batch_size=self.batch_size,
#                 field_names=TRAINING_INPUT_NAMES + ["data_id", "item_id"],
#                 # field_names=TRAINING_INPUT_NAMES,
#                 output_type=torch.tensor,
#             )
    
#     def create_trainer_dl(self, dataset, module):
#         # instances = self._create_instance_splitter(module, "training").apply(
#         #     dataset, is_train=True
#         # )
#         if self.time_feat:
#             # return as_stacked_batches(
#             #     instances,
#             #     batch_size=self.batch_size,
#             #     field_names=TRAINING_INPUT_NAMES
#             #     + ["past_time_feat", "future_time_feat", "data_id", "item_id"],
#             #     # + ["past_time_feat", "future_time_feat"],
#             #     output_type=torch.tensor,
#             # )
#             data_loader = TrainDataLoader(
#     # We cache the dataset, to make training faster
#                 Cached(dataset),
#                 batch_size=self.batch_size,
#                 stack_fn=batchify,
#                 transform=self.create_transformation(),
#                 # num_batches_per_epoch=100,
#             )
#             return data_loader

#     def train_model(
#         self,
#         training_data: List[Dataset],
#         validation_data: Optional[Dataset] = None,
#         from_predictor: Optional[PyTorchPredictor] = None,
#         shuffle_buffer_length: Optional[int] = None,
#         cache_data: bool = False,
#         ckpt_path: Optional[str] = None,
#         **kwargs,
#     ) -> TrainOutput:
#         transformation = self.create_transformation()

#         with env._let(max_idle_transforms=max(len(training_data), 100)):
#             transformed_training_data = transformation.apply(
#                 training_data, is_train=True
#             )
#             if cache_data:
#                 transformed_training_data = Cached(transformed_training_data)

#             training_network = self.create_lightning_module()

#             training_data_loader = self.create_training_data_loader(
#                 transformed_training_data,
#                 training_network,
#                 shuffle_buffer_length=shuffle_buffer_length,
#             )

#         validation_data_loader = None

#         if validation_data is not None:
#             with env._let(max_idle_transforms=max(len(validation_data), 100)):
#                 transformed_validation_data = transformation.apply(
#                     validation_data, is_train=True
#                 )
#                 if cache_data:
#                     transformed_validation_data = Cached(
#                         transformed_validation_data
#                     )

#                 validation_data_loader = self.create_validation_data_loader(
#                     transformed_validation_data,
#                     training_network,
#                 )

#         if from_predictor is not None:
#             training_network.load_state_dict(
#                 from_predictor.network.state_dict()
#             )

#         monitor = "train_loss" if validation_data is None else "val_loss"
#         checkpoint = pl.callbacks.ModelCheckpoint(
#             monitor=monitor, mode="min", verbose=True
#         )

#         custom_callbacks = self.trainer_kwargs.get("callbacks", [])
#         callbacks = [checkpoint] + custom_callbacks
#         trainer_kwargs = {**self.trainer_kwargs, "callbacks": callbacks}
#         trainer = DomainAdaptationTrainer(**trainer_kwargs)
#         training_network.strict_loading = False
#         trainer.fit(
#             model=training_network,
#             train_dataloaders=training_data_loader,
#             val_dataloaders=validation_data_loader,
#             ckpt_path=ckpt_path,
#         )

#         # logger.info(f"Loading best model from {checkpoint.best_model_path}")
#         best_model = training_network.__class__.load_from_checkpoint(
#             checkpoint.best_model_path,
#             strict=False,
#         )

#         return TrainOutput(
#             transformation=transformation,
#             trained_net=best_model,
#             trainer=trainer,
#             predictor=self.create_predictor(transformation, best_model),
#         )


#     def create_predictor(
#         self,
#         transformation: Transformation,
#         module,
#     ) -> PyTorchPredictor:
#         prediction_splitter = self._create_instance_splitter(module, "test")
#         if self.time_feat:
#             return PyTorchPredictor(
#                 input_transform=transformation + prediction_splitter,
#                 input_names=PREDICTION_INPUT_NAMES
#                 + ["past_time_feat", "future_time_feat"],
#                 prediction_net=module,
#                 batch_size=self.batch_size,
#                 prediction_length=self.prediction_length,
#                 device="cuda" if torch.cuda.is_available() else "cpu",
#             )
#         else:
#             return PyTorchPredictor(
#                 input_transform=transformation + prediction_splitter,
#                 input_names=PREDICTION_INPUT_NAMES,
#                 prediction_net=module,
#                 batch_size=self.batch_size,
#                 prediction_length=self.prediction_length,
#                 device="cuda" if torch.cuda.is_available() else "cpu",
#             )
