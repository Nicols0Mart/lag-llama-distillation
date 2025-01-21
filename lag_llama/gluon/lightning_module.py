# Following code is to import from the parent directory (for augmentation)
import inspect
import os
from pathlib import Path
import sys
from setuptools import setup, Extension
from lag_llama.model import losses

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0, parentdir)

import numpy as np
import pytorch_lightning as pl
import torch
import random
from gluonts.core.component import validated
from gluonts.itertools import prod
from gluonts.torch.modules.loss import DistributionLoss, NegativeLogLikelihood
from gluonts.torch.util import repeat_along_dim, take_last

from data.augmentations.freq_mask import freq_mask
from data.augmentations.freq_mix import freq_mix
from data.augmentations.augmentations import (
    ApplyAugmentations,
    Jitter,
    MagnitudeWarp,
    Permutation,
    Rotation,
    Scaling,
    TimeWarp,
    WindowSlice,
    WindowWarp,
)


from gluon_utils.gluon_ts_distributions.implicit_quantile_network import (
    ImplicitQuantileNetworkOutput,
)

from lag_llama.model.module import LagLlamaModel, LLMMistralModel, LagLlamaDAModel
from peft.tuners import lora
from dataclasses import dataclass, field
from functools import reduce
from typing import IO, Any, Callable, Dict, List, Optional, Tuple, Union
from peft import LoraConfig, get_peft_model
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from transformers import Trainer, TrainingArguments
from transformers.data.data_collator import DataCollator
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer import (EvalPrediction, PreTrainedModel,
                                  PreTrainedTokenizerBase, TrainerCallback)
from transformers.trainer_pt_utils import get_parameter_names
from transformers.utils import is_sagemaker_mp_enabled, logging


if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp

logger = logging.get_logger(__name__)



def set_requires_grad(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad = requires_grad

@dataclass
class LoraPlusTrainingArguments(TrainingArguments):
    do_train: bool = field(default=True, metadata={"help": "Whether to run training."})
    do_eval: bool = field(
        default=True, metadata={"help": "Whether to run eval on the dev set."}
    )
    keep_checkpoints: str = field(
        default="all",
        metadata={"help": "keep all, eval, or none checkpoints after end of training"},
    )
    lora_rank: int = field(default=8, metadata={"help": "LoRA rank r"})
    lora_alpha: float = field(default=16, metadata={"help": "LoRA alpha parameter"})
    lora_dropout: float = field(
        default=0.1, metadata={"help": "dropout rate for LoRA modules"}
    )
    target_modules: Optional[str] = field(
        default=None, metadata={"help": "which modules to add LoRA layer to"}
    )
    use_lora: bool = field(
        default=True, metadata={"help": "whether to finetune using LoRA"}
    )
    lora_use_original_init: bool = field(
        default=False,
        metadata={"help": "whether to use the original LoRA initialization"},
    )
    bf16: bool = field(default=False, metadata={"help": "use bfloat16"})
    fp16: bool = field(default=False, metadata={"help": "use bfloat16"})
    gradient_checkpointing: bool = field(
        default=False, metadata={"help": "use gradient checkpointing"}
    )
    loraplus_lr_ratio: Optional[float] = field(
        default=None, metadata={"help": "loraplus learning rate ratio lr_B / lr_A."}
    )
    loraplus_lr_embedding: Optional[float] = field(
        default=1e-6, metadata={"help": "loraplus learning rate for lora embedding layers."}
    )


def get_module(name, opt_model):
    """
    Get the module from the given name in the optimized model.

    Args:
        name (str): The name of the module.
        opt_model: The optimized model.

    Returns:
        module: The module corresponding to the given name in the optimized model.
    """
    parent_idx = 2 if "lora" in name else 1
    module_names = name.split(sep=".")[:-parent_idx]
    module = reduce(getattr, module_names, opt_model)
    return module

def create_loraplus_optimizer(
    opt_model,
    optimizer_cls,
    optimizer_kwargs,
    loraplus_lr_ratio,
    loraplus_lr_embedding=None,
):
    """
    Creates an optimizer for the given model, applying LoRA-specific learning rate adjustments to different parameter groups.
    
    Args:
        opt_model (torch.nn.Module): The model for which the optimizer is being created.
        optimizer_cls (class): The class of the optimizer to be used (e.g., torch.optim.Adam).
        optimizer_kwargs (dict): A dictionary of keyword arguments for the optimizer's initialization.
        loraplus_lr_ratio (float): The learning rate ratio to be applied to LoRA parameters.
        loraplus_lr_embedding (float, optional): A specific learning rate for embedding parameters, with a default value if not provided.
    
    Returns:
        An instance of the specified optimizer class configured with the model's parameters organized into groups with custom learning rates.
    """

    assert loraplus_lr_ratio is not None, "loraplus_lr_ratio must be provided."

    if loraplus_lr_embedding is None:
        loraplus_lr_embedding = 1e-6

    decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    param_groups = {
        "groupA": {},
        "groupB": {},
        "groupB_no_decay": {},
        "embedding": {},
    }

    for name, param in opt_model.named_parameters():
        if not param.requires_grad:
            continue

        module = get_module(name, opt_model)
        if isinstance(module, lora.Embedding):
            param_groups["embedding"][name] = param
        elif "lora_B" in name or param.ndim == 1:
            if name in decay_parameters:
                param_groups["groupB"][name] = param
            else:
                param_groups["groupB_no_decay"][name] = param
        else:
            param_groups["groupA"][name] = param

    assigned_param_groups = ""
    for group in param_groups:
        assigned_param_groups += f"{group}\n {list(param_groups[group].keys())}\n\n"
    logger.debug(assigned_param_groups)

    lr = optimizer_kwargs["lr"]
    weight_decay = optimizer_kwargs.get("weight_decay", 0.0)

    optimizer_grouped_parameters = [
        {
            "params": list(param_groups["groupA"].values()),
            "weight_decay": weight_decay,
            "lr": lr,
        },
        {
            "params": list(param_groups["embedding"].values()),
            "weight_decay": weight_decay,
            "lr": loraplus_lr_embedding,
        },
        {
            "params": list(param_groups["groupB"].values()),
            "weight_decay": weight_decay,
            "lr": lr * loraplus_lr_ratio,
        },
        {
            "params": list(param_groups["groupB_no_decay"].values()),
            "weight_decay": 0.0,
            "lr": lr * loraplus_lr_ratio,
        },
    ]

    optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
    if optimizer_cls.__name__ == "Adam8bit":
        import bitsandbytes

        manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

        skipped = 0
        for module in opt_model.modules():
            if isinstance(module, nn.Embedding):
                skipped += sum(
                    {p.data_ptr(): p.numel() for p in module.parameters()}.values()
                )
                logger.info(f"skipped {module}: {skipped/2**20}M params")
                manager.register_module_override(module, "weight", {"optim_bits": 32})
                logger.debug(f"bitsandbytes: will optimize {module} in fp32")
        logger.info(f"skipped: {skipped/2**20}M params")

    return optimizer


class LagLlamaLightningModule(pl.LightningModule):
    """
    A ``pl.LightningModule`` class that can be used to train a
    ``LagLlamaLightningModule`` with PyTorch Lightning.

    This is a thin layer around a (wrapped) ``LagLlamaLightningModule`` object,
    that exposes the methods to evaluate training and validation loss.

    Parameters
    ----------
    model
        ``LagLlamaLightningModule`` to be trained.
    loss
        Loss function to be used for training,
        default: ``NegativeLogLikelihood()``.
    lr
        Learning rate, default: ``1e-3``.
    weight_decay
        Weight decay regularization parameter, default: ``1e-8``.
    """

    def print_trainable_parameters(self, model):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for name, param in model.named_parameters():
            if "lora" in name:
                print(name, param)
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
        )

    @validated()
    def __init__(
        self,
        model_kwargs: dict,
        context_length: int,
        prediction_length: int,
        loss: DistributionLoss = NegativeLogLikelihood(),
        lr: float = 1e-3,
        weight_decay: float = 1e-8,
        aug_prob: float = 0.1,
        freq_mask_rate: float = 0.1,
        freq_mixing_rate: float = 0.1,
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
        data_id_to_name_map: dict = {},
        use_cosine_annealing_lr: bool = False,
        cosine_annealing_lr_args: dict = {},
        track_loss_per_series: bool = False,
        use_kv_cache: bool = True,
        mistral=False,
        use_feat_dynamic_real=False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.context_length = self.hparams.context_length
        self.prediction_length = self.hparams.prediction_length
        model = LagLlamaModel(**self.hparams.model_kwargs) if not mistral else LLMMistralModel(**self.hparams.model_kwargs)         
        # lora_model = model
        self.model = model
        self.use_feat_dynamic_real = use_feat_dynamic_real
        self.loss = self.hparams.loss
        self.lr = self.hparams.lr
        self.weight_decay = self.hparams.weight_decay
        self.aug_prob = self.hparams.aug_prob
        self.freq_mask_rate = self.hparams.freq_mask_rate
        self.freq_mixing_rate = self.hparams.freq_mixing_rate
        self.jitter_prob = self.hparams.jitter_prob
        self.jitter_sigma = self.hparams.jitter_sigma
        self.scaling_prob = self.hparams.scaling_prob
        self.scaling_sigma = self.hparams.scaling_sigma
        self.rotation_prob = self.hparams.rotation_prob
        self.permutation_prob = self.hparams.permutation_prob
        self.permutation_max_segments = self.hparams.permutation_max_segments
        self.permutation_seg_mode = self.hparams.permutation_seg_mode
        self.magnitude_warp_prob = self.hparams.magnitude_warp_prob
        self.magnitude_warp_sigma = self.hparams.magnitude_warp_sigma
        self.magnitude_warp_knot = self.hparams.magnitude_warp_knot
        self.time_warp_prob = self.hparams.time_warp_prob
        self.time_warp_sigma = self.hparams.time_warp_sigma
        self.time_warp_knot = self.hparams.time_warp_knot
        self.window_slice_prob = self.hparams.window_slice_prob
        self.window_slice_reduce_ratio = self.hparams.window_slice_reduce_ratio
        self.window_warp_prob = self.hparams.window_warp_prob
        self.window_warp_window_ratio = self.hparams.window_warp_window_ratio
        self.window_warp_scales = self.hparams.window_warp_scales
        self.data_id_to_name_map = self.hparams.data_id_to_name_map
        self.use_cosine_annealing_lr = self.hparams.use_cosine_annealing_lr
        self.cosine_annealing_lr_args = self.hparams.cosine_annealing_lr_args
        self.track_loss_per_series = self.hparams.track_loss_per_series

        self.time_feat = self.hparams.model_kwargs["time_feat"]
        # data_id based
        self.train_loss_dict = {}
        self.val_loss_dict = {}
        # item_id based - to be used only in single-dataset mode
        self.train_loss_dict_per_series = {}
        self.val_loss_dict_per_series = {}
        self.use_kv_cache = use_kv_cache
        self.transforms = []
        aug_probs = dict(
            Jitter=dict(prob=self.jitter_prob, sigma=self.jitter_sigma),
            Scaling=dict(prob=self.scaling_prob, sigma=self.scaling_sigma),
            Rotation=dict(prob=self.rotation_prob),
            Permutation=dict(
                prob=self.permutation_prob,
                max_segments=self.permutation_max_segments,
                seg_mode=self.permutation_seg_mode,
            ),
            MagnitudeWarp=dict(
                prob=self.magnitude_warp_prob,
                sigma=self.magnitude_warp_sigma,
                knot=self.magnitude_warp_knot,
            ),
            TimeWarp=dict(
                prob=self.time_warp_prob,
                sigma=self.time_warp_sigma,
                knot=self.time_warp_knot,
            ),
            WindowSlice=dict(
                prob=self.window_slice_prob, reduce_ratio=self.window_slice_reduce_ratio
            ),
            WindowWarp=dict(
                prob=self.window_warp_prob,
                window_ratio=self.window_warp_window_ratio,
                warp_slices=self.window_warp_scales,
            ),
        )
        for aug, params in aug_probs.items():
            if params["prob"] > 0:
                if aug == "Jitter":
                    self.transforms.append(Jitter(params["prob"], params["sigma"]))
                elif aug == "Scaling":
                    self.transforms.append(Scaling(params["prob"], params["sigma"]))
                elif aug == "Rotation":
                    self.transforms.append(Rotation(params["prob"]))
                elif aug == "Permutation":
                    self.transforms.append(
                        Permutation(
                            params["prob"], params["max_segments"], params["seg_mode"]
                        )
                    )
                elif aug == "MagnitudeWarp":
                    self.transforms.append(
                        MagnitudeWarp(params["prob"], params["sigma"], params["knot"])
                    )
                elif aug == "TimeWarp":
                    self.transforms.append(
                        TimeWarp(params["prob"], params["sigma"], params["knot"])
                    )
                elif aug == "WindowSlice":
                    self.transforms.append(
                        WindowSlice(params["prob"], params["reduce_ratio"])
                    )
                elif aug == "WindowWarp":
                    self.transforms.append(
                        WindowWarp(
                            params["prob"],
                            params["window_ratio"],
                            params["warp_slices"],
                        )
                    )

        self.augmentations = ApplyAugmentations(self.transforms)

    # def visualize():
    #     import numpy as np
    #     import matplotlib.pyplot as plt
    #     from matplotlib.animation import FuncAnimation
    #     from scipy.stats import t

    #     # Create a figure and axis
    #     fig, ax = plt.subplots()
    #     ax.set_xlim(-5, 5)
    #     ax.set_ylim(0, 0.5)
    #     ax.set_title('T-Student Distribution')
    #     ax.set_xlabel('x')
    #     ax.set_ylabel('Probability Density')

    #     # Initialize the line object
    #     line, = ax.plot([], [], label='T-Student Distribution')

    #     # Function to initialize the plot
    #     def init():
    #         line.set_data([], [])
    #         return line,

    #     # Function to update the plot for each frame
    #     def update(df):
    #         x = np.linspace(-5, 5, 1000)
    #         pdf = t.pdf(x, df)
    #         line.set_data(x, pdf)
    #         ax.set_title(f'T-Student Distribution (df={df})')
    #         return line,

    #     # Create the animation
    #     ani = FuncAnimation(fig, update, frames=range(1, 21), init_func=init, blit=True)

    #     # Display the animation
    #     plt.show()

    # def vis_distribution(self, samples, dist):
    #     import matplotlib.pyplot as plt
    #     samples_np = samples.numpy()
    #     # Plot a histogram of the samples
    #     plt.figure(figsize=(8, 6))
    #     plt.hist(samples_np, bins=50, density=True, alpha=0.7, color='blue', edgecolor='black')
    #     # Plot the probability density function (PDF)
    #     x = torch.linspace(-5, 5, 1000)
    #     pdf = dist.log_prob(x).exp().numpy()  # Evaluate the PDF at x
    #     plt.plot(x.numpy(), pdf, 'r-', lw=2, label='TStudent PDF')
    #     # Set labels and title
    #     plt.title('Visualizing PyTorch Distribution')
    #     plt.xlabel('x')
    #     plt.ylabel('Probability Density')
    #     plt.legend()
    #     plt.grid(True)
    #     # Show the plot
    #     plt.show()

    # greedy prediction
    def forward(self, *args, **kwargs):
        past_target = kwargs[
            "past_target"
        ]  # (bsz, model.context_length+max(model.lags_seq))
        past_observed_values = kwargs[
            "past_observed_values"
        ]  # (bsz, model.context_length+max(model.lags_seq))
        if self.time_feat:
            past_time_feat = kwargs["past_time_feat"]
            lpls = kwargs["feat_static_real"]
            future_time_feat = kwargs["future_time_feat"]
            repeated_past_time_feat = past_time_feat.repeat_interleave(
                self.model.num_parallel_samples, 0
            )
            repeated_future_time_feat = future_time_feat.repeat_interleave(
                self.model.num_parallel_samples, 0
            )
            lpls = lpls.repeat_interleave(
                self.model.num_parallel_samples, 0
            )

        repeated_past_target = past_target.repeat_interleave(
            self.model.num_parallel_samples, 0
        )  # (bsz* self.model.num_parallel_samples, model.context_length+max(model.lags_seq))
        repeated_past_observed_values = past_observed_values.repeat_interleave(
            self.model.num_parallel_samples, 0
        )  # (bsz* self.model.num_parallel_samples, model.context_length+max(model.lags_seq))

        future_samples = []
        embs = []
        for t in range(self.prediction_length):
            if self.time_feat:
                params, loc, scale = self.model(
                    *args,
                    past_time_feat=repeated_past_time_feat,
                    future_time_feat=repeated_future_time_feat[..., : t + 1, :],
                    past_target=repeated_past_target,
                    past_observed_values=repeated_past_observed_values,
                    use_kv_cache=self.use_kv_cache,
                    lpls=lpls,
                )
                # embs.append(emb.cpu().deqtach())
            else:
                params, loc, scale = self.model(
                    *args,
                    past_time_feat=None,  # repeated_past_time_feat,
                    future_time_feat=None,  # repeated_future_time_feat[..., : t + 1, :],
                    past_target=repeated_past_target,
                    past_observed_values=repeated_past_observed_values,
                    use_kv_cache=self.use_kv_cache,
                )

            sliced_params = [
                p[:, -1:] for p in params
            ]  # Take the last timestep predicted. Each tensor is of shape (#bsz*#parallel_samples, 1)
            distr = self.model.distr_output.distribution(sliced_params, loc, scale)
            sample = distr.sample()  # (#bsz*#parallel_samples, 1)
            future_samples.append(sample)

            repeated_past_target = torch.cat((repeated_past_target, sample), dim=1)
            repeated_past_observed_values = torch.cat(
                (repeated_past_observed_values, torch.ones_like(sample)), dim=1
            )

        self.model.reset_cache()

        concat_future_samples = torch.cat(future_samples, dim=-1)
        return concat_future_samples.reshape(
            (-1, self.model.num_parallel_samples, self.prediction_length)
            + self.model.distr_output.event_shape,
        )


    # train
    def _compute_loss(self, batch, do_not_average=False, return_observed_values=False):
        past_target = batch[
            "past_target"
        ]  # (bsz, model.context_length+max(model.lags_seq))
        past_observed_values = batch[
            "past_observed_values"
        ]  # (bsz, model.context_length+max(model.lags_seq)) with 0s or 1s indicating available (1s) or missing (0s)
        future_target = batch["future_target"]  # (bsz, model.prediction_length)
        future_observed_values = batch[
            "future_observed_values"
        ]  # (bsz, model.prediction_length) with 0s or 1s indicating available (1s) or missing (0s)
        if self.time_feat:
            past_time_feat = batch["past_time_feat"]
            future_time_feat = batch["future_time_feat"]
        else:
            past_time_feat = None
            future_time_feat = None

        extra_dims = len(future_target.shape) - len(past_target.shape)  # usually 0
        extra_shape = future_target.shape[:extra_dims]  # shape remains the same

        repeats = prod(extra_shape)  # usually 1
        past_target = repeat_along_dim(
            past_target, 0, repeats
        )  # (bsz, model.context_length+max(model.lags_seq))
        past_observed_values = repeat_along_dim(
            past_observed_values, 0, repeats
        )  # (bsz, model.context_length+max(model.lags_seq))

        future_target_reshaped = future_target.reshape(
            -1,
            *future_target.shape[extra_dims + 1 :],
        )  # (bsz, model.prediction_length)
        future_observed_reshaped = future_observed_values.reshape(
            -1,
            *future_observed_values.shape[extra_dims + 1 :],
        )  # (bsz, model.prediction_length)

        distr_args, loc, scale = self.model(
            past_target=past_target,
            past_observed_values=past_observed_values,
            past_time_feat=past_time_feat,
            future_time_feat=future_time_feat,
            future_target=future_target_reshaped,
            lpls=batch["feat_static_real"],
        )  # distr_args is a tuple with two tensors of shape (bsz, context_length+pred_len-1)
        context_target = take_last(
            past_target, dim=-1 if not self.use_feat_dynamic_real else -2, num=self.context_length - 1
        )  # (bsz, context_length-1) # Basically removes the first value since it cannot be predicted
        target = torch.cat(
            (context_target, future_target_reshaped),
            dim=1,
        )  # (bsz, context_length-1+pred_len) # values that can be predicted
        context_observed = take_last(
            past_observed_values, dim=-1 if not self.use_feat_dynamic_real else -2, num=self.context_length - 1
        )  # same as context_target, but for observed_values tensor
        observed_values = torch.cat(
            (context_observed, future_observed_reshaped), dim=1
        )  # same as target but for observed_values tensor

        if type(self.model.distr_output) == ImplicitQuantileNetworkOutput:
            if not do_not_average:
                loss = (
                    self.model.distr_output.loss(target, distr_args, loc, scale)
                    * observed_values
                ).sum() / observed_values.sum().clamp_min(1.0)
            else:
                loss = (
                    self.model.distr_output.loss(target, distr_args, loc, scale)
                    * observed_values
                )
        else:
            if torch.any(torch.isnan(distr_args[0])) or torch.any(torch.isnan(distr_args[1])):
                print(
                    batch[
                        "past_target"
                    ]
                )
            distr = self.model.distr_output.distribution(
                distr_args, loc=loc, scale=scale
            )  # an object representing a distribution with the specified parameters. We need this to compute the NLL loss.
            if not do_not_average:
                loss = (
                    self.loss(distr, target) * observed_values
                ).sum() / observed_values.sum().clamp_min(1.0)
            else:
                loss = self.loss(distr, target) * observed_values

        if not return_observed_values:
            return loss
        else:
            return loss, observed_values

    def training_step(self, batch, batch_idx: int):  # type: ignore
        """
        Execute training step.
        """
        if random.random() < self.aug_prob:
            # Freq mix and Freq mask have separate functions
            if self.freq_mask_rate > 0:
                batch["past_target"], batch["future_target"] = freq_mask(
                    batch["past_target"],
                    batch["future_target"],
                    rate=self.freq_mask_rate,
                )
            if self.freq_mixing_rate:
                batch["past_target"], batch["future_target"] = freq_mix(
                    batch["past_target"],
                    batch["future_target"],
                    rate=self.freq_mixing_rate,
                )
            # Other augmentation
            if len(self.transforms):
                batch["past_target"], batch["future_target"] = self.augmentations(
                    batch["past_target"], batch["future_target"]
                )

        train_loss_per_sample, observed_values = self._compute_loss(
            batch, do_not_average=True, return_observed_values=True
        )
        for idx, data_id in enumerate(batch["data_id"]):
            if data_id not in self.train_loss_dict:
                self.train_loss_dict[data_id.item()] = []
            self.train_loss_dict[data_id.item()].append(
                (
                    train_loss_per_sample[idx].sum()
                    / observed_values[idx].sum().clamp_min(1.0)
                ).item()
            )

        if self.track_loss_per_series:
            for idx, item_id in enumerate(batch["item_id"]):
                if item_id not in self.train_loss_dict_per_series:
                    self.train_loss_dict_per_series[item_id.item()] = []
                self.train_loss_dict_per_series[item_id.item()].append(
                    (
                        train_loss_per_sample[idx].sum()
                        / observed_values[idx].sum().clamp_min(1.0)
                    ).item()
                )

        train_loss_avg = train_loss_per_sample.sum() / observed_values.sum().clamp_min(
            1.0
        )
        self.log(
            "train_loss", train_loss_avg, on_epoch=True, on_step=False, prog_bar=False
        )
        return train_loss_avg

    def on_train_epoch_end(self):
        # Log all losses
        for key, value in self.train_loss_dict.items():
            loss_avg = np.mean(value)
            self.log(
                f"train_loss_avg_per_train_dataset/{key}",
                loss_avg,
                on_epoch=True,
                on_step=False,
                prog_bar=False,
            )

        if self.track_loss_per_series:
            # Log all losses
            for key, value in self.train_loss_dict_per_series.items():
                loss_avg = np.mean(value)
                self.log(
                    f"train_loss_avg_per_train_series/{key}",
                    loss_avg,
                    on_epoch=True,
                    on_step=False,
                    prog_bar=False,
                )

        # Reset loss_dict
        self.train_loss_dict = {}
        self.train_loss_dict_per_series = {}

    def validation_step(self, batch, batch_idx: int):  # type: ignore
        """
        Execute validation step.
        """
        val_loss_per_sample, observed_values = self._compute_loss(
            batch, do_not_average=True, return_observed_values=True
        )

        val_loss_without_test_set = 0.0
        for idx, data_id in enumerate(batch["data_id"]):
            if data_id not in self.val_loss_dict:
                self.val_loss_dict[data_id.item()] = []
            self.val_loss_dict[data_id.item()].append(
                (
                    val_loss_per_sample[idx].sum()
                    / observed_values[idx].sum().clamp_min(1.0)
                ).item()
            )
            if data_id >= 0:
                val_loss_without_test_set += val_loss_per_sample[idx].sum()

        if self.track_loss_per_series:
            for idx, item_id in enumerate(batch["item_id"]):
                if item_id not in self.val_loss_dict_per_series:
                    self.val_loss_dict_per_series[item_id.item()] = []
                self.val_loss_dict_per_series[item_id.item()].append(
                    (
                        val_loss_per_sample[idx].sum()
                        / observed_values[idx].sum().clamp_min(1.0)
                    ).item()
                )

        val_loss_avg = val_loss_without_test_set / observed_values.sum().clamp_min(1.0)
        self.log("val_loss", val_loss_avg, on_epoch=True, on_step=False, prog_bar=False)
        return val_loss_avg

    def on_validation_epoch_end(self):
        # Log all losses
        for key, value in self.val_loss_dict.items():
            loss_avg = np.mean(value)
            if key >= 0:
                self.log(
                    f"val_loss_avg_per_train_dataset/{key}",
                    loss_avg,
                    on_epoch=True,
                    on_step=False,
                    prog_bar=False,
                )
            else:
                self.log(
                    f"val_loss_avg_per_test_dataset/{key}",
                    loss_avg,
                    on_epoch=True,
                    on_step=False,
                    prog_bar=False,
                )

        if self.track_loss_per_series:
            # Log all losses
            for key, value in self.val_loss_dict_per_series.items():
                loss_avg = np.mean(value)
                self.log(
                    f"val_loss_avg_per_train_series/{key}",
                    loss_avg,
                    on_epoch=True,
                    on_step=False,
                    prog_bar=False,
                )

        # Reset loss_dict
        self.val_loss_dict = {}
        self.val_loss_dict_per_series = {}

    def configure_optimizers(self):
        """
        Returns the optimizer to use.
        """
        # optimizer = torch.optim.Adam(
        #     self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        # )
        optimizer = create_loraplus_optimizer(
            self.model,
            optimizer_cls=torch.optim.Adam,
            optimizer_kwargs={"lr": self.lr, "weight_decay": self.weight_decay},
            loraplus_lr_embedding=1e-06,
            loraplus_lr_ratio=1.25,
        )

        if self.use_cosine_annealing_lr:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, **self.cosine_annealing_lr_args, verbose=True
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        else:
            return optimizer


class LagLlamaDALightningModule(pl.LightningModule):
    """
    A ``pl.LightningModule`` class that can be used to train a
    ``LagLlamaLightningModule`` with PyTorch Lightning.

    This is a thin layer around a (wrapped) ``LagLlamaLightningModule`` object,
    that exposes the methods to evaluate training and validation loss.

    Parameters
    ----------
    model
        ``LagLlamaLightningModule`` to be trained.
    loss
        Loss function to be used for training,
        default: ``NegativeLogLikelihood()``.
    lr
        Learning rate, default: ``1e-3``.
    weight_decay
        Weight decay regularization parameter, default: ``1e-8``.
    """

    def print_trainable_parameters(self, model):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for name, param in model.named_parameters():
            if "lora" in name:
                print(name, param)
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
        )

    @validated()
    def __init__(
        self,
        model_kwargs: dict,
        context_length: int,
        prediction_length: int,
        loss: DistributionLoss = NegativeLogLikelihood(),
        lr: float = 1e-3,
        weight_decay: float = 1e-8,
        aug_prob: float = 0.1,
        freq_mask_rate: float = 0.1,
        freq_mixing_rate: float = 0.1,
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
        data_id_to_name_map: dict = {},
        use_cosine_annealing_lr: bool = False,
        cosine_annealing_lr_args: dict = {},
        track_loss_per_series: bool = False,
        use_kv_cache: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.context_length = self.hparams.context_length
        self.prediction_length = self.hparams.prediction_length
        model = LagLlamaDAModel(**self.hparams.model_kwargs)        
        # lora_model = model
        self.model: LagLlamaDAModel = model
        self.loss = self.hparams.loss
        self.lr = self.hparams.lr
        self.weight_decay = self.hparams.weight_decay
        self.aug_prob = self.hparams.aug_prob
        self.freq_mask_rate = self.hparams.freq_mask_rate
        self.freq_mixing_rate = self.hparams.freq_mixing_rate
        self.jitter_prob = self.hparams.jitter_prob
        self.jitter_sigma = self.hparams.jitter_sigma
        self.scaling_prob = self.hparams.scaling_prob
        self.scaling_sigma = self.hparams.scaling_sigma
        self.rotation_prob = self.hparams.rotation_prob
        self.permutation_prob = self.hparams.permutation_prob
        self.permutation_max_segments = self.hparams.permutation_max_segments
        self.permutation_seg_mode = self.hparams.permutation_seg_mode
        self.magnitude_warp_prob = self.hparams.magnitude_warp_prob
        self.magnitude_warp_sigma = self.hparams.magnitude_warp_sigma
        self.magnitude_warp_knot = self.hparams.magnitude_warp_knot
        self.time_warp_prob = self.hparams.time_warp_prob
        self.time_warp_sigma = self.hparams.time_warp_sigma
        self.time_warp_knot = self.hparams.time_warp_knot
        self.window_slice_prob = self.hparams.window_slice_prob
        self.window_slice_reduce_ratio = self.hparams.window_slice_reduce_ratio
        self.window_warp_prob = self.hparams.window_warp_prob
        self.window_warp_window_ratio = self.hparams.window_warp_window_ratio
        self.window_warp_scales = self.hparams.window_warp_scales
        self.data_id_to_name_map = self.hparams.data_id_to_name_map
        self.use_cosine_annealing_lr = self.hparams.use_cosine_annealing_lr
        self.cosine_annealing_lr_args = self.hparams.cosine_annealing_lr_args
        self.track_loss_per_series = self.hparams.track_loss_per_series

        self.time_feat = self.hparams.model_kwargs["time_feat"]
        # data_id based
        self.train_loss_dict = {}
        self.val_loss_dict = {}
        # item_id based - to be used only in single-dataset mode
        self.train_loss_dict_per_series = {}
        self.val_loss_dict_per_series = {}
        self.use_kv_cache = use_kv_cache
        self.transforms = []
        aug_probs = dict(
            Jitter=dict(prob=self.jitter_prob, sigma=self.jitter_sigma),
            Scaling=dict(prob=self.scaling_prob, sigma=self.scaling_sigma),
            Rotation=dict(prob=self.rotation_prob),
            Permutation=dict(
                prob=self.permutation_prob,
                max_segments=self.permutation_max_segments,
                seg_mode=self.permutation_seg_mode,
            ),
            MagnitudeWarp=dict(
                prob=self.magnitude_warp_prob,
                sigma=self.magnitude_warp_sigma,
                knot=self.magnitude_warp_knot,
            ),
            TimeWarp=dict(
                prob=self.time_warp_prob,
                sigma=self.time_warp_sigma,
                knot=self.time_warp_knot,
            ),
            WindowSlice=dict(
                prob=self.window_slice_prob, reduce_ratio=self.window_slice_reduce_ratio
            ),
            WindowWarp=dict(
                prob=self.window_warp_prob,
                window_ratio=self.window_warp_window_ratio,
                warp_slices=self.window_warp_scales,
            ),
        )
        for aug, params in aug_probs.items():
            if params["prob"] > 0:
                if aug == "Jitter":
                    self.transforms.append(Jitter(params["prob"], params["sigma"]))
                elif aug == "Scaling":
                    self.transforms.append(Scaling(params["prob"], params["sigma"]))
                elif aug == "Rotation":
                    self.transforms.append(Rotation(params["prob"]))
                elif aug == "Permutation":
                    self.transforms.append(
                        Permutation(
                            params["prob"], params["max_segments"], params["seg_mode"]
                        )
                    )
                elif aug == "MagnitudeWarp":
                    self.transforms.append(
                        MagnitudeWarp(params["prob"], params["sigma"], params["knot"])
                    )
                elif aug == "TimeWarp":
                    self.transforms.append(
                        TimeWarp(params["prob"], params["sigma"], params["knot"])
                    )
                elif aug == "WindowSlice":
                    self.transforms.append(
                        WindowSlice(params["prob"], params["reduce_ratio"])
                    )
                elif aug == "WindowWarp":
                    self.transforms.append(
                        WindowWarp(
                            params["prob"],
                            params["window_ratio"],
                            params["warp_slices"],
                        )
                    )

        self.augmentations = ApplyAugmentations(self.transforms)

    # greedy prediction
    def forward(self, *args, **kwargs):
        past_target = kwargs[
            "past_target"
        ]  # (bsz, model.context_length+max(model.lags_seq))
        past_observed_values = kwargs[
            "past_observed_values"
        ]  # (bsz, model.context_length+max(model.lags_seq))
        if self.time_feat:
            past_time_feat = kwargs["past_time_feat"]
            future_time_feat = kwargs["future_time_feat"]
            repeated_past_time_feat = past_time_feat.repeat_interleave(
                self.model.num_parallel_samples, 0
            )
            repeated_future_time_feat = future_time_feat.repeat_interleave(
                self.model.num_parallel_samples, 0
            )

        repeated_past_target = past_target.repeat_interleave(
            self.model.num_parallel_samples, 0
        )  # (bsz* self.model.num_parallel_samples, model.context_length+max(model.lags_seq))
        repeated_past_observed_values = past_observed_values.repeat_interleave(
            self.model.num_parallel_samples, 0
        )  # (bsz* self.model.num_parallel_samples, model.context_length+max(model.lags_seq))

        future_samples = []
        for t in range(self.prediction_length):
            if self.time_feat:
                params, loc, scale, domain_logits = self.model(
                    *args,
                    past_time_feat=repeated_past_time_feat,
                    future_time_feat=repeated_future_time_feat[..., : t + 1, :],
                    past_target=repeated_past_target,
                    past_observed_values=repeated_past_observed_values,
                    use_kv_cache=self.use_kv_cache,
                )
            else:
                params, loc, scale = self.model(
                    *args,
                    past_time_feat=None,  # repeated_past_time_feat,
                    future_time_feat=None,  # repeated_future_time_feat[..., : t + 1, :],
                    past_target=repeated_past_target,
                    past_observed_values=repeated_past_observed_values,
                    use_kv_cache=self.use_kv_cache,
                )

            sliced_params = [
                p[:, -1:] for p in params
            ]  # Take the last timestep predicted. Each tensor is of shape (#bsz*#parallel_samples, 1)
            distr = self.model.distr_output.distribution(sliced_params, loc, scale)
            sample = distr.sample()  # (#bsz*#parallel_samples, 1)
            future_samples.append(sample)

            repeated_past_target = torch.cat((repeated_past_target, sample), dim=1)
            repeated_past_observed_values = torch.cat(
                (repeated_past_observed_values, torch.ones_like(sample)), dim=1
            )

        self.model.reset_cache()

        concat_future_samples = torch.cat(future_samples, dim=-1)
        return concat_future_samples.reshape(
            (-1, self.model.num_parallel_samples, self.prediction_length)
            + self.model.distr_output.event_shape,
        )

    def model_forward(self, batch):
        past_target = batch[
            "past_target"
        ]  # (bsz, model.context_length+max(model.lags_seq))
        past_observed_values = batch[
            "past_observed_values"
        ]  # (bsz, model.context_length+max(model.lags_seq)) with 0s or 1s indicating available (1s) or missing (0s)
        future_target = batch["future_target"]  # (bsz, model.prediction_length)
        future_observed_values = batch[
            "future_observed_values"
        ]  # (bsz, model.prediction_length) with 0s or 1s indicating available (1s) or missing (0s)
        if self.time_feat:
            past_time_feat = batch["past_time_feat"]
            future_time_feat = batch["future_time_feat"]
        else:
            past_time_feat = None
            future_time_feat = None

        extra_dims = len(future_target.shape) - len(past_target.shape)  # usually 0
        extra_shape = future_target.shape[:extra_dims]  # shape remains the same

        repeats = prod(extra_shape)  # usually 1
        past_target = repeat_along_dim(
            past_target, 0, repeats
        )  # (bsz, model.context_length+max(model.lags_seq))
        past_observed_values = repeat_along_dim(
            past_observed_values, 0, repeats
        )  # (bsz, model.context_length+max(model.lags_seq))

        future_target_reshaped = future_target.reshape(
            -1,
            *future_target.shape[extra_dims + 1 :],
        )  # (bsz, model.prediction_length)
        future_observed_reshaped = future_observed_values.reshape(
            -1,
            *future_observed_values.shape[extra_dims + 1 :],
        )  # (bsz, model.prediction_length)

        distr_args, loc, scale, domain_logit = self.model(
            past_target=past_target,
            past_observed_values=past_observed_values,
            past_time_feat=past_time_feat,
            future_time_feat=future_time_feat,
            future_target=future_target_reshaped,
        )  # distr_args is a tuple with two tensors of shape (bsz, context_length+pred_len-1)
        context_target = take_last(
            past_target, dim=-1, num=self.context_length - 1
        )  # (bsz, context_length-1) # Basically removes the first value since it cannot be predicted
        target = torch.cat(
            (context_target, future_target_reshaped),
            dim=1,
        )  # (bsz, context_length-1+pred_len) # values that can be predicted
        context_observed = take_last(
            past_observed_values, dim=-1, num=self.context_length - 1
        )  # same as context_target, but for observed_values tensor
        observed_values = torch.cat(
            (context_observed, future_observed_reshaped), dim=1
        )  # same as target but for observed_values tensor
        return target, distr_args, loc, scale, domain_logit, observed_values

    # train
    def _compute_loss(self, batches, do_not_average=False, return_observed_values=False):
        batch, batch_target = batches
        target, distr_args, loc, scale, domain_logit, observed_values = self.model_forward(batch)
        if type(self.model.distr_output) == ImplicitQuantileNetworkOutput:
            if not do_not_average:
                loss = (
                    self.model.distr_output.loss(target, distr_args, loc, scale)
                    * observed_values
                ).sum() / observed_values.sum().clamp_min(1.0)
            else:
                loss = (
                    self.model.distr_output.loss(target, distr_args, loc, scale)
                    * observed_values
                )
        else:
            distr = self.model.distr_output.distribution(
                distr_args, loc=loc, scale=scale
            )  # an object representing a distribution with the specified parameters. We need this to compute the NLL loss.
            if not do_not_average:
                loss = (
                    self.loss(distr, target) * observed_values
                ).sum() / observed_values.sum().clamp_min(1.0)
            else:
                loss = self.loss(distr, target) * observed_values
        _, dok_src = losses.cross_entropy_logits(domain_logit, torch.zeros(batch["past_observed_values"].size(0)))
        ###################Target Domain#####################
        past_target = batch_target[
            "past_target"
        ]  # (bsz, model.context_length+max(model.lags_seq))
        past_observed_values = batch_target[
            "past_observed_values"
        ]  # (bsz, model.context_length+max(model.lags_seq)) with 0s or 1s indicating available (1s) or missing (0s)
        future_target = batch_target["future_target"]  # (bsz, model.prediction_length)
        future_observed_values = batch_target[
            "future_observed_values"
        ]  # (bsz, model.prediction_length) with 0s or 1s indicating available (1s) or missing (0s)
        if self.time_feat:
            past_time_feat = batch_target["past_time_feat"]
            future_time_feat = batch_target["future_time_feat"]
        else:
            past_time_feat = None
            future_time_feat = None

        extra_dims = len(future_target.shape) - len(past_target.shape)  # usually 0
        extra_shape = future_target.shape[:extra_dims]  # shape remains the same

        repeats = prod(extra_shape)  # usually 1
        past_target = repeat_along_dim(
            past_target, 0, repeats
        )  # (bsz, model.context_length+max(model.lags_seq))
        past_observed_values = repeat_along_dim(
            past_observed_values, 0, repeats
        )  # (bsz, model.context_length+max(model.lags_seq))

        future_target_reshaped = future_target.reshape(
            -1,
            *future_target.shape[extra_dims + 1 :],
        )  # (bsz, model.prediction_length)
        future_observed_reshaped = future_observed_values.reshape(
            -1,
            *future_observed_values.shape[extra_dims + 1 :],
        )  # (bsz, model.prediction_length)

        distr_args_target, loc_target, scale_target, domain_logit_target = self.model(
            past_target=past_target,
            past_observed_values=past_observed_values,
            past_time_feat=past_time_feat,
            future_time_feat=future_time_feat,
            future_target=future_target_reshaped,
        )  # distr_args is a tuple with two tensors of shape (bsz, context_length+pred_len-1)
        context_target = take_last(
            past_target, dim=-1, num=self.context_length - 1
        )  # (bsz, context_length-1) # Basically removes the first value since it cannot be predicted
        target_target = torch.cat(
            (context_target, future_target_reshaped),
            dim=1,
        )  # (bsz, context_length-1+pred_len) # values that can be predicted
        context_observed = take_last(
            past_observed_values, dim=-1, num=self.context_length - 1
        )  # same as context_target, but for observed_values tensor
        observed_values = torch.cat(
            (context_observed, future_observed_reshaped), dim=1
        )  # same as target but for observed_values tensor

        if type(self.model.distr_output) == ImplicitQuantileNetworkOutput:
            if not do_not_average:
                loss_tgt = (
                    self.model.distr_output.loss(target_target, distr_args_target, loc_target, scale_target)
                    * observed_values
                ).sum() / observed_values.sum().clamp_min(1.0)
            else:
                loss_tgt = (
                    self.model.distr_output.loss(target_target, distr_args_target, loc_target, scale_target)
                    * observed_values
                )
        else:
            distr_tgt = self.model.distr_output.distribution(
                distr_args_target, loc=loc_target, scale=scale_target
            )  # an object representing a distribution with the specified parameters. We need this to compute the NLL loss.
            if not do_not_average:
                loss_tgt = (
                    self.loss(distr_tgt, target_target) * observed_values
                ).sum() / observed_values.sum().clamp_min(1.0)
            else:
                loss_tgt = self.loss(distr, target) * observed_values

        _, dok_tgt = losses.cross_entropy_logits(domain_logit_target, torch.ones(len(domain_logit_target)))
        wasserstein_distance = domain_logit.mean() - 1 * domain_logit_target.mean()
        loss = (loss , loss_tgt , dok_tgt , dok_src , wasserstein_distance)
        if not return_observed_values:
            return loss
        else:
            return loss, observed_values

    def training_step(self, batch, batch_idx: int):  # type: ignore
        """
        Execute training step.
        """
        if random.random() < self.aug_prob:
            # Freq mix and Freq mask have separate functions
            if self.freq_mask_rate > 0:
                batch["past_target"], batch["future_target"] = freq_mask(
                    batch["past_target"],
                    batch["future_target"],
                    rate=self.freq_mask_rate,
                )
            if self.freq_mixing_rate:
                batch["past_target"], batch["future_target"] = freq_mix(
                    batch["past_target"],
                    batch["future_target"],
                    rate=self.freq_mixing_rate,
                )
            # Other augmentation
            if len(self.transforms):
                batch["past_target"], batch["future_target"] = self.augmentations(
                    batch["past_target"], batch["future_target"]
                )
        
        (loss , loss_tgt , dok_tgt , dok_src , wasserstein_distance), observed_values = self._compute_loss(
            batch, do_not_average=True, return_observed_values=True
        )
        self.log(
            "train_src_loss", loss, on_epoch=True, on_step=False, prog_bar=False
        )
        self.log(
            "train_tgt_loss", loss_tgt, on_epoch=True, on_step=False, prog_bar=False
        )
        self.log(
            "train_critic_src_loss", dok_src, on_epoch=True, on_step=False, prog_bar=False
        )
        self.log(
            "train_critic_tgt_loss", dok_tgt, on_epoch=True, on_step=False, prog_bar=False
        )
        self.log(
            "train_wasserstein_loss", wasserstein_distance, on_epoch=True, on_step=False, prog_bar=False
        )
        train_loss_per_sample = sum(loss , loss_tgt , dok_tgt , dok_src , wasserstein_distance)
        for idx, data_id in enumerate(batch["data_id"]):
            if data_id not in self.train_loss_dict:
                self.train_loss_dict[data_id.item()] = []
            self.train_loss_dict[data_id.item()].append(
                (
                    train_loss_per_sample[idx].sum()
                    / observed_values[idx].sum().clamp_min(1.0)
                ).item()
            )

        if self.track_loss_per_series:
            for idx, item_id in enumerate(batch["item_id"]):
                if item_id not in self.train_loss_dict_per_series:
                    self.train_loss_dict_per_series[item_id.item()] = []
                self.train_loss_dict_per_series[item_id.item()].append(
                    (
                        train_loss_per_sample[idx].sum()
                        / observed_values[idx].sum().clamp_min(1.0)
                    ).item()
                )

        train_loss_avg = train_loss_per_sample.sum() / observed_values.sum().clamp_min(
            1.0
        )
        self.log(
            "train_loss", train_loss_avg, on_epoch=True, on_step=False, prog_bar=False
        )
        return train_loss_avg
    
    def critic_update_steps(self, batches):
        if self.current_epoch < 70:
            return

        set_requires_grad(self.feat, requires_grad=False)
        set_requires_grad(self.domain_classifier, requires_grad=True)

        batch, batch_tgt = batches
        with torch.no_grad():
            _, _, _, _, domain_logit, _ = self.model_forward(batch)
            h_s = domain_logit
            _, _, _, _, tgt_logit, _ = self.model_forward(batch_tgt)
            h_t = tgt_logit
        for _ in range(self._k_critic):
            gp = losses.gradient_penalty(self.model.domain_classifier, h_s, h_t)

            critic_s = self.model.domain_classifier(h_s)
            critic_t = self.model.domain_classifier(h_t)
            wasserstein_distance = (
                critic_s.mean() - (1) * critic_t.mean()
            )

            critic_cost = -wasserstein_distance + 10 * gp

            self.critic_opt.zero_grad()
            critic_cost.backward()
            self.critic_opt.step()
            if self.critic_sched:
                self.critic_sched.step()

        set_requires_grad(self.model.feature_extractor, requires_grad=True)
        set_requires_grad(self.model.domain_classifier, requires_grad=False)

    def on_train_epoch_end(self):
        # Log all losses
        for key, value in self.train_loss_dict.items():
            loss_avg = np.mean(value)
            self.log(
                f"train_loss_avg_per_train_dataset/{key}",
                loss_avg,
                on_epoch=True,
                on_step=False,
                prog_bar=False,
            )

        if self.track_loss_per_series:
            # Log all losses
            for key, value in self.train_loss_dict_per_series.items():
                loss_avg = np.mean(value)
                self.log(
                    f"train_loss_avg_per_train_series/{key}",
                    loss_avg,
                    on_epoch=True,
                    on_step=False,
                    prog_bar=False,
                )

        # Reset loss_dict
        self.train_loss_dict = {}
        self.train_loss_dict_per_series = {}

    def validation_step(self, batch, batch_idx: int):  # type: ignore
        """
        Execute validation step.
        """
        val_loss_per_sample, observed_values = self._compute_loss(
            batch, do_not_average=True, return_observed_values=True
        )

        val_loss_without_test_set = 0.0
        for idx, data_id in enumerate(batch["data_id"]):
            if data_id not in self.val_loss_dict:
                self.val_loss_dict[data_id.item()] = []
            self.val_loss_dict[data_id.item()].append(
                (
                    val_loss_per_sample[idx].sum()
                    / observed_values[idx].sum().clamp_min(1.0)
                ).item()
            )
            if data_id >= 0:
                val_loss_without_test_set += val_loss_per_sample[idx].sum()

        if self.track_loss_per_series:
            for idx, item_id in enumerate(batch["item_id"]):
                if item_id not in self.val_loss_dict_per_series:
                    self.val_loss_dict_per_series[item_id.item()] = []
                self.val_loss_dict_per_series[item_id.item()].append(
                    (
                        val_loss_per_sample[idx].sum()
                        / observed_values[idx].sum().clamp_min(1.0)
                    ).item()
                )

        val_loss_avg = val_loss_without_test_set / observed_values.sum().clamp_min(1.0)
        self.log("val_loss", val_loss_avg, on_epoch=True, on_step=False, prog_bar=False)
        return val_loss_avg

    def on_validation_epoch_end(self):
        # Log all losses
        for key, value in self.val_loss_dict.items():
            loss_avg = np.mean(value)
            if key >= 0:
                self.log(
                    f"val_loss_avg_per_train_dataset/{key}",
                    loss_avg,
                    on_epoch=True,
                    on_step=False,
                    prog_bar=False,
                )
            else:
                self.log(
                    f"val_loss_avg_per_test_dataset/{key}",
                    loss_avg,
                    on_epoch=True,
                    on_step=False,
                    prog_bar=False,
                )

        if self.track_loss_per_series:
            # Log all losses
            for key, value in self.val_loss_dict_per_series.items():
                loss_avg = np.mean(value)
                self.log(
                    f"val_loss_avg_per_train_series/{key}",
                    loss_avg,
                    on_epoch=True,
                    on_step=False,
                    prog_bar=False,
                )

        # Reset loss_dict
        self.val_loss_dict = {}
        self.val_loss_dict_per_series = {}

    def configure_optimizers(self):
        """
        Returns the optimizer to use.
        """
        # optimizer = torch.optim.Adam(
        #     self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        # )
        optimizer = create_loraplus_optimizer(
            self.model,
            optimizer_cls=torch.optim.Adam,
            optimizer_kwargs={"lr": self.lr, "weight_decay": self.weight_decay},
            loraplus_lr_embedding=1e-06,
            loraplus_lr_ratio=1.25,
        )

        if self.use_cosine_annealing_lr:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, **self.cosine_annealing_lr_args, verbose=True
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        else:
            return optimizer
