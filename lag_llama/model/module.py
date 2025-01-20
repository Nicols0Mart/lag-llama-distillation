import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Iterable

import torch
from fairscale.nn.model_parallel.initialize import initialize_model_parallel
from gluonts.torch.scaler import MeanScaler, NOPScaler, StdScaler
from gluonts.torch.util import lagged_sequence_values, unsqueeze_expand
from torch import nn
from torch.nn import functional as F
from gluon_utils.scalers.robust_scaler import RobustScaler
from gluonts.torch.distributions import DistributionOutput
from ..gluon.mistral import MistralAttention, MistralMLP, MistralSdpaAttention, MistralRMSNorm, config_mist

from .llama3 import TransformerBlock, FeedForward as FeedForward3, RMSNorm as RMSNorm3

INTERMEDIATE_REPR = 127

@dataclass
class LTSMConfig:
    # feature_size: int = 3 + 6 + 2  # target + loc + scale + time features
    feature_size: int = 3 + 6  # target + loc + scale + time features
    block_size: int = 2048
    n_layer: int = 32
    n_head: int = 32
    n_embd_per_head: int = 128
    rope_scaling: Optional[dict] = None
    dropout: float = 0.0


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000

    max_batch_size: int = 32
    max_seq_len: int = 2048


def llama2_to_3_adaptor(llama2_config: LTSMConfig) -> ModelArgs:
    return ModelArgs(
        dim=llama2_config.n_embd_per_head * llama2_config.n_head,
        n_layers=llama2_config.n_layer,
        n_heads=llama2_config.n_head,
        multiple_of=256,
        ffn_dim_multiplier=None,
        norm_eps=1e-5,
        rope_theta=500000,
        max_batch_size=32,
        max_seq_len=2048,
    )


def scaled_dot_product_attention(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None
) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(
            diagonal=0
        )
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_mask.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value


class Block(nn.Module):
    def __init__(self, config: LTSMConfig) -> None:
        super().__init__()
        self.rms_1 = RMSNorm(config.n_embd_per_head * config.n_head)
        self.attn = CausalSelfAttention(config)
        self.rms_2 = RMSNorm(config.n_embd_per_head * config.n_head)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor, use_kv_cache: bool) -> torch.Tensor:
        x = x + self.attn(self.rms_1(x), use_kv_cache)
        y = x + self.mlp(self.rms_2(x))
        return y
    


class LlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False
        )
        self.register_buffer(
            "sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False
        )

    def forward(self, device, dtype, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=device, dtype=dtype)

        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=dtype),
        )


class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
    ):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )
        t = t / self.scaling_factor

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False
        )
        self.register_buffer(
            "sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False
        )


class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
    ):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings)
                - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (
                base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False
        )
        self.register_buffer(
            "sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False
        )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class CausalSelfAttention(nn.Module):
    def __init__(self, config: LTSMConfig) -> None:
        super().__init__()
        # query projections for all heads, but in a batch
        self.q_proj = nn.Linear(
            config.n_embd_per_head * config.n_head,
            config.n_embd_per_head * config.n_head,
            bias=False,
        )
        # key, value projections
        self.kv_proj = nn.Linear(
            config.n_embd_per_head * config.n_head,
            2 * config.n_embd_per_head * config.n_head,
            bias=False,
        )
        # output projection
        self.c_proj = nn.Linear(
            config.n_embd_per_head * config.n_head,
            config.n_embd_per_head * config.n_head,
            bias=False,
        )

        self.n_head = config.n_head
        self.n_embd_per_head = config.n_embd_per_head
        self.block_size = config.block_size
        self.dropout = config.dropout

        self.rope_scaling = config.rope_scaling
        self._rope_scaling_validation()

        self._init_rope()
        self.kv_cache = None

    def _init_rope(self):
        if self.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.n_embd_per_head, max_position_embeddings=self.block_size
            )
        else:
            scaling_type = self.rope_scaling["type"]
            scaling_factor = self.rope_scaling["factor"]
            if scaling_type == "nope":
                self.rotary_emb = None
            elif scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.n_embd_per_head,
                    max_position_embeddings=self.block_size,
                    scaling_factor=scaling_factor,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.n_embd_per_head,
                    max_position_embeddings=self.block_size,
                    scaling_factor=scaling_factor,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _rope_scaling_validation(self):
        """
        Validate the `rope_scaling` configuration.
        """
        if self.rope_scaling is None:
            return

        if not isinstance(self.rope_scaling, dict) or len(self.rope_scaling) != 2:
            raise ValueError(
                "`rope_scaling` must be a dictionary with with two fields, `name` and `factor`, "
                f"got {self.rope_scaling}"
            )
        rope_scaling_type = self.rope_scaling.get("type", None)
        rope_scaling_factor = self.rope_scaling.get("factor", None)
        if rope_scaling_type is None or rope_scaling_type not in [
            "linear",
            "dynamic",
            "nope",
        ]:
            raise ValueError(
                f"`rope_scaling`'s name field must be one of ['linear', 'dynamic'], got {rope_scaling_type}"
            )
        if rope_scaling_type in ["linear", "dynamic"]:
            if (
                rope_scaling_factor is None
                or not isinstance(rope_scaling_factor, float)
                or rope_scaling_factor < 1.0
            ):
                raise ValueError(
                    f"`rope_scaling`'s factor field must be an float >= 1, got {rope_scaling_factor}"
                )

    def forward(self, x: torch.Tensor, use_kv_cache: bool) -> torch.Tensor:
        # batch size, sequence length, embedding dimensionality (n_embd)
        B, T, C = None, None, None
        if x.ndim==3:
            B, T, C = x.size()
        else:
            B, T, N, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q = self.q_proj(x)
        k, v = self.kv_proj(x).split(self.n_embd_per_head * self.n_head, dim=2 if x.ndim==3 else 3)

        if use_kv_cache:
            # Optimized for single next prediction
            if self.kv_cache is not None:
                # Update cache
                k = torch.cat([self.kv_cache[0], k], dim=1)[:, 1:]
                v = torch.cat([self.kv_cache[1], v], dim=1)[:, 1:]
                self.kv_cache = k, v
            else:
                # Build cache
                self.kv_cache = k, v

        k = k.view(B, -1, self.n_head, self.n_embd_per_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, -1, self.n_head, self.n_embd_per_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, -1, self.n_head, self.n_embd_per_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        if x.ndim==4:
            q = q.reshape(B, self.n_head * N, T, self.n_embd_per_head)
            k = k.reshape(B, self.n_head * N, T, self.n_embd_per_head)
            v = v.reshape(B, self.n_head * N, T, self.n_embd_per_head)
        if self.rotary_emb is not None:
            cos, sin = self.rotary_emb(device=v.device, dtype=v.dtype, seq_len=T)
            q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids=None)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        #  att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        #  att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        #  att = F.softmax(att, dim=-1)
        #  y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        # efficient attention using Flash Attention CUDA kernels
        # When using kv cache at inference, is_causal=False since decoder is causal, at each generation step we want
        # to avoid recalculating the same previous token attention
        if use_kv_cache:
            y = scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.dropout, is_causal=False
            )
        else:
            y = scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.dropout, is_causal=True
            )

        # re-assemble all head outputs side by side
        if x.ndim==4:
            y = y.transpose(1, 2).contiguous().view(B, T, N, C)
        else:
            y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        y = self.c_proj(y)

        return y


def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)


class MLP(nn.Module):
    def __init__(self, config: LTSMConfig) -> None:
        super().__init__()
        hidden_dim = 4 * config.n_embd_per_head * config.n_head
        n_hidden = int(2 * hidden_dim / 3)
        n_hidden = find_multiple(n_hidden, 256)

        self.c_fc1 = nn.Linear(
            config.n_embd_per_head * config.n_head, n_hidden, bias=False
        )
        self.c_fc2 = nn.Linear(
            config.n_embd_per_head * config.n_head, n_hidden, bias=False
        )
        self.c_proj = nn.Linear(
            n_hidden, config.n_embd_per_head * config.n_head, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.c_fc1(x)) * self.c_fc2(x)
        x = self.c_proj(x)
        return x


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Derived from https://github.com/bzhangGo/rmsnorm/blob/master/rmsnorm_torch.py. BSD 3-Clause License:
    https://github.com/bzhangGo/rmsnorm/blob/master/LICENSE.
    """

    def __init__(self, size: int, dim: int = -1, eps: float = 1e-5) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(size))
        self.eps = eps
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # NOTE: the original RMSNorm paper implementation is not equivalent
        # norm_x = x.norm(2, dim=self.dim, keepdim=True)
        # rms_x = norm_x * d_x ** (-1. / 2)
        # x_normed = x / (rms_x + self.eps)
        # keep RMSNorm in float32
        norm_x = x.to(torch.float32).pow(2).mean(dim=self.dim, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return (self.scale * x_normed).type_as(x)


class LagLlamaModel(nn.Module):
    def __init__(
        self,
        context_length: int,
        max_context_length: int,
        scaling: str,
        input_size: int,
        n_layer: int,
        n_embd_per_head: int,
        n_head: int,
        lags_seq: List[int],
        distr_output: DistributionOutput,
        rope_scaling=None,
        num_parallel_samples: int = 100,
        time_feat: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.context_length = context_length
        self.lags_seq = lags_seq
        if time_feat:
            feature_size = input_size * (len(self.lags_seq)) + 2 * input_size + 5
            # feature_size = 94
        else:
            feature_size = input_size * (len(self.lags_seq)) + 2 * input_size

        config = LTSMConfig(
            n_layer=n_layer,
            n_embd_per_head=n_embd_per_head,
            n_head=n_head,
            block_size=max_context_length,
            feature_size=feature_size,
            rope_scaling=rope_scaling,
            dropout=dropout,
        )
        self.config = config
        self.num_parallel_samples = num_parallel_samples

        if scaling == "mean":
            self.scaler = MeanScaler(keepdim=True, dim=1)
        elif scaling == "std":
            self.scaler = StdScaler(keepdim=True, dim=1)
        elif scaling == "robust":
            self.scaler = RobustScaler(keepdim=True, dim=1)
        else:
            self.scaler = NOPScaler(keepdim=True, dim=1)

        self.distr_output = distr_output
        self.param_proj = self.distr_output.get_args_proj(
            config.n_embd_per_head * config.n_head
        )
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Linear(
                    #TODO: Check if this is correct
                    # config.feature_size, config.n_embd_per_head * config.n_head
                    # 123, config.n_embd_per_head * config.n_head
                    INTERMEDIATE_REPR, config.n_embd_per_head * config.n_head
                ),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=RMSNorm(config.n_embd_per_head * config.n_head),
            )
        )
        self.y_cache = False  # used at time of inference when kv cached is used
        self.LaplacianPE1 = nn.Linear(32, 32)
        self.LaplacianPE2 = nn.Linear(32, 32)
        self.act = nn.LeakyReLU()


    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(
                module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layer)
            )
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(
                module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layer)
            )

    def prepare_input(
        self,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        past_time_feat: Optional[torch.Tensor] = None,
        future_time_feat: Optional[torch.Tensor] = None,
        future_target: Optional[torch.Tensor] = None,
        lpls: Optional[torch.Tensor] = None,
    ):
        scaled_past_target, loc, scale = self.scaler(
            past_target, past_observed_values
        )  # Data is standardized (past_observed_values is passed as "weights" parameter) # (bsz, context_length+max(self.lags_seq)

        # In the below code, instead of max(self.lags_seq), it was previously -self.context_length
        if future_target is not None:
            if scaled_past_target.ndim==2:
                input = torch.cat(
                    (
                        scaled_past_target[..., max(self.lags_seq) :] ,  # Just the context
                        (future_target[..., :-1] - loc)
                        / scale,  # Not sure about the -1 here. Maybe so since the last value isn't used in the model for prediction of any new values. also if the prediction length is 1, this doesn't really affect anything
                    ),
                    dim=-1,
                )  # Shape is (bsz, context_length+(pred_len-1))
            else:
                input = torch.cat((
                    scaled_past_target[..., max(self.lags_seq) :,:],
                    (future_target[..., :-1, :] - loc) / scale,), dim=-2)


        else:
            input = scaled_past_target[..., max(self.lags_seq) :] if scaled_past_target.ndim==2 else scaled_past_target[..., max(self.lags_seq) :,:]
        if (past_time_feat is not None) and (future_time_feat is not None):
            time_feat = (
                torch.cat(
                    (
                        past_time_feat[..., max(self.lags_seq) :, :],
                        future_time_feat[..., :-1, :],
                    ),
                    dim=1,
                )
                if future_time_feat is not None
                else past_time_feat[..., max(self.lags_seq) :, :]
            )
        if past_target.ndim==2:
            prior_input = (
                past_target[..., : max(self.lags_seq)] - loc
            ) / scale  # This the history used to construct lags.  # bsz, max(self.lags_seq)
        else:
            prior_input = (
                past_target[..., : max(self.lags_seq), :] - loc
            ) / scale  # This the history used to construct lags.  # bsz, max(self.lags_seq)
        lags = lagged_sequence_values(
            self.lags_seq, prior_input, input, dim=-1 if input.ndim==2 else -2
        )  # Lags are added as an extra dim. Shape is (bsz, context_length+(pred_len-1), len(self.lags_seq))

        static_feat = torch.cat(
            (loc.abs().log1p(), scale.log()), dim=-1
        )  # (bsz, 2) (loc and scale are concatenated)
        expanded_static_feat = unsqueeze_expand(
            static_feat, dim=-2 if lags.ndim==3 else -3, size=lags.shape[-2] if lags.ndim==3 else lags.shape[-3]
        )  # (bsz, context_length+(pred_len-1), 2)
        # expanded_static_feat: (bsz, context_length+(pred_len-1), len(self.lags_seq) + 2); (bsz, 1); (bsz, 1)
        try:
            lap_pos_enc = self.LaplacianPE2(self.act(self.LaplacianPE1(lpls.to(torch.bfloat16))))
        except:
            lap_pos_enc = self.LaplacianPE2(self.act(self.LaplacianPE1(lpls.to(torch.float))))
        if lap_pos_enc.ndim==2:
            if lap_pos_enc.shape[0] == lags.shape[0]:
                lap_pos_enc = unsqueeze_expand(
                    lap_pos_enc, dim=-2, size=lags.shape[-2]
                )
                lap_pos_enc = unsqueeze_expand(lap_pos_enc, dim=1, size=lags.shape[1])
                time_feat = time_feat.unsqueeze(2).repeat(1, 1, lags.shape[2], 1)
        else:
            lap_pos_enc = lap_pos_enc.unsqueeze(1).repeat(1, lags.shape[1], 1, 1)
            time_feat = time_feat.unsqueeze(2).repeat(1, 1, lags.shape[2], 1)
        if lags.ndim==4:
            # expanded_static_feat = expanded_static_feat.reshape(expanded_static_feat.shape[0], expanded_static_feat.shape[1], expanded_static_feat.shape[3]//4, 4).repeat(1, 1, 2, 1)
            expanded_static_feat = expanded_static_feat.repeat(1, 1, 3, 1)
        if past_time_feat is not None:
            return (
                torch.cat((lags, expanded_static_feat, lap_pos_enc, time_feat), dim=-1),
                loc,
                scale,
            )
        else:
            return torch.cat((lags, expanded_static_feat), dim=-1), loc, scale

    def forward(
        self,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        past_time_feat: Optional[torch.Tensor] = None,
        future_time_feat: Optional[torch.Tensor] = None,
        future_target: Optional[torch.Tensor] = None,
        use_kv_cache: bool = False,
        lpls: Optional[torch.Tensor] = None,
    ) -> Iterable[torch.Tensor]:
        # if past_time_feat is not None:
        transformer_input, loc, scale = self.prepare_input(
            past_target=past_target,
            past_observed_values=past_observed_values,
            future_target=future_target,
            past_time_feat=past_time_feat,
            future_time_feat=future_time_feat,
            lpls=lpls,
        )  # return: (bsz, context_length+(pred_len-1), len(self.lags_seq) + 2); (bsz, 1); (bsz, 1)
        # To use kv cache for inference and pass recent token to transformer
        if use_kv_cache and self.y_cache:
            # Only use the most recent one, rest is in cache
            transformer_input = transformer_input[:, -1:]

        # forward the LLaMA model itself
        x = self.transformer.wte(
            transformer_input
        )  # token embeddings of shape (b, t, n_embd_per_head*n_head) # (bsz, context_length+(pred_len-1), n_embd_per_head*n_head)

        for block in self.transformer.h:
            x = block(x, use_kv_cache)
        transformer_output = x
        x = self.transformer.ln_f(
            x
        )  # (bsz, context_length+(pred_len-1), n_embd_per_head*n_head)
        if use_kv_cache:
            self.y_cache = True
        params = self.param_proj(
            x
        )  # (bsz, context_length+(pred_len-1)) ; (bsz, context_length+(pred_len-1))
        return params, loc, scale # (torch.Size([3200, 32]), torch.Size([3200, 32]), torch.Size([3200, 32])), torch.Size([3200, 1]), torch.Size([3200, 1])

    def reset_cache(self) -> None:
        """
        Resets all cached key-values in attention.
        Has to be called after prediction loop in predictor
        """
        for block in self.transformer.h:
            block.y_cache = None
            block.attn.kv_cache = None



class LLMMistralModel(LagLlamaModel):
    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.transformer = nn.ModuleDict(
            dict(
                wte=MistralMLP(
                    #TODO: Check if this is correct
                    # config.feature_size, config.n_embd_per_head * config.n_head
                    INTERMEDIATE_REPR, self.config.n_embd_per_head * self.config.n_head
                ),
                h=nn.ModuleList([MistralSdpaAttention(config_mist) for _ in range(self.config.n_layer)]),
                ln_f=MistralRMSNorm(self.config.n_embd_per_head * self.config.n_head),
            )
        )


    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(
                module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layer)
            )
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(
                module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layer)
            )

    def reset_cache(self) -> None:
        """
        Resets all cached key-values in attention.
        Has to be called after prediction loop in predictor
        """
        pass



class LagLlamaSeqToSeq(LagLlamaModel):
    def forward(
        self,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        past_time_feat: Optional[torch.Tensor] = None,
        future_time_feat: Optional[torch.Tensor] = None,
        future_target: Optional[torch.Tensor] = None,
        use_kv_cache: bool = False,
    ) -> Iterable[torch.Tensor]:
        # if past_time_feat is not None:
        transformer_input, loc, scale = self.prepare_input(
            past_target=past_target,
            past_observed_values=past_observed_values,
            future_target=future_target,
            past_time_feat=past_time_feat,
            future_time_feat=future_time_feat,
        )  # return: (bsz, context_length+(pred_len-1), len(self.lags_seq) + 2); (bsz, 1); (bsz, 1)
        # To use kv cache for inference and pass recent token to transformer
        if use_kv_cache and self.y_cache:
            # Only use the most recent one, rest is in cache
            transformer_input = transformer_input[:, -1:]

        # forward the LLaMA model itself
        x = self.transformer.wte(
            transformer_input
        )  # token embeddings of shape (b, t, n_embd_per_head*n_head) # (bsz, context_length+(pred_len-1), n_embd_per_head*n_head)
        for block in self.transformer.h:
            x = block(x, use_kv_cache)
        latent_x = x
        x = self.transformer.ln_f(
            x
        )  # (bsz, context_length+(pred_len-1), n_embd_per_head*n_head)
        if use_kv_cache:
            self.y_cache = True
        params = self.param_proj(
            x
        )  # (bsz, context_length+(pred_len-1)) ; (bsz, context_length+(pred_len-1))
        return params, loc, scale, latent_x  # (torch.Size([3200, 32]), torch.Size([3200, 32]), torch.Size([3200, 32])), torch.Size([3200, 1]), torch.Size([3200, 1])

class LagLlamaDAModel(nn.Module):
    def __init__(
        self,
        context_length: int,
        max_context_length: int,
        scaling: str,
        input_size: int,
        n_layer: int,
        n_embd_per_head: int,
        n_head: int,
        lags_seq: List[int],
        distr_output: DistributionOutput,
        rope_scaling=None,
        num_parallel_samples: int = 100,
        time_feat: bool = True,
        dropout: float = 0.0,
    ) -> None:
        """
        Domain adaptation model for LLaMA. This model is used to adapt the LLaMA model to a new dataset. MMD loss is used to align the source and target domains.
        """
        super().__init__()
        self.feature_extractor = LagLlamaSeqToSeq(
            context_length=context_length,
            max_context_length=max_context_length,
            scaling=scaling,
            input_size=input_size,
            n_layer=n_layer,
            n_embd_per_head=n_embd_per_head,
            n_head=n_head,
            lags_seq=lags_seq,
            distr_output=distr_output,
            rope_scaling=rope_scaling,
            num_parallel_samples=num_parallel_samples,
            time_feat=time_feat,
            dropout=dropout,
        )
        self.domain_classifier = nn.Sequential(
            nn.Linear(n_layer * n_embd_per_head * n_head, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 2),
        )
    
    def forward(self,
                past_target: torch.Tensor,
                past_observed_values: torch.Tensor,
                past_time_feat: Optional[torch.Tensor] = None,
                future_time_feat: Optional[torch.Tensor] = None,
                future_target: Optional[torch.Tensor] = None,
                use_kv_cache: bool = False,
                ):
        params, loc, scale, latent = self.feature_extractor(past_target, past_observed_values, past_time_feat, future_time_feat, future_target, use_kv_cache)
        domain_logits = self.domain_classifier(latent)
        return params, loc, scale, domain_logits
