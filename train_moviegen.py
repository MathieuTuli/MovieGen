"""
Reference code for Movie Gen training and inference.

References:
    # NOQA 1) https://ai.meta.com/static-resource/movie-gen-research-paper/?utm_source=twitter&utm_medium=organic_social&utm_content=thread&utm_campaign=moviegen
"""
from typing import Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

import argparse
import inspect
import time
import math
import os

from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import functional as F
from einops import rearrange
from torch import einsum

import torch.distributed as dist
import torch.nn as nn
import numpy as np
import torch

from text_encoder import TextEncoder, TextEncoderConfig
from tae import TAE, TAEConfig

# NOQA TODO: dim issues for sure, need to test: particularly the cross attention I just plopped in there
# NOQA TODO: conv3d patchifier also is naively wrote in from first paper reading: needs review
# NOQA DONE: need to add the rest of the TAE augs for time inflation and weight loading
# NOQA TODO: review all REVISIT tags
# NOQA TODO: dataloader also empty, fill that
# NOQA TODO: DDP is copied naively from llm.c - confirm
# NOQA TODO: shard text/vae probably (other parallels from paper maybe?)
# NOQA DONE: refactor to enable pre-train tae - or keep in same file? idk

"""
-------------------------------------------------------------------------------
helpers
-------------------------------------------------------------------------------
"""


def asspath(strarg):
    """helper to ensure arg path exists"""
    p = Path(strarg)
    if p.exists():
        return p
    else:
        raise NotADirectoryError(strarg)


def mkpath(strarg):
    """helper to mkdir arg path if it doesn't exist"""
    if not strarg:
        return ""
    p = Path(strarg)
    p.mkdir(exist_ok=True, parents=True)
    return p


def print0(*args, **kwargs):
    """modified print that only prints from the master process"""
    if int(os.environ.get("RANK", 0)) == 0:
        print(*args, **kwargs)


"""
-------------------------------------------------------------------------------
torch helpers
-------------------------------------------------------------------------------
"""


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


def linear_quadratic_t_schedule(a: float = 0.,
                                b: float = 1.,
                                steps: int = 50,
                                N: int = 1000):
    assert steps < N
    halfsteps = steps // 2
    first = np.linspace(a, b, N)[:halfsteps].tolist()
    second = np.geomspace(first[-1], b, halfsteps).tolist()
    return first + second


def euler_sampler():
    pass


"""
-------------------------------------------------------------------------------
RoPE related
-------------------------------------------------------------------------------
"""


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim -
             1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_scaling(freqs: torch.Tensor):
    # Values obtained from grid search
    scale_factor = 8
    low_freq_factor = 1
    high_freq_factor = 4
    old_context_len = 6144  # original moviegen length

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    new_freqs = []
    for freq in freqs:
        wavelen = 2 * math.pi / freq
        if wavelen < high_freq_wavelen:
            new_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            new_freqs.append(freq / scale_factor)
        else:
            assert low_freq_wavelen != high_freq_wavelen
            smooth = (old_context_len / wavelen - low_freq_factor) / (
                high_freq_factor - low_freq_factor
            )
            new_freqs.append((1 - smooth) * freq /
                             scale_factor + smooth * freq)
    return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)


def precompute_freqs_cis(
    dim: int, end: int, theta: float = 10000.0, use_scaled: bool = False
):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                   [: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    if use_scaled:
        freqs = apply_scaling(freqs)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


"""
-------------------------------------------------------------------------------
building blocks
-------------------------------------------------------------------------------
"""


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class BidirectionalSelfAttention(nn.Module):
    """
    source: https://github.com/karpathy/llm.c
        - Copy of the Causal attention without the masking basically
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_rep = self.n_head // self.n_kv_head
        self.hd = config.n_embd // config.n_head
        self.use_kv = config.use_kv
        self.flash = config.flash

        # key, query, value projections
        self.c_attn = nn.Linear(config.n_embd,
                                (config.n_head + 2 * config.n_kv_head)
                                * self.hd, bias=False)
        self.c_proj = nn.Linear(
            config.n_embd, config.n_embd, bias=False)  # output projection

        # static KV cache:
        # we could allocate it outside model and pass it in but we don't
        if self.use_kv:
            self.cache_k = torch.zeros((
                config.max_gen_batch_size, config.block_size,
                config.n_kv_head, self.hd))
            self.cache_v = torch.zeros((
                config.max_gen_batch_size, config.block_size,
                config.n_kv_head, self.hd))

    def forward(self, x, freqs_cis=None, start_pos=None, mask=None):
        assert mask is None
        # batch size, sequence length, embedding dimensionality (n_embd)
        B, T, C = x.size()
        # calculate query, key, values for all heads in batch
        # & move head forward to be the batch dim
        qkv = self.c_attn(x)
        q, k, v = qkv.split([self.n_head * self.hd,
                             self.n_kv_head * self.hd,
                             self.n_kv_head * self.hd], dim=-1)
        q, k, v = map(lambda t: t.view(B, T, -1, self.hd),
                      (q, k, v))  # (B, T, NH, HD)

        # rotate QK (rope)  <-- 1. difference compared to GPT-2
        q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)

        # use kv-caching during inference
        if self.use_kv and not self.training and start_pos >= 0:
            self.cache_k[:B, start_pos: start_pos + T] = k
            self.cache_v[:B, start_pos: start_pos + T] = v
            k = self.cache_k[:B, : start_pos + T]
            v = self.cache_v[:B, : start_pos + T]

        k = repeat_kv(k, self.n_rep)  # GQA <-- 2. difference compared to GPT-2
        v = repeat_kv(v, self.n_rep)

        q, k, v = map(lambda t: t.transpose(1, 2), (q, k, v))  # (B, NH, T, HD)

        if self.flash:
            # flashattention
            # if T == 1 no need to mask, otherwise the function complains
            # scaled_dot_product_attention expects a mask where value of True
            # indicates that the element should take part in attention
            # our mask is the opposite, so we need to invert it
            y = F.scaled_dot_product_attention(
                q, k, v, mask == 0 if T > 1 else None)
        else:
            # manual implementation of attention
            # this materializes the large (T,T) matrix for all queries and keys
            scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.hd))
            if mask is not None:
                scores.masked_fill_(mask, torch.finfo(scores.dtype).min)
            att = F.softmax(scores.float(), dim=-1).type_as(q)
            y = att @ v  # (B, NH, T, T) x (B, NH, T, HD) -> (B, NH, T, HD)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class BidirectionalCrossAttention(nn.Module):
    """
    # NOQA source: https://github.com/lucidrains/bidirectional-cross-attention/tree/main
    """

    def __init__(
        self,
        *,
        dim,
        heads=8,
        dim_head=64,
        context_dim=None,
        dropout=0.,
        talking_heads=False,
        prenorm=False
    ):
        super().__init__()
        context_dim = context_dim or dim

        self.norm = nn.LayerNorm(dim) if prenorm else nn.Identity()
        self.context_norm = nn.LayerNorm(
            context_dim) if prenorm else nn.Identity()

        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.dropout = nn.Dropout(dropout)
        self.context_dropout = nn.Dropout(dropout)

        self.to_qk = nn.Linear(dim, inner_dim, bias=False)
        self.context_to_qk = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.context_to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Linear(inner_dim, dim)
        self.context_to_out = nn.Linear(inner_dim, context_dim)

        self.talking_heads = nn.Conv2d(
            heads, heads, 1, bias=False) if talking_heads else nn.Identity()
        self.context_talking_heads = nn.Conv2d(
            heads, heads, 1, bias=False) if talking_heads else nn.Identity()

    def forward(
            self,
            x,
            context,
            mask=None,
            context_mask=None,
            return_attn=False,
            rel_pos_bias=None):
        b, i, j, h, device = \
            x.shape[0], x.shape[-2], context.shape[-2], self.heads, x.device

        x = self.norm(x)
        context = self.context_norm(context)

        # get shared query/keys and values for sequence and context

        qk, v = self.to_qk(x), self.to_v(x)
        context_qk, context_v = self.context_to_qk(
            context), self.context_to_v(context)

        # split out head

        qk, context_qk, v, context_v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=h), (qk, context_qk, v, context_v))

        # get similarities

        sim = einsum('b h i d, b h j d -> b h i j',
                     qk, context_qk) * self.scale

        # relative positional bias, if supplied

        if rel_pos_bias is not None:
            sim = sim + rel_pos_bias

        # mask

        if mask is not None or context_mask is not None:
            mask = mask or torch.ones(
                (b, i), device=device, dtype=torch.bool)
            context_mask = context_mask or torch.ones(
                (b, j), device=device, dtype=torch.bool)

            attn_mask = rearrange(mask, 'b i -> b 1 i 1') * \
                rearrange(context_mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~attn_mask, -torch.finfo(sim.dtype).max)

        # get attention along both sequence len and context len dimensions
        # shared similarity matrix

        attn = sim.softmax(dim=-1)
        context_attn = sim.softmax(dim=-2)

        # dropouts

        attn = self.dropout(attn)
        context_attn = self.context_dropout(context_attn)

        # talking heads

        attn = self.talking_heads(attn)
        context_attn = self.context_talking_heads(context_attn)

        # src sequence aggregates values from context,
        # context aggregates values from src sequence

        out = einsum('b h i j, b h j d -> b h i d', attn, context_v)
        context_out = einsum('b h j i, b h j d -> b h i d', context_attn, v)

        # merge heads and combine out

        out, context_out = map(lambda t: rearrange(
            t, 'b h n d -> b n (h d)'), (out, context_out))

        out = self.to_out(out)
        context_out = self.context_to_out(context_out)

        if return_attn:
            return out, context_out, attn, context_attn

        return out, context_out


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.ffn_dim, bias=False)
        self.c_fc2 = nn.Linear(config.n_embd, config.ffn_dim, bias=False)
        self.c_proj = nn.Linear(config.ffn_dim, config.n_embd, bias=False)

    def forward(self, x):
        # REVISIT: SwiGLU self.c_proj(F.silu(self.c_fc2(x)) * self.c_fc(x))  <-- 3.
        x1 = self.c_fc(x)
        x2 = self.c_fc2(x)
        x2 = F.silu(x2)
        x = x1 * x2
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        # REVISIT: where does the adaptive layer norm for t go?
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd, config.norm_eps)
        # REVISIT: Check this bi self attention naively
        self.attn = BidirectionalSelfAttention(config)
        self.ln_2 = RMSNorm(config.n_embd, config.norm_eps)
        # REVISIT: Check this cross attention added naively
        self.bd_cross_attn = BidirectionalCrossAttention(
            dim=config.n_embd, heads=config.n_head,
            dim_head=config.dim_head)
        # REVISIT: should be another norm here
        self.mlp = MLP(config)

    def forward(self, x, ctx, freqs_cis=None, start_pos=None, mask=None):
        x = x + self.attn(self.ln_1(x), freqs_cis, start_pos, mask)
        x = self.bd_cross_attn(x, ctx)
        x = x + self.mlp(self.ln_2(x))
        return x


"""
-------------------------------------------------------------------------------
models
-------------------------------------------------------------------------------
"""


@dataclass
class MovieGenConfig:
    version: str = "1.0"
    block_size: int = 8192
    in_channels: int = 64
    vocab_size: int = ...
    n_layer: int = 48
    n_head: int = 48
    dim_head: int = 128  # REVISIT: or 96?
    n_embd: int = 6144
    ffn_dim: int = 16384  # 6144 * 4 * 2/3
    multiple_of: int = 1024
    norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    use_scaled_rope: bool = True
    max_gen_batch_size: int = 4
    use_kv: bool = True
    flash: bool = False  # use flashattention?
    patch_k: Tuple[int, int, int] = (1, 2, 2)

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)


class MovieGen(nn.Module):
    def __init__(self,
                 config: MovieGenConfig,
                 ):
        super().__init__()
        self.config = config

        # auxilary models
        self.text_encoder = TextEncoder(TextEncoderConfig())
        self.tae = TAE(TAEConfig())

        self.patchifier = nn.Conv3d(in_channels=config.in_channels,
                                    out_channels=config.n_embd,
                                    kernel_size=config.patch_k,
                                    stride=config.patch_k)
        self.transformer = nn.ModuleDict(dict(
            # REVISIT: wte=nn.Embedding(config.vocab_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=RMSNorm(config.n_embd, config.norm_eps),
        ))

        # init all weights, use a torch rng object to be very careful
        self.init_rng = torch.Generator()
        self.init_rng.manual_seed(420)

        self.freqs_cis = precompute_freqs_cis(
            config.n_embd // config.n_head,
            config.block_size * 2,
            config.rope_theta,
            config.use_scaled_rope,)

    @classmethod
    def initialize_auxiliary_models(self,
                                    metaclip_ckpt: Optional[Path],
                                    tae_ckpt: Optional[Path]):
        if metaclip_ckpt is not None:
            self.text_encoder.from_pretrained(metaclip_ckpt)
        if tae_ckpt is not None:
            self.tae.from_pretrained(tae_ckpt)

    @classmethod
    def from_pretrained(self,
                        ckpt: Path, ul2_ckpt: Path, byt5_ckpt: Path,
                        metaclip_ckpt: Path, tae_ckpt: Path):
        model_args = MovieGenConfig()

        checkpoint = torch.load(ckpt, map_location="cpu")

        # save the default type
        original_default_type = torch.get_default_dtype()
        # much faster loading
        torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)
        model = MovieGen(model_args)
        model.load_state_dict(checkpoint, strict=False)
        # restore type
        torch.set_default_tensor_type(torch.tensor(
            [], dtype=original_default_type, device="cpu").type())

        self.initialize_auxiliary_models(
            ul2_ckpt, byt5_ckpt, metaclip_ckpt, tae_ckpt)
        return model

    def loss(self,):
        ...

    def configure_optimizers(self,
                             lr: float,
                             weight_decay: float,
                             betas: Tuple[float, float],
                             device_type: torch.device,
                             zero_stage: int):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups.
        # Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay,
        #   all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print0(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")  # NOQA
        print0(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")  # NOQA
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(
            torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        print0(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(
            optim_groups, lr=lr, betas=betas, fused=use_fused)
        return optimizer

    def forward(self, x, prompt,
                targets=None, return_logits=True, start_pos=0):
        _, t = x.size()  # REVISIT: from llama3, needs to be updated
        ctx = self.text_encoder(prompt)
        x = self.tae.encode(x)
        # REVISIT: does this need permuting?
        # x = x.permute(0, 2, 1, 3, 4)
        x = self.patchifier(x)
        # REVISIT:
        # pos = self.pos_embd(x)
        x = torch.flatten(x, start_dim=1)

        freqs_cis = self.freqs_cis[start_pos:start_pos+t]
        for i, block in enumerate(self.transformer.h):
            x = block(x, ctx, freqs_cis, start_pos)
        x = self.transformer.ln_f(x)

        x = self.tae.decode(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x).float()
            loss = F.cross_entropy(
                logits.view(
                    -1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head
            # on the very last position
            # note: using list [-1] to preserve the time dim
            logits = self.lm_head(x[:, [-1], :]).float()
            loss = None

        if not return_logits:
            logits = None

        return logits, loss


"""
-------------------------------------------------------------------------------
dataloader
-------------------------------------------------------------------------------
"""


class DDataLoader:
    def __init__(self):
        # TODO: prepend "FPS-16" to prompts
        pass


"""
-------------------------------------------------------------------------------
args
-------------------------------------------------------------------------------
"""

parser = argparse.ArgumentParser()
# io
# yes you can use parser types like this
parser.add_argument("--ckpt", type=asspath, required=False)
parser.add_argument("--output-dir", type=mkpath, default="")
parser.add_argument("--ul2-ckpt", type=asspath, required=False)
parser.add_argument("--byt5-ckpt", type=asspath, required=False)
parser.add_argument("--metaclip-ckpt", type=asspath, required=False)
parser.add_argument("--tae-ckpt", type=asspath, required=False)
# optimization
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--weight-decay", type=float, default=0.0)
parser.add_argument("--grad_clip", type=float, default=1.0)
parser.add_argument("--fps", type=int, default=16)
parser.add_argument("--batch-size", type=int, default=1)
parser.add_argument("--total-batch-size", type=int, default=1)
parser.add_argument("--sequence-length", type=int, default=64)

parser.add_argument("--num-iterations", type=int, default=10)
parser.add_argument("--val-loss-every", type=int, default=0)
parser.add_argument("--val-max-steps", type=int, default=20)
parser.add_argument("--overfit-batch", default=False, action="store_true")
# memory management
parser.add_argument("--dtype", type=str, default="bfloat16")
parser.add_argument("--compile", default=False, action="store_true")


if __name__ == "__main__":
    args = parser.parse_args()
    # args error checking
    assert args.dtype in {"float32", "float16", "bfloat16"}

    def get_lr(it):
        min_lr = args.learning_rate * args.learning_rate_decay_frac
        # 1) linear warmup for warmup_iters steps
        if it < args.warmup_iters:
            return args.learning_rate * (it+1) / args.warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > args.num_iterations:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - args.warmup_iters) / \
            (args.num_iterations - args.warmup_iters)
        assert 0 <= decay_ratio <= 1
        # coeff starts at 1 and goes to 0
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (args.learning_rate - min_lr)

    logfile = None
    if args.output_dir:
        logfile = args.output_dir / "main.log"

    ddp = int(os.environ.get('RANK', -1)) != -1  # is this a ddp run?
    if ddp:
        assert torch.cuda.is_available()
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = torch.device(f'cuda:{ddp_local_rank}')
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
        seed_offset = 0  # each process gets the exact same seed
        zero_stage = args.zero_stage
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        zero_stage = 0
        ddp_world_size = 1
        master_process = True
        seed_offset = 0
        device = torch.device("cuda")

    B, T = args.batch_size, args.sequence_length
    tokens_per_fwdbwd = B * T * ddp_world_size
    assert args.total_batch_size % tokens_per_fwdbwd == 0
    grad_accum_steps = args.total_batch_size // tokens_per_fwdbwd

    # set up a context manager following the desired dtype and device
    ptdtype = {'float32': torch.float32,
               'bfloat16': torch.bfloat16,
               'float16': torch.float16}[args.dtype]
    ctx = torch.amp.autocast(device_type="cuda", dtype=ptdtype)

    torch.manual_seed(420)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(420)

    config = MovieGenConfig()
    model = MovieGen(config)
    if args.compile:
        model = torch.compile(model)

    train_loader = ...
    val_loader = ...

    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model
    optimizer = model.configure_optimizers(lr=args.lr,
                                           weight_decay=args.weight_decay,
                                           betas=(0.9, 0.95),
                                           device_type=device)

    torch.cuda.reset_peak_memory_stats()
    timings = list()
    for step in range(args.num_iterations + 1):
        t0 = time.time()
        last_step = (step == args.num_iterations - 1)

        # once in a while evaluate the validation dataset
        if (args.val_loss_every > 0
                and (step % args.val_loss_every == 0 or last_step)) \
                and (val_loader is not None):
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss = 0.
                for _ in range(args.val_max_steps):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    _, loss = model(x, y, return_logits=False)
                    val_loss += loss.item()
                val_loss /= args.val_max_steps
            # log to console and to file
            print0(f"val loss {val_loss}")
            if master_process and logfile is not None:
                with open(logfile, "a") as f:
                    f.write("s:%d vl:%f\n" % (step, val_loss))

        if last_step:
            break

        # --------------- TRAINING SECTION BEGIN -----------------
        model.train()
        optimizer.zero_grad(set_to_none=True)
        if args.overfit_single_batch:
            train_loader.reset()

        lossf = 0.
        for micro_step in range(grad_accum_steps):
            # fetch a batch
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)

            if ddp:
                model.require_backward_grad_sync = (
                    micro_step == grad_accum_steps - 1)
            # forward pass
            with ctx:
                _, loss = model(x, y, return_logits=False)
                loss = loss / grad_accum_steps
                lossf += loss.detach()  # keep track of the mean loss
            loss.backward()
        if ddp:
            dist.all_reduce(lossf, op=dist.ReduceOp.AVG)
        lossf = lossf.item()
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.grad_clip)
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        # step the optimizer
        optimizer.step()
        # --------------- TRAINING SECTION END -------------------

        torch.cuda.synchronize()
        t1 = time.time()
        tokens_per_second = \
            grad_accum_steps * ddp_world_size * B * T / (t1 - t0)
        print0(f"""step {step+1:4d}/{args.num_iterations} | \
               train loss {lossf:.6f} | \
               norm {norm:.4f} | \
               lr {lr:.2e} | \
               ({(t1-t0)*1000:.2f} ms | {tokens_per_second:.0f} tok/s) \
               """)
        if master_process and logfile is not None:
            with open(logfile, "a") as f:
                f.write("s:%d tl:%f\n" % (step, lossf))

        # keep track of smooth timings, last 20 iterations
        if step > 0 and step > args.num_iterations - 20:
            timings.append(t1 - t0)

    # print the average of the last 20 timings, to get something smooth-ish
    timings = timings[-20:]
    print0(f"final {len(timings)} iters avg: {np.mean(timings)*1000:.3f}ms")
    print0(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")  # NOQA

    # -------------------------------------------------------------------------
    # clean up nice
    if ddp:
        destroy_process_group()
