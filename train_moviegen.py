"""
Reference code for Movie Gen training and inference.

References:
    # NOQA 1) https://ai.meta.com/static-resource/movie-gen-research-paper/?utm_source=twitter&utm_medium=organic_social&utm_content=thread&utm_campaign=moviegen
"""
from typing import Tuple, Optional
from dataclasses import dataclass
from inspect import isfunction
from pathlib import Path

import argparse
import inspect
import random
import signal
import math
import time
import os

from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import functional as F
from einops import rearrange, repeat
from torch import einsum
from PIL import Image

# import torch.distributed as dist
import torch.nn as nn
import numpy as np
import torchvision
import torch
import cv2

from text_encoder import TextEncoder, TextEncoderConfig
from tae import TAE, TAEConfig

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


def cleanup():
    """Cleanup function to destroy the process group"""
    if int(os.environ.get('RANK', -1)) != -1:
        destroy_process_group()


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print0("\nCtrl+C caught. Cleaning up...")
    cleanup()
    exit(0)


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


def exists(val):
    return val is not None


def uniq(arr):
    return {el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


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


class CrossAttention(nn.Module):
    """
    pulled from https://github.com/facebookresearch/DiT
    """

    def __init__(self, query_dim,
                 context_dim=None,
                 heads=8,
                 dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = context_dim if context_dim is not None else query_dim

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.ffn_dim, bias=False)
        self.c_fc2 = nn.Linear(config.n_embd, config.ffn_dim, bias=False)
        self.c_proj = nn.Linear(config.ffn_dim, config.n_embd, bias=False)

    def forward(self, x):
        # REVISIT:uses SwiGLU
        # SwiGLU self.c_proj(F.silu(self.c_fc2(x)) * self.c_fc(x))
        x1 = self.c_fc(x)
        x2 = self.c_fc2(x)
        x2 = F.silu(x2)
        x = x1 * x2
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.rmsnorm1 = RMSNorm(config.n_embd, config.norm_eps)
        self.attn = CrossAttention(query_dim=config.n_embd,
                                   heads=config.n_head,
                                   dim_head=config.dim_head,
                                   dropout=config.dropout)
        self.rmsnorm2 = RMSNorm(config.n_embd, config.norm_eps)
        self.cross_attn = CrossAttention(query_dim=config.n_embd,
                                         context_dim=config.n_embd,
                                         heads=config.n_head,
                                         dim_head=config.dim_head,
                                         dropout=config.dropout)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.n_embd, 6 * config.n_embd, bias=True)
        )
        self.mlp = MLP(config)

    def forward(self, x, ctx, t, pos_emb, mask=None):
        # get adaln scale/shift
        shift_1, scale_1, alpha_1, shift_2, scale_2, alpha_2 = \
            self.adaLN_modulation(t).chunk(6, dim=1)
        x = x + pos_emb
        x = x + scale_1 * self.attn(modulate(self.rmsnorm1(x),
                                             shift_1, scale_1), mask=mask)
        x = x + self.cross_attn(x, ctx, mask=mask)
        x = x + scale_2 * self.mlp(modulate(self.rmsnorm2(x),
                                            shift_2, scale_2))
        return x


class Head(nn.Module):
    """
    The final layer of MovieGen
    """

    def __init__(self, config):
        super().__init__()
        self.norm = RMSNorm(config.n_embd, config.norm_eps)
        self.linear = nn.Linear(config.n_embed,
                                config.in_channels,
                                bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.n_embed, 2 * config.n_embed, bias=True)
        )

    def forward(self, x, c):
        # in: [B, ctx_len, 6144]
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        # out: [B, ctx_len, in_channels=16]
        return x


class TimestepEmbedder(nn.Module):
    """
    Copied from DiT
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) *
                          torch.arange(start=0, end=half, dtype=torch.float32)
                          / half).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


"""
-------------------------------------------------------------------------------
models
-------------------------------------------------------------------------------
"""


@dataclass
class MovieGenConfig:
    version: str = "1.0"
    in_channels: int = 16
    n_layer: int = 48
    n_head: int = 48
    dim_head: int = 128
    n_embd: int = 6144
    ffn_dim: int = 16384  # 6144 * 4 * 2/3
    context_len: int = 8192  # (256 * 256 * 256) / [(8 * 8 * 8) * (1 * 2 * 2)]
    max_frames: int = 256
    spatial_resolution: int = 256
    patch_k: Tuple[int, int, int] = (1, 2, 2)
    norm_eps: float = 1e-5
    # use kv cache?
    # - max_gen_batch_size: int = 4
    # - block_size: int = 8192
    # - use_kv: bool = False
    # use flashattention?
    # flash: bool = False

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)


class MovieGen(nn.Module):
    def __init__(self,
                 config: MovieGenConfig,
                 ):
        super().__init__()
        assert config.context_len == (
            (config.max_frames * config.spatial_resolution ** 2) /
            (8 ** 3) * np.prod(config.patch_k),
            "Context length is wrong.")
        self.config = config

        self.patchifier = nn.Conv3d(in_channels=config.in_channels,
                                    out_channels=config.n_embd,
                                    kernel_size=config.patch_k,
                                    stride=config.patch_k,
                                    bias=True)
        self.patch_proj = nn.Sequential(
            nn.LayerNorm(config.in_channels),
            nn.Linear(config.in_channels, config.n_embd),
            nn.LayerNorm(config.n_embd),
        )
        self.t_embedder = TimestepEmbedder(config.n_embed)
        # NOTE: 8-compression from TAE
        self.pos_embed_h = nn.Parameter(torch.randn(
            config.spatial_resolution / (8 * config.patch_k[1]),
            config.n_embd))
        self.pos_embed_w = nn.Parameter(torch.randn(
            config.spatial_resolution / (8 * config.patch_k[2]),
            config.n_embd))
        self.pos_embed_t = nn.Parameter(torch.randn(
            config.max_frames / (8 * config.patch_k[0]),
            config.n_embd))

        # auxilary models
        self.text_encoder = TextEncoder(TextEncoderConfig())
        self.tae = TAE(TAEConfig())

        self.transformer = nn.ModuleDict(dict(
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))
        self.head = Head(config)
        # I borrow the lm head from DiT instead of the Llama head

        # init all weights, use a torch rng object to be very careful
        self.init_rng = torch.Generator()
        self.init_rng.manual_seed(420)

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

    def unpatchify(self, x):
        """
        x: (N, T, prod(patch_k) * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, prompt, t, mask, targets=None):
        ctx = self.text_encoder(prompt)
        x = self.tae.encode(x)  # [B, T, C, H, W]
        B, T, C, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
        x = self.patchifier(x)
        x = torch.flatten(x, start_dim=2).permute(0, 2, 1)  # [B, T*H*W, C]
        # note that T*H*W is the max seq length, since we pad
        x = self.patch_proj(x)  # [B, T*H*W, 6144]

        # NOTE: Positional Embedding
        # - hardcoded resolution, for now - see 'revisit' above
        assert (H < self.config.max_spatial_seq_len and
                W < self.config.max_spatial_seq_len and
                T < self.config.max_temporal_seq_len)
        pos = torch.stack(torch.meshgrid((
            torch.arange(H, device=x.device),
            torch.arange(W, device=x.device)
        ), indexing='ij'), dim=-1)
        # this is a [num patches x spatial res (x, y)] tensor
        pos = rearrange(pos, 'h w c -> (h w) c')
        pos = repeat(pos, 'n d -> b n d', b=B)
        # these are now [B, num patches]
        h_indices, w_indices = pos.unbind(dim=-1)
        # these are [B, num patches, dim]
        # expand dims for time dimension to be [B, 1, num patches, dim]
        pos_h = self.pos_embed_h[h_indices]
        pos_w = self.pos_embed_w[w_indices]

        t_indices = torch.arange(T,  # self.config.max_temporal_seq_len,
                                 device=x.device)
        t_indices = repeat(t_indices, 'n -> b n', b=x.shape[0]).unsqueze
        pos_t = self.pos_embed_t[t_indices]
        # todo: need to pad patches and roll the time dimesion into the
        # representation
        # basically, repeat the spatial dimension max_T times
        # and for the time emd, repeat is hxw times, but spaced
        # [B, num patches, dim] +
        # [B, num patches, dim] +
        # [B, T          , dim]

        # REVISIT: pad to the max seq length
        # the thing is, the dataloader already pads at the batch level
        # this will pad the remaining tokens, up to context len
        # I might want to aggregate the padding logic to one spot
        pad_len = self.config.context_len - x.shape[1]
        x = F.pad(x, (0, 0, 0, pad_len))
        pos_t = torch.repeat_interleave(pos_h, pos_h.shape[1], dim=1)
        pos_t = F.pad(pos_h, (0, 0, 0, pad_len))
        pos_h = repeat(pos_h, 'b n d -> b (n t) d', t=T)
        pos_h = F.pad(pos_h, (0, 0, 0, pad_len))
        pos_w = repeat(pos_w, 'b n d -> b (n t) d', t=T)
        pos_w = F.pad(pos_w, (0, 0, 0, pad_len))
        assert (pos_h.shape[1] == x.shape[1] and
                pos_h.shape[1] == pos_w.shape[1] and
                pos_h.shape[1] == pos_t.shape[1]), \
            "Pos emb and inputs don't match shape"

        pos_emb = pos_h + pos_w + pos_t
        t = self.t_embedder(t)
        # Original mask is [B, T]
        # Each mask element corresponds to (256/8/2)^2 = 256 spatial patches
        # - or (16x16)
        # First compress temporal dimension by 8 (from TAE)
        # this ensures that we "round up"
        mask = mask.view(B, T//8, 8).any(dim=2).float()  # [B, T/8]
        # Each temporal position expands to 256 spatial positions
        spatial_per_temporal = (
                self.config.spatial_resolution // 8 // self.config.patch_k[1] *
                self.config.spatial_resolution // 8 // self.config.patch_k[2]
        )
        # [B, T/8, 256]
        mask = mask.unsqueeze(-1).repeat(1, 1, spatial_per_temporal)
        mask = mask.view(B, -1)  # [B, (T/8) * spatial_per_temporal]
        # Pad to match the context length
        mask = F.pad(mask, (0, pad_len))  # [B, context_len]

        for i, block in enumerate(self.transformer.h):
            x = block(x, ctx, t, pos_emb, mask=mask)

        x = self.head(x)  # [B, ctx_len, in_channels=16]
        x = x.permute(0, 2, 1).view(
            B, self.config.in_channels,
            self.config.max_frames,
            self.config.spatial_resolution,
            self.config.spatial_resolution).permute(0, 2, 1, 3, 4)

        x = self.tae.decode(x)

        if targets is not None:
            raise NotImplementedError
            loss = None
        else:
            loss = None

        return x, loss


"""
-------------------------------------------------------------------------------
dataloader
-------------------------------------------------------------------------------
"""


class Dataset:
    def __init__(self, root: Path, T: int, size: int = 64, train: bool = True):
        self.train = train
        self.T = T
        self.files = list()
        for ext in ('*.mp4', '*.avi', '*.mov', '*.mkv', '*.webm', '*.gif'):
            self.files.extend(root.glob(ext))
        self.files = sorted(self.files)  # Sort for consistency
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((size, size)),
            torchvision.transforms.PILToTensor(),
            torchvision.transforms.ConvertImageDtype(torch.float32),
            torchvision.transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.files)

    def load_video(self, fname):
        vidcap = cv2.VideoCapture(fname)
        ret, x = vidcap.read()
        x = Image.fromarray(cv2.cvtColor(x, cv2.COLOR_BGR2RGB))
        x = self.transforms(x)[None, :]
        while ret:
            ret, frame = vidcap.read()
            if ret:
                frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                x = torch.cat([x, self.transforms(frame)[None, :]])
        return x

    def __getitem__(self, index):
        mask = torch.ones([self.T])
        video = self.load_video(self.files[index])
        vlen = video.shape[0]
        # 1:3 image/video ratio from paper
        if index % 4 == 0:
            id = random.randint(0, vlen - 1) if self.train else 0
            frame = video[id]
            zero_pad = torch.zeros(self.T - 1, *frame.shape,
                                   dtype=frame.dtype)
            x = torch.cat([frame[None, :], zero_pad], dim=0)
            mask[1:] = 0
        else:
            id = 0
            if self.train:
                id = random.randint(0, max(vlen - self.T - 1, 0))
            x = video[id:id + self.T]
            if x.shape[0] < self.T:
                zero_pad = torch.zeros(self.T - x.shape[0],
                                       *x.shape[1:],
                                       dtype=frame.dtype)
                x = torch.cat([x, zero_pad], dim=0)
            mask[x.shape[1]:] = 0
        return x, mask


"""
-------------------------------------------------------------------------------
args
-------------------------------------------------------------------------------
"""

parser = argparse.ArgumentParser()
# io
# yes you can use parser types like this
parser.add_argument("--ul2-ckpt", type=asspath, required=False)
parser.add_argument("--byt5-ckpt", type=asspath, required=False)
parser.add_argument("--metaclip-ckpt", type=asspath, required=False)
parser.add_argument("--tae-ckpt", type=asspath, required=False)
parser.add_argument("--output-dir", type=mkpath, default="")
parser.add_argument("--train-dir", type=asspath, default="dev/data/train-smol")
parser.add_argument("--val-dir", type=asspath, default="dev/data/val-smol")

# checkpointing
parser.add_argument("--ckpt", type=asspath, required=False)
parser.add_argument("--ckpt-from-ldm", type=int, default=0, choices=[0, 1])
parser.add_argument("--resume", type=int, default=0, choices=[0, 1])
parser.add_argument("--ckpt-freq", type=int, default=-1)
parser.add_argument("--device", type=str, default="cuda")

# optimization
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--weight-decay", type=float, default=0.0)
parser.add_argument("--grad-clip", type=float, default=1.0)
parser.add_argument("--batch-size", type=int, default=1)
parser.add_argument("--max-frames", type=int, default=32)
parser.add_argument("--resolution", type=int, default=64)

parser.add_argument("--num-iterations", type=int, default=10)
parser.add_argument("--val-loss-every", type=int, default=0)
parser.add_argument("--val-max-steps", type=int, default=20)
parser.add_argument("--overfit-batch", default=1, type=int)

parser.add_argument("--inference-only", default=0, type=int, choices=[0, 1])
parser.add_argument("--verbose-loss", default=0, type=int, choices=[0, 1])
parser.add_argument("--seed", default=420, type=int)

# memory management
parser.add_argument("--dtype", type=str, default="float32")
parser.add_argument("--compile", default=0, type=int)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    args = parser.parse_args()
    # args error checking
    assert args.dtype in {"float32"}

    train_logfile, val_logfile = None, None
    if args.output_dir:
        train_logfile = args.output_dir / "train.log"
        val_logfile = args.output_dir / "val.log"
        if args.resume == 0:
            open(train_logfile, 'w').close()
            open(val_logfile, 'w').close()

    ddp = int(os.environ.get('RANK', -1)) != -1  # is this a ddp run?
    if ddp:
        assert torch.cuda.is_available(), "We need CUDA for DDP"
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        device = torch.device(args.device)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    config = MovieGenConfig()
    model = MovieGen(config)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
    print0(f"Total Parameters: {total_params:,}")
    print0(f"Trainable Parameters: {trainable_params:,}")
    if args.ckpt:
        ignore_keys = list()
        # The checkpoint from LDM needs to ignore certain layers
        if args.ckpt_from_ldm == 1:
            ignore_keys = [
                "encoder.conv_out",
                "decoder.conv_in",
                "quant_conv",
                "post_quant_conv",
            ]

        # if args.resume == 0:
        #     ignore_keys.append("loss")

        model.from_pretrained(args.ckpt, ignore_keys=ignore_keys)

    model.train()
    if args.compile:
        model = torch.compile(model)
    model.to(device)

    trainset = Dataset(args.train_dir, T=args.max_frames, size=args.resolution)
    valset = Dataset(args.val_dir, T=args.max_frames,
                     size=args.resolution, train=False)
    train_sampler = DistributedSampler(trainset, shuffle=True) if ddp else None
    val_sampler = DistributedSampler(valset) if ddp else None
    train_loader = DataLoader(trainset, shuffle=(train_sampler is None),
                              num_workers=4, sampler=train_sampler)
    val_loader = DataLoader(valset, shuffle=False,
                            num_workers=4, sampler=val_sampler)

    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank],
                    find_unused_parameters=True)
    unwrapped_model = model.module if ddp else model

    optimizer = unwrapped_model.configure_optimizers(
        lr=args.lr, weight_decay=args.weight_decay,
        betas=(0.5, 0.9))

    torch.cuda.reset_peak_memory_stats()
    timings = list()
    if args.inference_only:
        print0("Starting inference only.")
    else:
        print0("Starting training.")

    start_step = 0
    if args.resume == 1:
        start_step = torch.load(args.ckpt)["step"]
    trainset_size = len(trainset) // ddp_world_size if ddp else len(trainset)
    for step in range(start_step, args.num_iterations + 1):
        if step % trainset_size == 0:
            train_sampler.set_epoch(step % len(trainset))
            train_iter = iter(train_loader)
        t0 = time.time()
        last_step = (step == args.num_iterations - 1)

        # once in a while evaluate the validation dataset
        if ((args.val_loss_every > 0 and step % args.val_loss_every == 0) or last_step or args.inference_only) and (val_loader is not None) and master_process:  # noqa
            model.eval()
            with torch.no_grad():
                val_loss = 0.
                for vali, (x, mask) in enumerate(val_loader):
                    if vali >= args.val_max_steps:
                        break
                    x, mask = x.to(device), mask.to(device)
                    loss, loss_dict = model(x, mask, "val", 1, step)
                    val_loss += loss.item()
                    metrics = None
                val_loss /= args.val_max_steps
            # log to console and to file
            print0(f"val loss {val_loss}")
            metrics = {k: np.mean(v) for k, v in metrics.items()}
            x = " | ".join([f'{x}: {y:.4f}' for x, y in metrics.items()])
            print0(f" val metrics: {x}")
            if args.verbose_loss:
                x = " | ".join([f'{x}: {y.item():.4f}' for x, y in loss_dict.items()])  # noqa
                print0(f"    val verbose loss: {x}")
            if val_logfile is not None:
                with open(val_logfile, "a") as f:
                    f.write(f"{step},")
                    f.write(f"val/loss,{val_loss},")
                    f.write(",".join([f'{x},{y.item():.4f}' for x, y in loss_dict.items()]))  # noqa
                    f.write("\n")
                    f.write(",".join([f'{x},{y:.4f}' for x, y in metrics.items()]))  # noqa
                    f.write("\n")

        if last_step or args.inference_only:
            break

        model.train()
        # --------------- TRAINING SECTION BEGIN -----------------
        optimizer.zero_grad(set_to_none=True)

        # fetch a batch
        x, mask = next(train_iter)
        x, mask = x.to(device), mask.to(device)
        loss, loss_dict = model(x, mask, step)
        loss.backward()
        # if ddp:
        #     dist.all_reduce(loss, op=dist.ReduceOp.AVG)
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.grad_clip)
        # step the optimizer
        optimizer.step()

        torch.cuda.synchronize()
        t1 = time.time()
        print0(f"""step {step+1:4d}/{args.num_iterations} | \
               train loss {loss.item():.6f} | \
               norm {norm:.4f} | \
               """)
        if args.verbose_loss:
            x = " | ".join([f'{x}: {y.item():.4f}' for x, y in loss_dict.items()])  # noqa
            print0(f"    verbose loss: {x}")
        if master_process and train_logfile is not None:
            with open(train_logfile, "a") as f:
                f.write(f"{step},")
                f.write(f"train/loss,{loss.item()},")
                f.write(",".join([f'{x},{y.item():.4f}' for x, y in loss_dict.items()]))  # noqa
                f.write("\n")

        # keep track of smooth timings, last 20 iterations
        if step > 0 and step > args.num_iterations - 20:
            timings.append(t1 - t0)

        if master_process and step > 0 and args.ckpt_freq > 0 and step % args.ckpt_freq == 0:
            torch.save({"step": step, "state_dict": unwrapped_model.state_dict()},
                       args.output_dir / f"tae_{step}.ckpt")

    # print the average of the last 20 timings, to get something smooth-ish
    timings = timings[-20:]
    print0(f"final {len(timings)} iters avg: {np.mean(timings)*1000:.3f}ms")
    print0(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")  # NOQA

    if master_process:
        torch.save({"step": step, "state_dict": model.state_dict()},
                   args.output_dir / "tae_last.ckpt")

    cleanup()
    # -------------------------------------------------------------------------
