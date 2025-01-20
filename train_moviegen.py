"""
Reference code for Movie Gen training and inference.
"""
from typing import Tuple, Optional, List
from dataclasses import dataclass
from inspect import isfunction
from functools import partial
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
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.nn import functional as F
from einops import rearrange, repeat
from torch import einsum
from PIL import Image

import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import torchvision
import torch
import cv2

from text_encoder import TextEncoder, TextEncoderConfig
from tae import TAE, TAEConfig
from util import (
    dump_dict_to_yaml, asspath, mkpath, print0, cleanup, signal_handler)


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
            mask = repeat(mask, 'b j -> (b h) j ()', h=h)
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

    def forward(self,
                x: torch.Tensor,
                ctx: torch.Tensor,
                t_emb: torch.Tensor,
                pos_emb: torch.Tensor,
                mask: torch.Tensor = None):
        # get adaln scale/shift
        # this is how time embedding is included in the model
        shift_1, scale_1, alpha_1, shift_2, scale_2, alpha_2 = \
            self.adaLN_modulation(t_emb).chunk(6, dim=1)
        x = x + pos_emb
        x = x + alpha_1 * self.attn(modulate(self.rmsnorm1(x),
                                             shift_1, scale_1), mask=mask)
        x = x + self.cross_attn(x, ctx, mask=mask)
        x = x + alpha_2 * self.mlp(modulate(self.rmsnorm2(x),
                                            shift_2, scale_2))
        return x


class Head(nn.Module):
    """
    The final layer of MovieGen
    """

    def __init__(self, config):
        super().__init__()
        self.norm = RMSNorm(config.n_embd, config.norm_eps)
        self.linear = nn.Linear(config.n_embd,
                                np.prod(config.patch_k) * config.in_channels,
                                bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.n_embd, 2 * config.n_embd, bias=True)
        )

    def forward(self, x, c):
        # in: [B, ctx_len, 6144]
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm(x), shift, scale)
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
paths
-------------------------------------------------------------------------------
"""


class OptimalTransportPath:
    def __init__(self, sig_min: float = 1e-5) -> None:
        self.sig_min = sig_min

    def sample(self, x1: torch.Tensor, x0: torch.Tensor, t: torch.Tensor
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        t = t.expand_as(x1)
        xt = x1 * t + (1 - (1 - self.sig_min) * t) * x0
        vt = x1 - (1 - self.sig_min) * x0
        return xt, vt


"""
-------------------------------------------------------------------------------
models
-------------------------------------------------------------------------------
"""


@dataclass
class MovieGenConfig:
    version: str = "1.0"
    in_channels: int = 16
    n_layer: int = 2  # 48
    n_head: int = 12  # 48
    dim_head: int = 128
    n_embd: int = 6144
    ffn_dim: int = 16384  # 6144 * 4 * 2/3
    # context_len: int = 8192  # (256 * 256 * 256) / [(8 * 8 * 8) * (1 * 2 * 2)]
    max_frames: int = 256
    spatial_resolution: int = 256
    patch_k: Tuple[int, int, int] = (1, 2, 2)
    norm_eps: float = 1e-5
    dropout: float = 0.0
    # flow matching related
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
        self.context_len = (
            config.max_frames * config.spatial_resolution ** 2) //\
            ((8 ** 3) * np.prod(config.patch_k))
        # assert config.context_len == calc_ctx_len, \
        #     f"Context length is wrong. Set to {config.context_len}. " + \
        #     f"Should be {calc_ctx_len}"
        self.config = config

        self.patchifier = nn.Conv3d(in_channels=config.in_channels,
                                    out_channels=config.n_embd,
                                    kernel_size=config.patch_k,
                                    stride=config.patch_k,
                                    bias=True)
        # self.patch_proj = nn.Sequential(
        #     nn.LayerNorm(config.in_channels),
        #     nn.Linear(config.in_channels, config.n_embd),
        #     nn.LayerNorm(config.n_embd),
        # )
        self.t_embedder = TimestepEmbedder(config.n_embd)
        # NOTE: 8-compression from TAE
        self.pos_embed_h = nn.Parameter(torch.randn(
            config.spatial_resolution // (8 * config.patch_k[1]),
            config.n_embd))
        self.pos_embed_w = nn.Parameter(torch.randn(
            config.spatial_resolution // (8 * config.patch_k[2]),
            config.n_embd))
        self.pos_embed_T = nn.Parameter(torch.randn(
            config.max_frames // (8 * config.patch_k[0]),
            config.n_embd))

        print0("Initializing auxiliary models")
        # self.text_encoder = TextEncoder(TextEncoderConfig())
        # self.text_encoder.eval()
        # for p in self.text_encoder.parameters():
        #     p.requires_grad_(False)
        self.tae = TAE(TAEConfig())
        self.tae.eval()
        for p in self.tae.parameters():
            p.requires_grad_(False)

        print0("Initializing transformer backbone")
        self.blocks = nn.ModuleList([
            Block(config) for _ in range(config.n_layer)])
        self.head = Head(config)
        # I borrow the lm head from DiT instead of the Llama head

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize patchifier like nn.Linear (instead of nn.Conv3d):
        # this is done as in DiT
        w = self.patchifier.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.patchifier.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.head.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.head.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.head.linear.weight, 0)
        nn.init.constant_(self.head.linear.bias, 0)

    def initialize_auxiliary_models(self,
                                    metaclip_ckpt: Optional[Path],
                                    tae_ckpt: Optional[Path]):
        if metaclip_ckpt is not None:
            print0(f"Loaded MetaClip weights from {metaclip_ckpt}.")
            self.text_encoder.from_pretrained(metaclip_ckpt)
        if tae_ckpt is not None:
            print0(f"Loaded TAE weights from. {tae_ckpt}")
            self.tae.from_pretrained(tae_ckpt)

    # REVISIT:
    # @classmethod
    def from_pretrained(self, ckpt: Path, ignore_keys: List[str] = None,):
        ignore_keys = ignore_keys or list()
        sd = torch.load(ckpt, map_location="cpu",
                        weights_only=False)["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("    - Deleting key {} from state_dict.".format(k))
                    del sd[k]

        temp_sd = self.state_dict()
        for k in temp_sd:
            if "temp_" in k and k not in sd:
                sd[k] = temp_sd[k]

        self.load_state_dict(sd, strict=True)

    def configure_optimizers(self,
                             lr: float,
                             weight_decay: float,
                             betas: Tuple[float, float],):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups.
        # Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay,
        #   all biases and layernorms don't.
        # decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        # nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        params = [p for n, p in param_dict.items()]
        optim_groups = [
            {'params': params, 'weight_decay': weight_decay},
            # {'params': decay_params, 'weight_decay': weight_decay},
            # {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=betas,
                                      fused=True)
        return optimizer

    def step(self, x: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor):
        return t * x_1 + (1 - (1 - self.config.sig_min) * t) * x

    def encode_frames(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.tae.encode(x, mask)  # [B, T, C, H, W]
        return x.sample()

    def decode_frames(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.tae.decode(x, mask)
        return x

    def encode_prompts(self, prompts: List[str]) -> torch.Tensor:
        tokens = self.text_encoder.tokenize(prompts, self.text_encoder.device)
        x = self.text_encoder(tokens)
        return x

    def forward(self,
                t: torch.Tensor,
                x: torch.Tensor,
                ctx: torch.Tensor,
                mask: torch.Tensor,) -> torch.Tensor:
        """
        Note that the TAE uses 8x compression and C=16
        Note that there is conflating 't' representations:
            - t is the timestep in the flow
            - T is the frame count
            - for simplicity, i will only use those notations
        @parameters
        - t: torch.Tensor (B,): timestep for flow
        - x: torch.Tensor (B, T // 8, C, H // 8, W // 8):
            TAE encoded input frames
        - ctx: torch.Tensor (B,): context - i.e. prompts from Text Encoders
        - mask: torch.Tensor (B,): attention/loss mask for padded frames

        @returns
        - vt: torch.Tensor (B, T // 8, C, H // 8, W // 8)
            velocity flow prediction latent
        """
        B, T, C, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
        x = self.patchifier(x)
        x = torch.flatten(x, start_dim=2).permute(0, 2, 1)  # [B, T*H*W, C]
        # x = self.patch_proj(x)  # [B, T*H*W, 6144]

        # pos embeddeding pulled from DiT
        assert (H < self.config.spatial_resolution and
                W < self.config.spatial_resolution and
                T < self.config.max_frames)
        pos = torch.stack(torch.meshgrid((
            torch.arange(H // self.config.patch_k[1], device=x.device),
            torch.arange(W // self.config.patch_k[2], device=x.device)
        ), indexing='ij'), dim=-1)
        pos = rearrange(pos, 'h w c -> (h w) c')
        pos = repeat(pos, 'n d -> b n d', b=B)
        # these are now [B, num patches]
        h_indices, w_indices = pos.unbind(dim=-1)
        pos_h = self.pos_embed_h[h_indices]
        pos_w = self.pos_embed_w[w_indices]

        T_indices = torch.arange(T,  # self.config.max_temporal_seq_len,
                                 device=x.device)
        T_indices = repeat(T_indices, 'n -> b n', b=x.shape[0])
        pos_T = self.pos_embed_T[T_indices]

        # REVISIT: pad to the max seq length
        # the thing is, the dataloader already pads at the batch level
        # this will pad the remaining tokens, up to context len
        # I might want to aggregate the padding logic to one spot
        pad_len = self.context_len - x.shape[1]
        x = F.pad(x, (0, 0, 0, pad_len))
        # interleave for num patches
        pad_len = self.context_len - (pos_T.shape[1] * pos_w.shape[1])
        pos_T = torch.repeat_interleave(pos_T, pos_w.shape[1], dim=1)
        pos_T = F.pad(pos_T, (0, 0, 0, pad_len))
        pos_h = repeat(pos_h, 'b n d -> b (n t) d', t=T)
        pos_h = F.pad(pos_h, (0, 0, 0, pad_len))
        pos_w = repeat(pos_w, 'b n d -> b (n t) d', t=T)
        pos_w = F.pad(pos_w, (0, 0, 0, pad_len))
        assert (pos_h.shape[1] == x.shape[1] and
                pos_h.shape[1] == pos_w.shape[1] and
                pos_h.shape[1] == pos_T.shape[1] and
                pos_h.shape[1] == self.context_len), \
            "Pos emb and inputs don't match shape.\n" + \
            f"pos_h: {pos_h.shape}, pos_w: {pos_w.shape}, pos_t: {pos_T.shape}"

        pos_emb = pos_h + pos_w + pos_T
        t_emb = self.t_embedder(t)

        # masking operations
        # Original mask is [B, T * 8], since T is the latent embedding here
        # Each mask element corresponds to
        # (spatial_res/8/2)^2 = 256 spatial patches
        mask = mask.view(B, T, 8).any(dim=2)  # [B, T/8]
        # Each temporal position expands to 256 spatial positions
        spatial_per_temporal = (
            self.config.spatial_resolution // 8 // self.config.patch_k[1] *
            self.config.spatial_resolution // 8 // self.config.patch_k[2]
        )
        # [B, T, 256]
        mask = mask.unsqueeze(-1).repeat(1, 1, spatial_per_temporal)
        mask = mask.view(B, -1)  # [B, T * spatial_per_temporal]
        # Pad to match the context length
        mask = F.pad(mask, (0, pad_len))  # [B, context_len]
        assert mask.shape[1] == self.context_len

        for i, block in enumerate(self.blocks):
            x = block(x, ctx, t_emb, pos_emb, mask=mask)

        # REVISIT: what is this supposed to be? time emb?
        x = self.head(x, t_emb)  # [B, ctxlen, patch_h * patch_w * in_channels]
        x = x.view(B,
                   self.config.max_frames // 8 // self.config.patch_k[0],
                   H // self.config.patch_k[1],
                   W // self.config.patch_k[2],
                   self.config.patch_k[0],  # this isn't necessary
                   self.config.patch_k[1],
                   self.config.patch_k[2],
                   self.config.in_channels)
        x = rearrange(
            x, 'b T H W x y z C -> b (T x) C (H y) (W z)').contiguous()
        # REVISIT: should I be doing this here
        return x[:, :T]


"""
-------------------------------------------------------------------------------
dataloader
-------------------------------------------------------------------------------
"""


class Dataset:
    def __init__(self, root: Path, T: int, image_only: bool,
                 size: int = 256, train: bool = True,):
        self.train = train
        self.image_only = image_only
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
        index = 0
        mask = torch.ones([self.T], dtype=torch.int)
        video = self.load_video(self.files[index])
        vlen = video.shape[0]
        if self.image_only:
            id = random.randint(0, vlen - 1) if self.train else 0
            id = 0
            frame = video[id]
            zero_pad = torch.zeros(self.T - 1, *frame.shape,
                                   dtype=frame.dtype)
            x = torch.cat([frame[None, :], zero_pad], dim=0)
            mask[1:] = 0
        else:
            if self.train:
                id = random.randint(0, max(vlen - self.T - 1, 0))
            id = 0
            x = video[id:id + self.T]
            if x.shape[0] < self.T:
                zero_pad = torch.zeros(self.T - x.shape[0],
                                       *x.shape[1:],
                                       dtype=frame.dtype)
                x = torch.cat([x, zero_pad], dim=0)
            mask[x.shape[1]:] = 0
        return x, ["video"] * x.shape[0], mask


"""
-------------------------------------------------------------------------------
args
-------------------------------------------------------------------------------
"""

parser = argparse.ArgumentParser()
# io
# yes you can use parser types like this
parser.add_argument("--metaclip-ckpt", type=asspath, required=False)
parser.add_argument("--tae-ckpt", type=asspath, required=False)
parser.add_argument("--output-dir", type=mkpath, default="")
parser.add_argument("--train-dir", type=asspath,
                    default="dev/data/train-overfit")
parser.add_argument("--val-dir", type=asspath, default="dev/data/val-overfit")

# checkpointing
parser.add_argument("--ckpt", type=asspath, required=False)
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
parser.add_argument("--sig-min", type=float, default=1e-5)
parser.add_argument("--image-only", type=int, default=0)

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
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
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
    else:
        args.output_dir = mkpath("tmp")

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
    config.spatial_resolution = args.resolution
    config.max_frames = args.max_frames
    model = MovieGen(config)
    model.initialize_auxiliary_models(metaclip_ckpt=args.metaclip_ckpt,
                                      tae_ckpt=args.tae_ckpt)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
    print0(f"Total Parameters: {total_params:,}")
    print0(f"Trainable Parameters: {trainable_params:,}")
    if args.ckpt:
        model.from_pretrained(args.ckpt)
        print0(f"Loaded ckpt {args.ckpt}")

    model.train()
    if args.compile:
        model = torch.compile(model)
    text_encoder = TextEncoder(TextEncoderConfig())
    text_encoder.eval()
    for p in text_encoder.parameters():
        p.requires_grad_(False)
    model.to(device)

    trainset = Dataset(args.train_dir, T=args.max_frames,
                       image_only=args.image_only == 1, size=args.resolution)
    valset = Dataset(args.val_dir, T=args.max_frames,
                     image_only=args.image_only == 1,
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
    best_val_loss = float('inf')
    if args.resume == 1:
        ckpt = torch.load(args.ckpt, weights_only=False)
        start_step = ckpt["step"]
        best_val_loss = ckpt.get("best_val_loss", best_val_loss)
        print0(f"Resuming from {start_step=}, {best_val_loss=}")
        del ckpt
    trainset_size = len(trainset) // ddp_world_size if ddp else len(trainset)
    valset_size = len(valset) // ddp_world_size if ddp else len(valset)
    path = OptimalTransportPath(sig_min=args.sig_min)
    for step in range(start_step, args.num_iterations + 1):
        if step % trainset_size == 0:
            train_iter = iter(train_loader)
            if train_sampler is not None:
                train_sampler.set_epoch(step % len(trainset))
        t0 = time.time()
        last_step = (step == args.num_iterations - 1)
        if ((args.val_loss_every > 0 and step % args.val_loss_every == 0) or
            last_step or args.inference_only) and (val_loader is not None) and master_process:  # noqa
            with torch.no_grad():
                val_loss = 0.
                val_iter = iter(val_loader)
                for vali in range(args.val_max_steps):
                    try:
                        x, prompts, mask = next(val_iter)
                    except StopIteration:
                        break
                    x, mask = x.to(device), mask.to(device)
                    prompt_embeds = text_encoder.tokenize("video", "cpu")
                    prompt_embeds = text_encoder(prompt_embeds).to(device)
                    prompt_embeds = prompt_embeds.expand(
                        x.shape[0], *prompt_embeds.shape)
                    x1 = model.encode_frames(
                        x, mask) * model.tae.scaling_factor
                    x0 = torch.randn_like(x1, device=device)
                    t = torch.rand(x1.shape[0], device=device)
                    xt, vt = path.sample(x1, x0, t)
                    v_model = model(t, xt, prompt_embeds, mask)
                    loss = torch.pow(v_model - vt, 2).mean()
                    val_loss += loss.item()
                    frames = model.decode_frames(
                        x1 / model.tae.scaling_factor, mask)
                val_loss /= (vali + 1)
            if master_process and val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({"step": step, "state_dict": model.state_dict(),
                            "best_val_loss": best_val_loss},
                           args.output_dir / "moviegen_best.ckpt",)
            print0(f"val loss {val_loss}")
            # log to console and to file
            if val_logfile is not None:
                with open(val_logfile, "a") as f:
                    f.write(f"{step},val/loss,{val_loss}\n")
            with torch.no_grad():
                # sample using midpoint solver
                latent_T = 4
                xt = torch.rand(1, latent_T, config.in_channels,
                                args.resolution // 8, args.resolution // 8,
                                dtype=torch.float32, device=device) * \
                    model.tae.scaling_factor
                T = torch.linspace(0, 1, 1001, device=device)  # sample times
                prompt_embeds = text_encoder.tokenize("video", "cpu")
                prompt_embeds = text_encoder(prompt_embeds).to(device)
                prompt_embeds = prompt_embeds.expand(
                    xt.shape[0], *prompt_embeds.shape)
                mask = torch.ones(1, 8 * latent_T,
                                  dtype=torch.int, device=device)
                odefunc = partial(model.forward, ctx=prompt_embeds, mask=mask)
                for i in range(len(T) - 1):
                    t_start = T[i].expand(xt.shape[0])
                    t_end = T[i + 1].expand(xt.shape[0])
                    xt = xt + (t_end - t_start)[..., None, None, None] * \
                        odefunc(t=t_start + (t_end - t_start) / 2,
                                x=xt + odefunc(x=xt, t=t_start) * (
                            (t_end - t_start) / 2)[..., None, None, None])
                frames = model.decode_frames(
                    xt / model.tae.scaling_factor, mask)
                for i, frame in enumerate(frames[0]):
                    torchvision.transforms.ToPILImage()(
                        frame * 0.5 + 0.5).save(
                        args.output_dir /
                        f"val_sample__step_{step}__frame_{i}.png")
        if last_step or args.inference_only:
            break

        # --------------- TRAINING SECTION BEGIN -----------------
        model.train()
        optimizer.zero_grad(set_to_none=True)

        # fetch a batch
        x, prompts, mask = next(train_iter)
        x, mask = x.to(device), mask.to(device)
        prompt_embeds = text_encoder.tokenize("video", "cpu")
        prompt_embeds = text_encoder(prompt_embeds).to(device)
        prompt_embeds = prompt_embeds.expand(x.shape[0], *prompt_embeds.shape)
        x1 = model.encode_frames(x, mask) * model.tae.scaling_factor
        x0 = torch.randn_like(x1, device=device)
        t = torch.rand(x1.shape[0], device=device)
        xt, vt = path.sample(x1, x0, t)
        v_model = model(t, xt, prompt_embeds, mask)
        loss = torch.pow(v_model - vt, 2).mean()
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
        if master_process and train_logfile is not None:
            with open(train_logfile, "a") as f:
                f.write(f"{step},")
                f.write(f"train/loss,{loss.item()},")
                f.write("\n")

        # keep track of smooth timings, last 20 iterations
        if step > 0 and step > args.num_iterations - 20:
            timings.append(t1 - t0)

        if master_process and step > 0 and args.ckpt_freq > 0 and step % args.ckpt_freq == 0:
            torch.save({"step": step, "state_dict": model.state_dict(),
                        "best_val_loss": best_val_loss},
                       args.output_dir / f"moviegen_{step}.ckpt")

    # print the average of the last 20 timings, to get something smooth-ish
    timings = timings[-20:]
    print0(f"final {len(timings)} iters avg: {np.mean(timings)*1000:.3f}ms")
    print0(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")  # NOQA

    if master_process:
        torch.save({"step": step, "state_dict": model.state_dict(),
                    "best_val_loss": best_val_loss},
                   args.output_dir / "moviegen_last.ckpt")
    cleanup()
