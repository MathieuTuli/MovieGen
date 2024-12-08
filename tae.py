"""
source: https://github.com/CompVis/latent-diffusion
"""
from collections import OrderedDict, namedtuple
from dataclasses import dataclass
from typing import Tuple, List
from pathlib import Path
import functools
import requests
import hashlib
import os

from torchvision import models
from einops import rearrange
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import torch

"""
-------------------------------------------------------------------------------
modules
-------------------------------------------------------------------------------
"""


def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(num_groups=num_groups,
                              num_channels=in_channels,
                              eps=1e-6, affine=True)


class TemporalConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        pad = kwargs.get("padding", 1)
        kwargs["padding"] = 0
        super().__init__(*args, **kwargs)
        self.pad = torch.nn.ReplicationPad1d(pad)

    def forward(self, x):
        x = self.pad(x)
        x = super().forward(x)
        return x


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(
            qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)',
            heads=self.heads, qkv=3)
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w',
                        heads=self.heads, h=h, w=w)
        return self.to_out(out)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv, with_temporal):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)
            # temporal inflation
            self.temp_conv = None
            if with_temporal:
                self.temp_conv = TemporalConv1d(
                    in_channels, in_channels, kernel_size=3, stride=1)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        x = torch.nn.functional.interpolate(
            x, scale_factor=2.0, mode="nearest")
        _, C, H, W = x.shape
        x = x.view(B, T, C, H, W)
        if self.with_conv:
            B, T, C, H, W = x.shape
            x = x.view(B * T, C, H, W)
            x = self.conv(x)
            _, C, H, W = x.shape
            x = x.view(B, T, C, H, W).permute(0, 3, 4, 2, 1).contiguous()
            x = x.view(B * H * W, C, T)
            # REVISIT:
            if self.temp_conv is not None:
                x = torch.nn.functional.interpolate(
                    x, scale_factor=2.0, mode="nearest")
                x = self.temp_conv(x)
            _, C, T = x.shape
            x = x.view(B, H, W, C, T).permute(0, 4, 3, 1, 2).contiguous()
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv, with_temporal):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)
            # temporal inflation
            self.temp_conv = None
            if with_temporal:
                self.temp_conv = TemporalConv1d(
                    in_channels, in_channels, kernel_size=3, stride=2)

    def forward(self, x):
        if self.with_conv:
            B, T, C, H, W = x.shape
            x = x.view(B * T, C, H, W)
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
            # temporal inflation
            _, C, H, W = x.shape
            x = x.view(B, T, C, H, W).permute(0, 3, 4, 2, 1).contiguous()
            x = x.view(B * H * W, C, T)
            if self.temp_conv is not None:
                x = self.temp_conv(x)
            _, C, T = x.shape
            x = x.view(B, H, W, C, T).permute(0, 4, 3, 1, 2).contiguous()
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class TemporalResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        # temporal inflation
        self.temp_conv1 = TemporalConv1d(
            out_channels, out_channels, kernel_size=3, stride=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        # temporal inflation
        self.temp_conv2 = TemporalConv1d(
            out_channels, out_channels, kernel_size=3, stride=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
                self.temp_conv_shortcut = TemporalConv1d(
                    out_channels, out_channels,
                    kernel_size=3, stride=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)
                # temporal inflation
                self.temp_conv_shortcut = TemporalConv1d(
                    out_channels, out_channels,
                    kernel_size=1, stride=1, padding=0)

    def forward(self, x, temb):
        h = x
        B, T, C, H, W = h.shape
        h = h.view(B * T, C, H, W)
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)
        # temporal inflation
        _, C, H, W = h.shape
        h = h.view(B, T, C, H, W).permute(0, 3, 4, 2, 1).contiguous()
        h = h.view(B * H * W, C, T)
        h = self.temp_conv1(h)
        _, C, T = h.shape
        h = h.view(B, H, W, C, T).permute(0, 4, 3, 1, 2).contiguous()

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        B, T, C, H, W = h.shape
        h = h.view(B * T, C, H, W)
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)
        # temporal inflation
        _, C, H, W = h.shape
        h = h.view(B, T, C, H, W).permute(0, 3, 4, 2, 1).contiguous()
        h = h.view(B * H * W, C, T)
        h = self.temp_conv2(h)
        _, C, T = h.shape
        h = h.view(B, H, W, C, T).permute(0, 4, 3, 1, 2).contiguous()

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                B, T, C, H, W = x.shape
                x = x.view(B * T, C, H, W)
                x = self.conv_shortcut(x)
                _, C, H, W = x.shape
                x = x.view(B, T, C, H, W).permute(0, 3, 4, 2, 1).contiguous()
                x = x.view(B * H * W, C, T)
                # temporal inflation
                x = self.temp_conv_shortcut(x)
                _, C, T = x.shape
                x = x.view(B, H, W, C, T).permute(0, 4, 3, 1, 2).contiguous()
            else:
                B, T, C, H, W = x.shape
                x = x.view(B * T, C, H, W)
                x = self.nin_shortcut(x)
                _, C, H, W = x.shape
                x = x.view(B, T, C, H, W).permute(0, 3, 4, 2, 1).contiguous()
                x = x.view(B * H * W, C, T)
                x = self.temp_conv_shortcut(x)
                _, C, T = x.shape
                x = x.view(B, H, W, C, T).permute(0, 4, 3, 1, 2).contiguous()

        return x + h


class TemporalAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = TemporalConv1d(in_channels,
                                in_channels,
                                kernel_size=1,
                                stride=1,
                                padding=0)
        self.k = TemporalConv1d(in_channels,
                                in_channels,
                                kernel_size=1,
                                stride=1,
                                padding=0)
        self.v = TemporalConv1d(in_channels,
                                in_channels,
                                kernel_size=1,
                                stride=1,
                                padding=0)
        self.proj_out = TemporalConv1d(in_channels,
                                       in_channels,
                                       kernel_size=1,
                                       stride=1,
                                       padding=0)

    def forward(self, x):
        return x
        # REVISIT: confirm attention shapes
        h_ = x
        B, T, C, H, W = h_.shape
        h_ = h_.permute(0, 3, 4, 2, 1).contiguous().view(B * H * W, C, T)
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        # REVISIT: confirm attention shapes
        b, c, t = q.shape
        q = q.permute(0, 2, 1)   # b,t,c
        w_ = torch.bmm(q, k)     # b,t,t   w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        w_ = w_.permute(0, 2, 1)   # b,,t (first hw of k, second of q)
        # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = torch.bmm(v, w_)

        h_ = self.proj_out(h_)
        _, C, T = h_.shape
        h_ = h_.view(B, H, W, C, T).permute(0, 4, 3, 1, 2).contiguous()
        return x + h_


class LinAttnBlock(LinearAttention):
    """to match AttnBlock usage"""

    def __init__(self, in_channels):
        super().__init__(dim=in_channels, heads=1, dim_head=in_channels)


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        B, T, C, H, W = h_.shape
        h_ = h_.view(B * T, C, H, W)
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h*w)
        q = q.permute(0, 2, 1)   # b,hw,c
        k = k.reshape(b, c, h*w)  # b,c,hw
        w_ = torch.bmm(q, k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h*w)
        w_ = w_.permute(0, 2, 1)   # b,hw,hw (first hw of k, second of q)
        # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)
        h_ = h_.view(B, T, C, H, W)

        return x+h_


def make_attn(in_channels, attn_type="vanilla"):
    assert attn_type in ["vanilla", "linear",
                         "none"], f'attn_type {attn_type} unknown'
    if attn_type == "vanilla":
        return AttnBlock(in_channels)
    elif attn_type == "none":
        return nn.Identity(in_channels)
    else:
        return LinAttnBlock(in_channels)


class TemporalEncoder(nn.Module):
    def __init__(self,
                 *,
                 ch,
                 out_ch,
                 ch_mult=(1, 2, 4, 8),
                 temporal_scaling_offset,
                 num_res_blocks,
                 attn_resolutions,
                 dropout=0.0,
                 resamp_with_conv=True,
                 in_channels,
                 resolution,
                 z_channels,
                 double_z=True,
                 use_linear_attn=False,
                 attn_type="vanilla",
                 **ignore_kwargs):
        super().__init__()
        if use_linear_attn:
            attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)
        # temporal inflation
        self.temp_conv_in = TemporalConv1d(
            self.ch, self.ch, kernel_size=3, stride=1)

        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn_dict = OrderedDict()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(TemporalResnetBlock(in_channels=block_in,
                                                 out_channels=block_out,
                                                 temb_channels=self.temb_ch,
                                                 dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn_dict.update(
                        {str(i_block): make_attn(block_in,
                                                 attn_type=attn_type)})
                    # temporal inflation
                    attn_dict.update(
                        {f"temp_attn_{i_block}": TemporalAttention(
                            block_in)})
            down = nn.Module()
            down.block = block
            down.attn = nn.ModuleDict(attn_dict)
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(
                    block_in, resamp_with_conv,
                    i_level < self.num_resolutions - temporal_scaling_offset)  # NOQA
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = TemporalResnetBlock(in_channels=block_in,
                                               out_channels=block_in,
                                               temb_channels=self.temb_ch,
                                               dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        # temporal inflation
        self.mid.temp_attn_1 = TemporalAttention(block_in)
        self.mid.block_2 = TemporalResnetBlock(in_channels=block_in,
                                               out_channels=block_in,
                                               temb_channels=self.temb_ch,
                                               dropout=dropout)

        # end
        self.norm_out = Normalize(block_in)
        block_out = 2*z_channels if double_z else z_channels
        self.conv_out = torch.nn.Conv2d(block_in,
                                        block_out,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)
        # temporal inflation
        self.temp_conv_out = TemporalConv1d(
            block_out, block_out, kernel_size=3, stride=1)

    def forward(self, x):
        # timestep embedding
        temb = None

        # downsampling
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        hs = self.conv_in(x)
        # temporal inflation
        _, C, H, W = hs.shape
        hs = hs.view(B, T, C, H, W).permute(0, 3, 4, 2, 1).contiguous()
        hs = hs.view(B * H * W, C, T)
        hs = self.temp_conv_in(hs)
        _, C, T = hs.shape
        hs = [hs.view(B, H, W, C, T).permute(0, 4, 3, 1, 2).contiguous()]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    # B, T, C, H, W = h.shape
                    # h = h.view(B * T, C, H, W)
                    h = self.down[i_level].attn[str(i_block)](h)
                    # h = h.view(B, T, C, H, W)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        # B, T, C, H, W = h.shape
        # h = h.view(B * T, C, H, W)
        h = self.mid.attn_1(h)
        # temporal inflation
        h = self.mid.temp_attn_1(h)
        # h = h.view(B, T, C, H, W)
        h = self.mid.block_2(h, temb)

        # end
        B, T, C, H, W = h.shape
        h = h.view(B * T, C, H, W)
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        # temporal inflation
        _, C, H, W = h.shape
        h = h.view(B, T, C, H, W).permute(0, 3, 4, 2, 1).contiguous()
        h = h.view(B * H * W, C, T)
        h = self.temp_conv_out(h)
        _, C, T = h.shape
        h = h.view(B, H, W, C, T).permute(0, 4, 3, 1, 2).contiguous()
        return h


class TemporalDecoder(nn.Module):
    def __init__(self,
                 *,
                 ch,
                 out_ch,
                 ch_mult=(1, 2, 4, 8),
                 temporal_scaling_offset,
                 num_res_blocks,
                 attn_resolutions,
                 dropout=0.0,
                 resamp_with_conv=True,
                 in_channels,
                 resolution,
                 z_channels,
                 give_pre_end=False,
                 tanh_out=False,
                 use_linear_attn=False,
                 attn_type="vanilla",
                 **ignorekwargs):
        super().__init__()
        if use_linear_attn:
            attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out

        # compute in_ch_mult, block_in and curr_res at lowest res
        # in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1, z_channels, curr_res, curr_res)

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)
        # temporal inflation
        self.temp_conv_in = TemporalConv1d(
            block_in, block_in, kernel_size=3, stride=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = TemporalResnetBlock(in_channels=block_in,
                                               out_channels=block_in,
                                               temb_channels=self.temb_ch,
                                               dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        # temporal inflation
        self.mid.temp_attn_1 = TemporalAttention(block_in)
        self.mid.block_2 = TemporalResnetBlock(in_channels=block_in,
                                               out_channels=block_in,
                                               temb_channels=self.temb_ch,
                                               dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn_dict = OrderedDict()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(TemporalResnetBlock(in_channels=block_in,
                                                 out_channels=block_out,
                                                 temb_channels=self.temb_ch,
                                                 dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn_dict.update(
                        {str(i_block): make_attn(block_in,
                                                 attn_type=attn_type)})
                    # temporal inflation
                    attn_dict.update(
                        {f"temp_attn_{i_block}": TemporalAttention(
                            block_in)})
            up = nn.Module()
            up.block = block
            up.attn = nn.ModuleDict(attn_dict)
            if i_level != 0:
                up.upsample = Upsample(
                    block_in, resamp_with_conv,
                    i_level >= temporal_scaling_offset)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)
        # temporal inflation
        self.temp_conv_out = TemporalConv1d(
            out_ch, out_ch, kernel_size=3, stride=1)

    def forward(self, z):
        # assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        B, T, C, H, W = z.shape
        z = z.view(B * T, C, H, W)
        h = self.conv_in(z)
        # temporal inflation
        _, C, H, W = h.shape
        h = h.view(B, T, C, H, W).permute(0, 3, 4, 2, 1).contiguous()
        h = h.view(B * H * W, C, T)
        h = self.temp_conv_in(h)
        _, C, T = h.shape
        h = h.view(B, H, W, C, T).permute(0, 4, 3, 1, 2).contiguous()

        # middle
        h = self.mid.block_1(h, temb)
        # B, T, C, H, W = h.shape
        # h = h.view(B * T, C, H, W)
        h = self.mid.attn_1(h)
        # temporal inflation
        self.mid.temp_attn_1(h)
        # h = h.view(B, T, C, H, W)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    # B, T, C, H, W = h.shape
                    # h = h.view(B * T, C, H, W)
                    h = self.up[i_level].attn[str(i_block)](h)
                    # h = h.view(B, T, C, H, W)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        B, T, C, H, W = h.shape
        h = h.view(B * T, C, H, W)
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)

        # temporal inflation
        _, C, H, W = h.shape
        h = h.view(B, T, C, H, W).permute(0, 3, 4, 2, 1).contiguous()
        h = h.view(B * H * W, C, T)
        h = self.temp_conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)
        _, C, T = h.shape
        h = h.view(B, H, W, C, T).permute(0, 4, 3, 1, 2).contiguous()
        return h


class DiagonalGaussianDistribution:
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        # REVISIT: changed to handle temporal dimension from dim=1 to 2
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=2)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(
                self.mean).to(device=self.parameters.device)

    def sample(self):
        x = self.mean + self.std * torch.randn(
            self.mean.shape).to(device=self.parameters.device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                mean = self.mean.flatten(0, 1)
                var = self.var.flatten(0, 1)
                logvar = self.logvar.flatten(0, 1)
                return 0.5 * torch.sum(torch.pow(mean, 2)
                                       + var - 1.0 - logvar,
                                       dim=[1, 2, 3])
            else:
                raise NotImplementedError
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2, 3])

    def nll(self, sample, dims=[1, 2, 3]):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar +
            torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)

    def mode(self):
        return self.mean


"""
-------------------------------------------------------------------------------
copied from https://github.com/CompVis/taming-transformers/tree/master
-------------------------------------------------------------------------------
"""


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class ActNorm(nn.Module):
    def __init__(self, num_features, logdet=False, affine=True,
                 allow_reverse_init=False):
        assert affine
        super().__init__()
        self.logdet = logdet
        self.loc = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.allow_reverse_init = allow_reverse_init

        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))

    def initialize(self, input):
        with torch.no_grad():
            flatten = input.permute(
                1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            mean = (
                flatten.mean(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )
            std = (
                flatten.std(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input, reverse=False):
        if reverse:
            return self.reverse(input)
        if len(input.shape) == 2:
            input = input[:, :, None, None]
            squeeze = True
        else:
            squeeze = False

        _, _, height, width = input.shape

        if self.training and self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        h = self.scale * (input + self.loc)

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)

        if self.logdet:
            log_abs = torch.log(torch.abs(self.scale))
            logdet = height*width*torch.sum(log_abs)
            logdet = logdet * torch.ones(input.shape[0]).to(input)
            return h, logdet

        return h

    def reverse(self, output):
        if self.training and self.initialized.item() == 0:
            if not self.allow_reverse_init:
                raise RuntimeError(
                    "Initializing ActNorm in reverse direction is "
                    "disabled by default. allow_reverse_init=True to enable."
                )
            else:
                self.initialize(output)
                self.initialized.fill_(1)

        if len(output.shape) == 2:
            output = output[:, :, None, None]
            squeeze = True
        else:
            squeeze = False

        h = output / self.scale - self.loc

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)
        return h


class NLayerDiscriminator(nn.Module):
    """# noqa Defines a PatchGAN discriminator as in Pix2Pix
        --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """

    def __init__(self, input_nc=3, ndf=64, n_layers=3, use_actnorm=False):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if not use_actnorm:
            norm_layer = nn.BatchNorm2d
        else:
            norm_layer = ActNorm
        # no need to use bias as BatchNorm2d has affine parameters
        if isinstance(norm_layer, functools.partial):
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw,
                              stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        # gradually increase the number of filters
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(
                    ndf * nf_mult_prev, ndf * nf_mult,
                    kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(
                ndf * nf_mult, 1, kernel_size=kw,
                stride=1, padding=padw)]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.main(input)


class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift', torch.Tensor(
            [-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor(
            [.458, .448, .450])[None, :, None, None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Module):
    """ A single linear layer which does a 1x1 conv """

    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()
        layers = [nn.Dropout(), ] if (use_dropout) else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1,
                             padding=0, bias=False), ]
        self.model = nn.Sequential(*layers)


class vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(vgg16, self).__init__()
        # REVISIT: weights here idk - updated api
        vgg_pretrained_features = models.vgg16(
                weights=models.VGG16_Weights.IMAGENET1K_V1  # pretrained
                ).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple(
            "VggOutputs",
            ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2,
                          h_relu3_3, h_relu4_3, h_relu5_3)
        return out


URL_MAP = {
    "vgg_lpips": "https://heibox.uni-heidelberg.de/f/607503859c864bc1b30b/?dl=1"
}

CKPT_MAP = {
    "vgg_lpips": "vgg.pth"
}

MD5_MAP = {
    "vgg_lpips": "d507d7349b931f0638a25a48a722f98a"
}


def download(url, local_path, chunk_size=1024):
    os.makedirs(os.path.split(local_path)[0], exist_ok=True)
    with requests.get(url, stream=True) as r:
        total_size = int(r.headers.get("content-length", 0))
        with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
            with open(local_path, "wb") as f:
                for data in r.iter_content(chunk_size=chunk_size):
                    if data:
                        f.write(data)
                        pbar.update(chunk_size)


def md5_hash(path):
    with open(path, "rb") as f:
        content = f.read()
    return hashlib.md5(content).hexdigest()


def get_ckpt_path(name, root, check=False):
    assert name in URL_MAP
    path = os.path.join(root, CKPT_MAP[name])
    if not os.path.exists(path) or (
            check and not md5_hash(path) == MD5_MAP[name]):
        print("Downloading {} model from {} to {}".format(
            name, URL_MAP[name], path))
        download(URL_MAP[name], path)
        md5 = md5_hash(path)
        assert md5 == MD5_MAP[name], md5
    return path


def normalize_tensor(x, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x**2, dim=1, keepdim=True))
    return x/(norm_factor+eps)


def spatial_average(x, keepdim=True):
    return x.mean([2, 3], keepdim=keepdim)


class LPIPS(nn.Module):
    # Learned perceptual metric
    def __init__(self, use_dropout=True):
        super().__init__()
        self.scaling_layer = ScalingLayer()
        self.chns = [64, 128, 256, 512, 512]  # vg16 features
        self.net = vgg16(pretrained=True, requires_grad=False)
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
        self.load_from_pretrained()
        for param in self.parameters():
            param.requires_grad = False

    def load_from_pretrained(self, name="vgg_lpips"):
        ckpt = get_ckpt_path(
                name, "pretrained-weights/taming/modules/autoencoder/lpips")
        self.load_state_dict(torch.load(ckpt, map_location=torch.device("cpu"),
                                        weights_only=True), strict=False)
        print("loaded pretrained LPIPS loss from {}".format(ckpt))

    @classmethod
    def from_pretrained(cls, name="vgg_lpips"):
        if name != "vgg_lpips":
            raise NotImplementedError
        model = cls()
        ckpt = get_ckpt_path(name)
        model.load_state_dict(torch.load(ckpt, map_location=torch.device("cpu"),
                                         weights_only=True, ), strict=False)
        return model

    def forward(self, input, target):
        in0_input, in1_input = (self.scaling_layer(
            input), self.scaling_layer(target))
        outs0, outs1 = self.net(in0_input), self.net(in1_input)
        feats0, feats1, diffs = {}, {}, {}
        lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        for kk in range(len(self.chns)):
            feats0[kk], feats1[kk] = normalize_tensor(
                outs0[kk]), normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        res = [spatial_average(lins[kk].model(diffs[kk]), keepdim=True)
               for kk in range(len(self.chns))]
        val = res[0]
        for l in range(1, len(self.chns)):
            val += res[l]
        return val


"""
-------------------------------------------------------------------------------
copied from https://github.com/CompVis/taming-transformers/tree/master
-------------------------------------------------------------------------------
"""


def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(torch.nn.functional.relu(1. - logits_real))
    loss_fake = torch.mean(torch.nn.functional.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss


class LPIPSWithDiscriminator(nn.Module):
    def __init__(self, disc_start, logvar_init=0.0, kl_weight=1.0,
                 pixelloss_weight=1.0, disc_num_layers=3,
                 disc_in_channels=3, disc_factor=1.0, disc_weight=1.0,
                 perceptual_weight=1.0, use_actnorm=False,
                 disc_conditional=False, disc_loss="hinge",
                 outlier_scaling_factor: int = 3,
                 outlier_loss_weight: float = 1e-5):

        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight
        # output log variance
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

        self.discriminator = NLayerDiscriminator(
            input_nc=disc_in_channels,
            n_layers=disc_num_layers,
            use_actnorm=use_actnorm
        ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else \
            vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional
        self.outlier_loss_weight = outlier_loss_weight
        self.outlier_scaling_factor = outlier_scaling_factor

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(
                nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(
                g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(
                nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(
                g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(self, inputs, reconstructions, posteriors, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train",
                weights=None):
        inputs = inputs.flatten(0, 1)
        reconstructions = reconstructions.flatten(0, 1)
        rec_loss = torch.abs(inputs.contiguous() -
                             reconstructions.contiguous())
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(
                inputs.contiguous(), reconstructions.contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss

        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        weighted_nll_loss = nll_loss
        if weights is not None:
            weighted_nll_loss = weights*nll_loss
        weighted_nll_loss = torch.sum(
            weighted_nll_loss) / weighted_nll_loss.shape[0]
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        B, T, C, H, W = posteriors.mean.shape
        X = posteriors.mean.view(B * T, C, H, W)
        outlier_loss = 0.
        for b in range(B):
            for i in range(H):
                for j in range(W):
                    outlier_loss += torch.max(
                            (X[b, :, i, j] - X.mean()).norm() -
                            (self.outlier_scaling_factor * X.std()).norm(),
                            torch.tensor([0.], device=inputs.device))
        outlier_loss /= H * W

        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(
                    torch.cat((reconstructions.contiguous(), cond), dim=1))
            g_loss = -torch.mean(logits_fake)

            if self.disc_factor > 0.0:
                try:
                    d_weight = self.calculate_adaptive_weight(
                        nll_loss, g_loss, last_layer=last_layer)
                except RuntimeError:
                    assert not self.training
                    d_weight = torch.tensor(0.0)
            else:
                d_weight = torch.tensor(0.0)

            disc_factor = adopt_weight(
                self.disc_factor, global_step,
                threshold=self.discriminator_iter_start)
            loss = weighted_nll_loss + self.kl_weight * \
                kl_loss + d_weight * disc_factor * g_loss + \
                outlier_loss * self.outlier_loss_weight

            log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                   "{}/logvar".format(split): self.logvar.detach(),
                   "{}/kl_loss".format(split): kl_loss.detach().mean(),
                   "{}/nll_loss".format(split): nll_loss.detach().mean(),
                   "{}/rec_loss".format(split): rec_loss.detach().mean(),
                   "{}/d_weight".format(split): d_weight.detach(),
                   "{}/disc_factor".format(split): torch.tensor(disc_factor),
                   "{}/g_loss".format(split): g_loss.detach().mean(),
                   "{}/outlier_loss".format(split): outlier_loss.detach(),
                   }
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            if cond is None:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(
                    reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(
                    torch.cat((inputs.contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(
                    torch.cat((reconstructions.contiguous().detach(),
                               cond), dim=1))

            disc_factor = adopt_weight(
                self.disc_factor, global_step,
                threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {
                "{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                "{}/logits_real".format(split): logits_real.detach().mean(),
                "{}/logits_fake".format(split): logits_fake.detach().mean()
            }
            return d_loss, log


"""
-------------------------------------------------------------------------------
model
-------------------------------------------------------------------------------
"""


@dataclass
class TAEConfig:
    version: str = "1.0"
    embed_dim: int = 16
    z_channels: int = 16
    double_z: bool = True
    resolution: int = 256
    in_channels: int = 3
    out_ch: int = 3
    ch: int = 128
    ch_mult: Tuple[int] = (1, 2, 4, 4)  # num_down = len(ch_mult)-1
    temporal_scaling_offset: int = 0
    num_res_blocks: int = 2
    attn_resolutions: Tuple[int] = tuple()
    dropout: float = 0.0
    scale_factor: float = 1.
    strict: bool = False
    loss_disc_start: int = 5001
    loss_kl_weight: float = 1e-6
    loss_disc_weight: float = 0.5

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)


class TAE(nn.Module):
    def __init__(self, config: TAEConfig):
        super().__init__()
        self.config = config
        self.encoder = TemporalEncoder(
            ch=config.ch,
            out_ch=config.out_ch,
            ch_mult=config.ch_mult,
            temporal_scaling_offset=config.temporal_scaling_offset,
            num_res_blocks=config.num_res_blocks,
            dropout=config.dropout,
            embed_dim=config.embed_dim,
            in_channels=config.in_channels,
            resolution=config.resolution,
            z_channels=config.z_channels,
            double_z=config.double_z,
            attn_resolutions=config.attn_resolutions)
        self.decoder = TemporalDecoder(
            ch=config.ch,
            out_ch=config.out_ch,
            ch_mult=config.ch_mult,
            temporal_scaling_offset=config.temporal_scaling_offset,
            num_res_blocks=config.num_res_blocks,
            embed_dim=config.embed_dim,
            dropout=config.dropout,
            in_channels=config.in_channels,
            resolution=config.resolution,
            z_channels=config.z_channels,
            double_z=config.double_z,
            attn_resolutions=config.attn_resolutions)
        self.strict = config.strict

        self.quant_conv = torch.nn.Conv2d(
            2 * config.z_channels, 2 * config.embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(
            config.embed_dim, config.z_channels, 1)
        self.embed_dim = config.embed_dim
        self.scale_factor = config.scale_factor
        self.loss = LPIPSWithDiscriminator(
                disc_start=config.loss_disc_start,
                kl_weight=config.loss_kl_weight,
                disc_weight=config.loss_disc_weight)

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

        # REVISIT: update this to train tae
        for k in list(sd.keys()):
            if k.startswith("loss"):
                del sd[k]

        temp_sd = self.state_dict()
        for k in temp_sd:
            if "temp_" in k and k not in sd:
                sd[k] = temp_sd[k]

        self.load_state_dict(sd, strict=self.strict)

    def get_encoding(self, encoder_posterior):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(
                "encoder_posterior of type " +
                f"{type(encoder_posterior)} not yet implemented")
        return self.scale_factor * z

    def encode(self, x):
        # temporal accounting
        x = self.encoder(x)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        moments = self.quant_conv(x)
        _, C, H, W = moments.shape
        moments = moments.view(B, T, C, H, W)
        # REVISIT: this
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        # temporal accounting
        B, T, C, H, W = z.shape
        z = z.view(B * T, C, H, W)
        z = self.post_quant_conv(z)
        _, C, H, W = z.shape
        z = z.view(B, T, C, H, W)
        dec = self.decoder(z)
        # temporal accounting
        return dec

    def forward(self, inputs, split, optimizer_idx, step,
                sample_posterior=True):
        # temporal accounting
        posterior = self.encode(inputs)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        Tprime = inputs.shape[1]
        assert dec.shape[1] >= Tprime
        # limit = dec.shape[1] - Tprime
        dec = dec[:, :Tprime]

        loss, log_dict = self.loss(
                inputs, dec, posterior, optimizer_idx,  # 1 for val
                step, last_layer=self.last_layer, split="train")
        return dec, posterior, loss

    def configure_optimizers(
            self, lr: float, weight_decay: float,
            betas: Tuple[float, float] = (0.5, 0.9)):
        opt_ae = torch.optim.Adam(list(self.encoder.parameters()) +
                                  list(self.decoder.parameters()) +
                                  list(self.quant_conv.parameters()) +
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=betas)
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=betas)
        return opt_ae, opt_disc

    @property
    def last_layer(self):
        return self.decoder.conv_out.weight
