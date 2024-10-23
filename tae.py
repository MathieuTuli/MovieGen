"""
source: https://github.com/CompVis/latent-diffusion
"""
from collections import OrderedDict
from dataclasses import dataclass
from typing import Tuple, List
from pathlib import Path

from einops import rearrange
import torch.nn as nn
import numpy as np
import torch


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
                return 0.5 * torch.sum(torch.pow(self.mean, 2)
                                       + self.var - 1.0 - self.logvar,
                                       dim=[1, 2, 3])
            else:
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

        # REVISIT: self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def loss(self, *args, **kwargs):
        raise NotImplementedError("VAE training not supported")

    def from_pretrained(self, ckpt: Path,
                        ignore_keys: List[str] = None,):
        ignore_keys = ignore_keys or list()
        sd = torch.load(ckpt, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]

        # REVISIT: update this to train tae
        for k in list(sd.keys()):
            if k.startswith("loss"):
                del sd[k]

        temp_sd = self.state_dict()
        for k in temp_sd:
            if "temp_" in k:
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

    def forward(self, input, sample_posterior=True):
        B, T, C, H, W = input.shape
        # temporal accounting
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        B, T, C, H, W = z.shape
        dec = self.decode(z)
        return dec, posterior

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(
            memory_format=torch.contiguous_format).float()
        return x

    def training_step(self, batch, batch_idx, optimizer_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)

        if optimizer_idx == 0:
            # train encoder+decoder+logvar
            aeloss, log_dict_ae = self.loss(
                inputs, reconstructions, posterior, optimizer_idx,
                self.global_step, last_layer=self.get_last_layer(),
                split="train")
            self.log("aeloss", aeloss, prog_bar=True,
                     logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False,
                          logger=True, on_step=True, on_epoch=False)
            return aeloss

        if optimizer_idx == 1:
            # train the discriminator
            discloss, log_dict_disc = self.loss(
                inputs, reconstructions, posterior, optimizer_idx,
                self.global_step, last_layer=self.get_last_layer(),
                split="train")

            self.log("discloss", discloss, prog_bar=True,
                     logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False,
                          logger=True, on_step=True, on_epoch=False)
            return discloss

    def validation_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)
        aeloss, log_dict_ae = self.loss(
            inputs, reconstructions, posterior, 0, self.global_step,
            last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(
            inputs, reconstructions, posterior, 1, self.global_step,
            last_layer=self.get_last_layer(), split="val")

        self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters()) +
                                  list(self.decoder.parameters()) +
                                  list(self.quant_conv.parameters()) +
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight