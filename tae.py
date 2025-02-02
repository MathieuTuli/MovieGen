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
losses
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


class NLayerDiscriminator3D(nn.Module):
    """# noqa Defines a 3D PatchGAN discriminator as in Pix2Pix
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
        super(NLayerDiscriminator3D, self).__init__()
        if not use_actnorm:
            norm_layer = nn.BatchNorm3d
        else:
            raise NotImplementedError
        # no need to use bias as BatchNorm2d has affine parameters
        if isinstance(norm_layer, functools.partial):
            use_bias = norm_layer.func != nn.BatchNorm3d
        else:
            use_bias = norm_layer != nn.BatchNorm3d

        kw = 3
        padw = 1
        sequence = [nn.Conv3d(input_nc, ndf, kernel_size=kw,
                              stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        # gradually increase the number of filters
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv3d(ndf * nf_mult_prev,
                          ndf * nf_mult,
                          kernel_size=(kw, kw, kw),
                          stride=(2 if n == 1 else 1, 2, 2),
                          padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv3d(ndf * nf_mult_prev,
                      ndf * nf_mult,
                      kernel_size=(kw, kw, kw),
                      stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        # output 1 channel prediction map
        sequence += [nn.Conv3d(ndf * nf_mult, 1,
                               kernel_size=kw, stride=1, padding=padw)]
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
        self.load_state_dict(torch.load(
            ckpt, map_location=torch.device("cpu"),), strict=False)
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
                 learn_logvar=False,
                 outlier_scaling_factor: int = 3,
                 outlier_loss_weight: float = 0):

        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight
        # output log variance
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init,
                                   requires_grad=learn_logvar)

        self.discriminator = NLayerDiscriminator3D(
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
        layer = last_layer if last_layer is not None else self.last_layer[0]

        nll_grads = torch.autograd.grad(nll_loss, layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, layer, retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(self, inputs, reconstructions, posteriors, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train",
                weights=None):
        B, T, C, H, W = inputs.shape
        if optimizer_idx == 0:
            inputs = inputs.flatten(0, 1)
            reconstructions = reconstructions.flatten(0, 1)

            # REVISIT: should I be masking?
            # mask = mask.flatten(0, 1)
            # valid_mask = mask != 0
            # inputs = inputs[valid_mask]
            # reconstructions = reconstructions[valid_mask]

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
            # Temporal adjustment: roll time into batch dimension
            pB, pT, pC, pH, pW = posteriors.mean.shape
            X = rearrange(posteriors.mean, "b t c h w -> (b t) c h w")
            outlier_loss = 0.
            for b in range(pB * pT):
                for i in range(pH):
                    for j in range(pW):
                        outlier_loss += torch.max(
                            (X[b, :, i, j] - X.mean()).norm() -
                            (self.outlier_scaling_factor * X.std()).norm(),
                            torch.tensor([0.], device=inputs.device))
            outlier_loss /= pH * pW

            inputs = rearrange(
                inputs, "(B T) C H W -> B C T H W", T=T).contiguous()
            reconstructions = rearrange(
                reconstructions, "(B T) C H W -> B C T H W", T=T
            ).contiguous()

            # generator update
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(
                    torch.cat((reconstructions.contiguous(), cond), dim=1))
            g_loss = -torch.mean(logits_fake)

            if global_step >= self.discriminator_iter_start:
                if self.disc_factor > 0.0:
                    d_weight = self.calculate_adaptive_weight(
                        nll_loss, g_loss, last_layer=last_layer
                    )
                else:
                    d_weight = torch.tensor(1.0)
            else:
                d_weight = torch.tensor(0.0)
                g_loss = torch.tensor(0.0, requires_grad=True)

            disc_factor = adopt_weight(
                self.disc_factor, global_step,
                threshold=self.discriminator_iter_start)
            d_weight = d_weight.to("cuda:0")
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
        elif optimizer_idx == 1:
            # second pass for discriminator update
            inputs = rearrange(
                inputs, "B T C H W -> B C T H W").contiguous()
            reconstructions = rearrange(
                reconstructions, "B T C H W -> B C T H W").contiguous()

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


class Conv2Plus1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv2d = nn.Conv3d(in_channels, out_channels,
                                kernel_size=(1, kernel_size, kernel_size),
                                stride=(1, stride, stride),
                                padding=(0, padding, padding))
        self.conv1d = nn.Conv3d(out_channels, out_channels,
                                kernel_size=(kernel_size, 1, 1),
                                stride=(stride, 1, 1),
                                padding=(padding, 0, 0),
                                padding_mode="replicate"
                                )

    def forward(self, x):
        x = self.conv2d(x)
        x = self.conv1d(x)
        return x


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = Conv2Plus1d(
                in_channels, in_channels,
                kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(
            x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = Conv2Plus1d(
                in_channels, in_channels,
                kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        if self.with_conv:
            # for the Conv2Plus1d you need to add the second pad
            pad = (0, 1, 0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            raise NotImplementedError
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class TemporalResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = Conv2Plus1d(
            in_channels, out_channels,
            kernel_size=3, stride=1, padding=1)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = Conv2Plus1d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = Conv2Plus1d(
                    in_channels, out_channels,
                    kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = Conv2Plus1d(
                    in_channels, out_channels,
                    kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)
        # temporal inflation

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)
        # temporal inflation

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = Conv2Plus1d(in_channels,
                             in_channels,
                             kernel_size=1,
                             stride=1,
                             padding=0)
        self.k = Conv2Plus1d(in_channels,
                             in_channels,
                             kernel_size=1,
                             stride=1,
                             padding=0)
        self.v = Conv2Plus1d(in_channels,
                             in_channels,
                             kernel_size=1,
                             stride=1,
                             padding=0)
        self.proj_out = Conv2Plus1d(in_channels,
                                    in_channels,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0)

    def forward(self, x):
        h_ = x
        # B, C, T, H, W = h_.shape
        # h_ = h_.permute(0, 2, 1, 3, 4).view(B * T, C, H, W)
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, t, h, w = q.shape
        q = q.permute(0, 2, 3, 4, 1).reshape(b * t, h * w, c).contiguous()
        k = k.permute(0, 2, 3, 4, 1).reshape(b * t, h * w, c).contiguous()
        v = v.permute(0, 2, 3, 4, 1).reshape(b * t, h * w, c).contiguous()
        q = q.view(b * t, -1, 1, c).transpose(1, 2)
        k = k.view(b * t, -1, 1, c).transpose(1, 2)
        v = v.view(b * t, -1, 1, c).transpose(1, 2)

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False
        )
        attn_output = attn_output.transpose(1, 2).reshape(b * t, -1, 1 * c)
        attn_output = attn_output.reshape(b, t, h, w, c).permute(0, 4, 1, 2, 3)

        h_ = self.proj_out(attn_output)

        return x + h_


class TemporalEncoder(nn.Module):
    def __init__(self,
                 *,
                 ch,
                 out_ch,
                 ch_mult=(1, 2, 4, 8),
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
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = Conv2Plus1d(
            in_channels, self.ch,
            kernel_size=3, stride=1, padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        # REVISIT:
        # idk what i'm doing with the temoral attn rn
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn_dict = OrderedDict()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(TemporalResnetBlock(in_channels=block_in,
                                                 out_channels=block_out,
                                                 dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn_dict.update(
                        {str(i_block): AttnBlock(block_in)})
                    # temporal inflation
            down = nn.Module()
            down.block = block
            down.attn = nn.ModuleDict(attn_dict)
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(
                    block_in, resamp_with_conv,)  # NOQA
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = TemporalResnetBlock(in_channels=block_in,
                                               out_channels=block_in,
                                               dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = TemporalResnetBlock(in_channels=block_in,
                                               out_channels=block_in,
                                               dropout=dropout)

        # end
        self.norm_out = Normalize(block_in)
        block_out = 2*z_channels if double_z else z_channels
        self.conv_out = Conv2Plus1d(
            block_in, block_out, kernel_size=3, stride=1, padding=1)
        # self.conv_out = torch.nn.Conv2d(block_in,
        #                                 block_out,
        #                                 kernel_size=3,
        #                                 stride=1,
        #                                 padding=1)
        # # temporal inflation
        # self.temp_conv_out = TemporalConv1d(
        #     block_out, block_out, kernel_size=3, stride=1)

    def forward(self, x):
        # timestep embedding

        # downsamplinj
        x = x.permute(0, 2, 1, 3, 4)
        hs = self.conv_in(x)
        hs = [hs]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[str(i_block)](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h)
        # B, T, C, H, W = h.shape
        # h = h.view(B * T, C, H, W)
        h = self.mid.attn_1(h)
        # temporal inflation
        # h = self.mid.temp_attn_1(h)
        # h = h.view(B, T, C, H, W)
        h = self.mid.block_2(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        h = h.permute(0, 2, 1, 3, 4).contiguous()
        return h


class TemporalDecoder(nn.Module):
    def __init__(self,
                 *,
                 ch,
                 out_ch,
                 ch_mult=(1, 2, 4, 8),
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
        self.conv_in = Conv2Plus1d(
            z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = TemporalResnetBlock(in_channels=block_in,
                                               out_channels=block_in,
                                               dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = TemporalResnetBlock(in_channels=block_in,
                                               out_channels=block_in,
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
                                                 dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn_dict.update(
                        {str(i_block): AttnBlock(block_in)})
            up = nn.Module()
            up.block = block
            up.attn = nn.ModuleDict(attn_dict)
            if i_level != 0:
                up.upsample = Upsample(
                    block_in, resamp_with_conv,)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = Conv2Plus1d(
            block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        z = z.permute(0, 2, 1, 3, 4)
        self.last_z_shape = z.shape

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        # h = self.mid.temp_attn_1(h)
        h = self.mid.block_2(h)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[str(i_block)](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)
        h = h.permute(0, 2, 1, 3, 4).contiguous()
        return h


class DiagonalGaussianDistribution:
    def __init__(self, parameters, deterministic=False):
        """
        parameters: shape [B, T, C, H, W]
        """
        self.parameters = parameters
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
    num_res_blocks: int = 2
    attn_resolutions: Tuple[int] = tuple()
    dropout: float = 0.0
    loss_disc_start: int = 20000
    loss_kl_weight: float = 0.000001
    loss_disc_weight: float = 0.5
    loss_outlier_weight: float = 1e2
    loss_outlier_scaling_factor: float = 3
    scaling_factor: float = 3.713188899219769

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
            num_res_blocks=config.num_res_blocks,
            embed_dim=config.embed_dim,
            dropout=config.dropout,
            in_channels=config.in_channels,
            resolution=config.resolution,
            z_channels=config.z_channels,
            double_z=config.double_z,
            attn_resolutions=config.attn_resolutions)

        self.quant_conv = Conv2Plus1d(
            2 * config.z_channels, 2 * config.embed_dim, kernel_size=1)
        self.post_quant_conv = Conv2Plus1d(
            config.embed_dim, config.z_channels, kernel_size=1)
        self.embed_dim = config.embed_dim
        self.scaling_factor = config.scaling_factor
        self.loss = LPIPSWithDiscriminator(
            disc_start=config.loss_disc_start,
            kl_weight=config.loss_kl_weight,
            disc_weight=config.loss_disc_weight,
            outlier_scaling_factor=config.loss_outlier_scaling_factor,
            outlier_loss_weight=config.loss_outlier_weight,
        )

    def to(self, device):
        self.encoder.to("cuda:0")
        self.quant_conv.to("cuda:0")
        self.decoder.to("cuda:1")
        self.post_quant_conv.to("cuda:1")
        self.loss.to("cuda:0")
        return self

    def from_pretrained(self, ckpt: Path, ignore_keys: List[str] | None = None,):
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

        self.load_state_dict(sd, strict=ignore_keys is None)

    def encode(self, x):
        # temporal accounting
        x = self.encoder(x)
        x = x.permute(0, 2, 1, 3, 4)
        moments = self.quant_conv(x)
        moments = moments.permute(0, 2, 1, 3, 4)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        # temporal accounting
        z = z.permute(0, 2, 1, 3, 4)
        z = self.post_quant_conv(z)
        z = z.permute(0, 2, 1, 3, 4)
        dec = self.decoder(z)
        # temporal accounting
        return dec

    @property
    def last_layer(self):
        return self.decoder.conv_out.conv1d.weight

    def forward(self, inputs,
                split: str = "val",
                optimizer_idx: int = 0,
                step: int = 0,
                sample_posterior: bool = True):
        # temporal accounting
        inputs = inputs.to("cuda:0")
        posterior = self.encode(inputs)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        z = z.to("cuda:1")
        dec = self.decode(z)
        dec = dec.to("cuda:0")

        # NOTE: we assume T of 8 is a single frame
        if dec.shape[1] == 8:
            dec = dec[:, 0:1]
            inputs = inputs[:, 0:1]
        loss, log_dict = self.loss(
            inputs, dec, posterior, optimizer_idx,  # 1 for val
            step, last_layer=self.last_layer, split=split)
        return dec, posterior, loss, log_dict

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
