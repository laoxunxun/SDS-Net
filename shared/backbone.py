"""Multi-scale encoder for 8-band multispectral input, with an optional
spectral-antagonistic center-surround STEM (Double-Opponent Conv, DOC).

Default stem: torchvision ResNet conv1 patched to in_channels=8 (RGB weights
copied to the first 3 bands). Returns stride-4/8/16/32 feature maps [C2..C5].

Optional `antagonistic_stem=True` REPLACES conv1 with a DOC: it computes
multi-scale center-surround spectral contrasts (x - Gaussian blur) per band, then
an antagonistically-initialised conv (+ on NIR/red-edge bands where camouflage
leaks, - on visible bands where camouflage matches). This injects spectral-spatial
differencing STRUCTURALLY at the network's first operation (bio: double-opponent
receptive field; physics: local spectral contrast). For SOTA, swap arch to
Res2Net-50 (same 4-scale output); DOC sits at the stem regardless.

NOTE: with antagonistic_stem the stem conv has NO pretrained weights (layer1-4
keep ImageNet weights) — a deliberate trade of pretrained-stem features for
structural physics guidance.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_resnet(arch: str, pretrained: bool):
    from torchvision import models
    try:
        m = getattr(models, arch)(weights="IMAGENET1K_V1" if pretrained else None)
    except TypeError:  # older torchvision API
        m = getattr(models, arch)(pretrained=pretrained)
    return m


def _gauss1d(sigma: float) -> torch.Tensor:
    k = max(int(2 * round(2 * sigma) + 1), 3)
    ax = torch.arange(k, dtype=torch.float32) - (k - 1) / 2
    g = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
    return g / g.sum()                                   # (k,), sums to 1


# MCOD band groups (0-based): camouflage LEAKS in NIR/red-edge; MATCHES in visible
_ANTAG_BANDS = (5, 6, 7)      # RedEdge725, NIR1_785, NIR2_887  -> positive weights
_CAMO_BANDS = (0, 1, 2, 3, 4)  # Violet,Blue,Green,Orange,Red   -> negative weights


class DoubleOpponentConv(nn.Module):
    """Spectral-antagonistic center-surround stem conv (replaces backbone conv1).

    out = Conv_{antag-init}( concat_sigma (x - blur_sigma(x)) )
    Each input channel is a center-surround spectral contrast (local spectral
    contrast = a physics quantity); the conv then mixes them with an
    antagonistically-initialised sign pattern (+NIR/red-edge, -visible)."""

    def __init__(self, in_bands: int, out_ch: int, kernel_size: int = 7, stride: int = 2,
                 padding: int = 3, sigmas=(1.5, 3.0),
                 antag_bands=_ANTAG_BANDS, camo_bands=_CAMO_BANDS):
        super().__init__()
        self.in_bands = in_bands
        self.sigmas = tuple(sigmas)
        for i, s in enumerate(self.sigmas):
            self.register_buffer(f"_g{i}", _gauss1d(s))   # separable 1D surround
        self.conv = nn.Conv2d(in_bands * len(self.sigmas), out_ch, kernel_size,
                              stride=stride, padding=padding, bias=False)
        self._init_antagonistic(antag_bands, camo_bands)

    def _init_antagonistic(self, antag, camo):
        nn.init.kaiming_normal_(self.conv.weight)
        with torch.no_grad():
            w = self.conv.weight                      # (out, in_bands*S, k, k)
            for si in range(len(self.sigmas)):
                off = si * self.in_bands
                for a in antag:
                    w[:, a + off, :, :] = w[:, a + off, :, :].abs()    # force +
                for c in camo:
                    w[:, c + off, :, :] = -w[:, c + off, :, :].abs()   # force -

    def _blur(self, x: torch.Tensor, g1d: torch.Tensor) -> torch.Tensor:
        """Separable Gaussian blur on x:(B,C,H,W)."""
        B, C, H, W = x.shape
        k = g1d.shape[0]
        xc = x.reshape(B * C, 1, H, W)
        y = F.conv2d(xc, g1d.view(1, 1, k, 1), padding=(k // 2, 0))   # horizontal
        y = F.conv2d(y, g1d.view(1, 1, 1, k), padding=(0, k // 2))    # vertical
        return y.view(B, C, H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:    # (B,in_bands,H,W)->(B,out_ch,H/2,W/2)
        contrasts = []
        for i in range(len(self.sigmas)):
            g = getattr(self, f"_g{i}")
            contrasts.append(x - self._blur(x, g))         # center-surround contrast per band
        return self.conv(torch.cat(contrasts, dim=1))


class MSBackbone(nn.Module):
    def __init__(self, in_channels: int = 8, arch: str = "resnet50", pretrained: bool = False,
                 antagonistic_stem: bool = False):
        super().__init__()
        m = _build_resnet(arch, pretrained)
        old = m.conv1
        ks = old.kernel_size[0] if isinstance(old.kernel_size, (list, tuple)) else old.kernel_size
        st = old.stride[0] if isinstance(old.stride, (list, tuple)) else old.stride
        pd = old.padding[0] if isinstance(old.padding, (list, tuple)) else old.padding

        if antagonistic_stem:
            stem_conv = DoubleOpponentConv(in_channels, old.out_channels,
                                           kernel_size=ks, stride=st, padding=pd)
        else:
            stem_conv = nn.Conv2d(in_channels, old.out_channels, kernel_size=ks,
                                  stride=st, padding=pd, bias=old.bias is not None)
            if pretrained and old.weight is not None:
                with torch.no_grad():
                    stem_conv.weight[:, :3] = old.weight              # seed RGB bands
                    nn.init.kaiming_normal_(stem_conv.weight[:, 3:])  # extra bands

        self.stem = nn.Sequential(stem_conv, m.bn1, m.relu)
        self.maxpool = m.maxpool
        self.layer1, self.layer2 = m.layer1, m.layer2
        self.layer3, self.layer4 = m.layer3, m.layer4
        self.channels = [64, 128, 256, 512] if arch in ("resnet18", "resnet34") \
            else [256, 512, 1024, 2048]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(x)
        x = self.maxpool(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return [c2, c3, c4, c5]


# TODO(paper): add build_res2net() for SOTA backbone; keep the same 4-scale output.
