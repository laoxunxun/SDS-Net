"""SFCBridge — Spectral-Frequency Coupled Bridge (the novel bridge).

Mounted on a FROZEN spectral-blind backbone (law-sanctioned non-redundancy). This
is the paper's headline novelty: it differs from HyperCOD's SSG bridge (spectral
saliency only) by fusing multispectral x frequency, injected at MULTIPLE scales.

Two coupled mechanisms:
  1. MULTI-SCALE SPECTRAL LADDER: extract 8-band spectral features at all 4 decoder
     scales (96/48/24/12) and inject at each -> 4 injection points (the simple
     bridge injected at only 1).
  2. FREQUENCY COUPLING: at each scale, DWT-decompose the spectral feature into
     low/high-frequency bands and gate the high-frequency (detail) bands by a
     spectral-anomaly map (where the camouflage spectral leak is strong). This
     couples spectral (anomaly gate) x frequency (detail emphasis) -> the deep
     fusion, distinct from HyperCOD's spectral-only SSG.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from sgfd import dwt_haar, idwt_haar


class _FreqMod(nn.Module):
    """DWT the feature, gate the high-freq detail bands by an anomaly map, IDWT back."""

    def __init__(self, c: int, mid: int = 16):
        super().__init__()
        # gate the 3 detail bands (LH,HL,HH = 3c) conditioned on the anomaly map
        self.gate = nn.Sequential(
            nn.Conv2d(3 * c + 1, mid, 1, bias=False), nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
            nn.Conv2d(mid, 3 * c, 1), nn.Sigmoid())

    def forward(self, feat: torch.Tensor, anomaly: torch.Tensor) -> torch.Tensor:
        if feat.shape[-1] % 2 != 0:                               # ensure even for DWT
            return feat
        LL, LH, HL, HH = dwt_haar(feat)
        H = torch.cat([LH, HL, HH], dim=1)                        # (B,3c,h/2,w/2)
        a = F.adaptive_avg_pool2d(anomaly, H.shape[-2:])          # (B,1,h/2,w/2)
        g = self.gate(torch.cat([H, a], dim=1))
        H_mod = H * g
        c = feat.shape[1]
        return idwt_haar(LL, H_mod[:, :c], H_mod[:, c:2 * c], H_mod[:, 2 * c:3 * c], feat.shape[-2:])


class SFCBridge(nn.Module):
    """Spectral-Frequency Coupled Bridge -> features at 4 decoder scales to inject."""

    def __init__(self, n_bands: int = 8, chans=(96, 192, 384, 768), mid: int = 48):
        super().__init__()
        # spectral anomaly map (the camouflage leak signal) from 8 bands
        self.anomaly = nn.Sequential(
            nn.Conv2d(n_bands, mid, 1, bias=False), nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
            nn.Conv2d(mid, 1, 1), nn.Sigmoid())
        # multi-scale spectral ladder: 8 bands -> features at /4 /8 /16 /32
        self.stem = nn.Sequential(
            nn.Conv2d(n_bands, mid, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(mid), nn.ReLU(inplace=True))  # /4
        self.proj = nn.ModuleList([nn.Conv2d(mid, c, 1) for c in chans])
        self.down = nn.ModuleList([
            nn.Sequential(nn.Conv2d(mid, mid, 3, stride=2, padding=1, bias=False),
                          nn.BatchNorm2d(mid), nn.ReLU(inplace=True)) for _ in range(3)])
        self.freq = nn.ModuleList([_FreqMod(c) for c in chans])
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor):                           # x: (B,8,H,W)
        anomaly = self.anomaly(x)                                  # (B,1,H,W) spectral anomaly
        feats = []
        f = self.stem(x)                                           # (B,mid,H/4,W/4) scale0
        for i in range(4):
            fi = self.freq[i](self.proj[i](f), anomaly)            # project + frequency-modulate
            feats.append(fi)
            if i < 3:
                f = self.down[i](f)                                # next scale
        return feats, anomaly
