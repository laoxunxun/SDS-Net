"""Spectral Predictive-Coding module (SPC) — a SELF-DESIGNED, hand-crafted operator
(fusing bionics + physics + COD; not a generic conv, not a stitched paper).

Idea: a discriminative conv can learn any spectral FEATURE from data (that's why our
earlier front-end spectral priors were redundant). To get a signal a classifier does
NOT compute, use a GENERATIVE/predictive step: predict the expected BACKGROUND spectrum
at each pixel, then the RESIDUAL (prediction error) is the detection signal.

- Bionic: predictive coding (predict input, flag the 'surprise'/residual) +
  center-surround (predict center from surround) + search-image (adapt which cue matters).
- Physics: a robust local BACKGROUND model — the surround IS the expected background;
  camouflage LEAKS in non-visible bands => high residual there.
- COD: detection = locate the residual violation.

HAND-CRAFTED core (fixed, not learned, not a generic conv): a multi-scale CENTER-
EXCLUDED RING aggregator estimates the local background spectrum per band. The only
learned parameters are two physically-meaningful adaptations:
  - scale_w : adaptive surround scale (how far to look for background) — search-image.
  - band_w  : which bands' surprises matter (e.g. NIR/red-edge leak) — search-image.
residual = center - predicted-background => a generative 'surprise' signal.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _ring_kernel(radius: int, width: int = 1) -> torch.Tensor:
    """Normalized center-excluded ring-average kernel: 1/|ring| on a ring at `radius`,
    0 elsewhere (including the center) -> averages the SURROUND, not the center."""
    k = 2 * radius + 1
    c = radius
    yy, xx = np.meshgrid(np.arange(k), np.arange(k), indexing="ij")
    d = np.sqrt((yy - c) ** 2 + (xx - c) ** 2)
    m = ((d >= radius - width + 0.5) & (d <= radius + 0.5)).astype(np.float32)
    t = torch.tensor(m)
    return t / (t.sum() + 1e-8)                        # (k,k), sums to 1


class SpectralPredictiveCoder(nn.Module):
    def __init__(self, in_bands: int = 8, radii=(3, 7, 13)):
        super().__init__()
        self.in_bands = in_bands
        self.radii = tuple(radii)
        self.out_channels = in_bands
        for i, r in enumerate(self.radii):
            self.register_buffer(f"ring{i}", _ring_kernel(r))   # fixed hand-crafted rings
        self.scale_w = nn.Parameter(torch.zeros(len(self.radii)))  # adaptive surround scale
        self.band_w = nn.Parameter(torch.zeros(in_bands))          # search-image band weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:           # (B,C,H,W) -> residual (B,C,H,W)
        C = x.shape[1]
        sw = torch.softmax(self.scale_w, dim=0)
        bg = 0.0
        for i in range(len(self.radii)):
            k = getattr(self, f"ring{i}"); ks = k.shape[0]
            w = k.expand(C, 1, ks, ks)                            # depthwise, shared ring per band
            bg = bg + sw[i] * F.conv2d(x, w, padding=ks // 2, groups=C)
        residual = x - bg                                          # predictive-coding surprise
        bw = torch.softmax(self.band_w, dim=0) * C
        return residual * bw[None, :, None, None]
