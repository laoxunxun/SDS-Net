"""Physics-guided spectral front-end + physics-derived saliency prior.

Concrete, computable spectral-PHYSICS quantities (NOT free parameters):
  - SpectralDerivative       : dR/dλ (red-edge feature between Red/RedEdge/NIR).
  - LPSI (lpsi.py)           : learnable normalised-difference indices (NDVI family).
  - physics_saliency_prior   : local spectral contrast (pixel vs local-mean spectrum)
                               = a physics-derived camouflage-saliency prior; used as
                               the soft target of the physics-consistency loss, so the
                               peripheral evidence is FORCED to align with a physical
                               quantity (hard physics guidance, not cosmetic injection).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralDerivative(nn.Module):
    """dR/dλ: finite differences between adjacent bands, with a learnable gain per
    inter-band gap (init 1 => starts as the plain physical derivative)."""

    def __init__(self, in_bands: int = 8):
        super().__init__()
        self.in_bands = in_bands
        self.gain = nn.Parameter(torch.ones(in_bands - 1))
        self.out_channels = in_bands - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:    # (B,in_bands,H,W)->(B,in_bands-1,H,W)
        d = x[:, 1:] - x[:, :-1]
        return d * self.gain.view(1, -1, 1, 1)


class SpectralAngle(nn.Module):
    """Optional: spectral-angle mapper (material similarity, illumination-invariant).
    Off by default (enable in ablation if it helps)."""

    def __init__(self, in_bands: int = 8, eps: float = 1e-6):
        super().__init__()
        self.ref = nn.Parameter(torch.rand(in_bands) * 0.1 + 0.1)
        self.eps = eps
        self.out_channels = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:    # (B,in_bands,H,W)->(B,1,H,W)
        ref = self.ref / (self.ref.norm() + self.eps)
        xn = x / (x.norm(dim=1, keepdim=True) + self.eps)
        cos = (xn * ref.view(1, -1, 1, 1)).sum(dim=1, keepdim=True).clamp(-1 + self.eps, 1 - self.eps)
        return torch.acos(cos)


def physics_saliency_prior(x: torch.Tensor, ksize: int = 15) -> torch.Tensor:
    """Physics-derived camouflage-saliency prior: per-band LOCAL SPECTRAL CONTRAST
    (each pixel's spectrum vs the mean spectrum of its local neighbourhood). High
    value => spectrally anomalous vs local background => candidate camouflage.

    Structure-preserving under per-band z-scoring: the local mean cancels the
    per-band shift, leaving (true local contrast)/σ per band. Returns (B,1,H,W),
    min-max normalised per image to [0,1].
    """
    pad = ksize // 2
    local_mean = F.avg_pool2d(x, kernel_size=ksize, stride=1, padding=pad)
    diff = (x - local_mean).abs().mean(dim=1, keepdim=True)        # (B,1,H,W)
    mn = diff.amin(dim=(2, 3), keepdim=True)
    mx = diff.amax(dim=(2, 3), keepdim=True)
    return (diff - mn) / (mx - mn + 1e-6)
