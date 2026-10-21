"""Spectral-divergence-guided Boundary Alignment (SBA).

THE NOVEL MECHANISM (post-DASA). Distinct from fusion (absorbed) and sharpening
(DASA gutted a pathway): this is spatial ALIGNMENT via deformable sampling.

Physics
-------
The camouflage boundary is physically localized by the V/NV spectral divergence
D = |mean(visible) - mean(non-visible)|, which peaks where the camouflage
material contrasts with the terrain. So D tells the model WHERE the true
boundary is. The decoder's predicted mask has an uncertain/fuzzy boundary.
We use D to drive a deformable conv's sampling offsets, snapping the boundary
refinement toward the divergence-determined location.

Operation
---------
    feat   = MLP(mask)                       # feature to refine (from mask only)
    offset = OffsetHead( SOURCE )            # 2D deformation field
    ref    = DeformConv2d(feat, offset)      # sample toward true boundary
    out    = mask + Conv(ref)

  SOURCE = D (spectral divergence, PHYSICS)  -> Ours
  SOURCE = mask (the prediction itself)      -> B0 control (same params, no physics)

Why it is load-bearing & not absorbed
--------------------------------------
The deformable sampling PATTERN is determined by the divergence. Removing the
divergence (B0) changes WHERE the conv samples — the backbone cannot reproduce
divergence-steered sampling by learning generic weights, because the sampling
locations are input-dependent via the divergence map. So the physics is in the
SPATIAL STRUCTURE of the operation, not in a feature the backbone can derive.

Why distinct from prior ops
---------------------------
- vs fusion/attention: those mix feature VALUES; SBA moves SAMPLING LOCATIONS.
- vs CSS sharpening: that gated V's own high-freq; SBA deforms sampling by D.
- vs FROB: FROB concatenated D as a feature channel; SBA uses D to STEER a
  deformable kernel (structural, not a concatenated prior).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.ops import DeformConv2d

V_BANDS = [0, 1, 2, 3, 4]
NV_BANDS = [5, 6, 7]


def spectral_divergence(x: torch.Tensor) -> torch.Tensor:
    """D = |mean(visible bands) - mean(non-visible bands)| at full resolution."""
    v = x[:, V_BANDS].mean(dim=1, keepdim=True)
    nv = x[:, NV_BANDS].mean(dim=1, keepdim=True)
    return torch.abs(v - nv)


class SpectralBoundaryAlignment(nn.Module):
    """Deformable boundary refinement with offsets driven by a SOURCE signal.

    Args:
        mid: feature width.
        k: deformable kernel size.
        offset_source: "div" = spectral divergence (PHYSICS, Ours);
                       "mask" = coarse mask (CONTROL, B0).
    """

    def __init__(self, mid: int = 32, k: int = 3, offset_source: str = "div"):
        super().__init__()
        self.offset_source = offset_source
        self.k = k
        # feature to be refined — derived from the coarse MASK only (same for both)
        self.feat = nn.Sequential(
            nn.Conv2d(1, mid, 3, padding=1, bias=False), nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, 3, padding=1, bias=False), nn.BatchNorm2d(mid), nn.ReLU(inplace=True))
        # offset predictor — identical arch; only the INPUT differs (div vs mask)
        self.offset_head = nn.Sequential(
            nn.Conv2d(1, mid, 3, padding=1, bias=False), nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
            nn.Conv2d(mid, 2 * k * k, 1))                    # no activation -> small init offsets (~0)
        self.dcn = DeformConv2d(mid, mid, kernel_size=k, padding=k // 2)
        self.out = nn.Conv2d(mid, 1, 1)
        self.last_offset = None   # cached for diagnostics

    def forward(self, mask: torch.Tensor, div: torch.Tensor) -> torch.Tensor:
        feat = self.feat(mask)
        src = div if self.offset_source == "div" else mask
        offset = self.offset_head(src)
        self.last_offset = offset
        return self.out(self.dcn(feat, offset))
