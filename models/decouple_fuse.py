"""Spectral-anomaly / spatial-boundary decoupling + cross-attention fusion.

Two evidence streams:
  - spectral anomaly: from SOM + LPSI (material-contrast evidence, illumination-robust)
  - spatial boundary : Sobel edge of the pseudo-grey cube (boundary evidence)
Cross-attention fuses them so the net focuses where BOTH spectra and edges agree.
This is the tractable re-framing of the original proposal's "feature decoupling +
multi-domain enhancement": spectral + spatial only (frequency lives in pcsu-net).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SobelEdge(nn.Module):
    """Fixed Sobel edge extractor (single channel). Returns edge magnitude."""
    def __init__(self):
        super().__init__()
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        ky = kx.t()
        self.register_buffer("kx", kx.view(1, 1, 3, 3))
        self.register_buffer("ky", ky.view(1, 1, 3, 3))

    def forward(self, gray: torch.Tensor) -> torch.Tensor:  # gray: (B,1,H,W)
        gx = F.conv2d(gray, self.kx, padding=1)
        gy = F.conv2d(gray, self.ky, padding=1)
        return torch.sqrt(gx * gx + gy * gy + 1e-6)


class CrossAttentionFuse(nn.Module):
    """Cheap cross-attention: spectral-evidence attends to boundary-evidence.

    NOTE: full HWxHW attention is O((HW)^2). At 352^2 (~4e7 entries) it is
    borderline; TODO(research) replace with axial / pooled attention for speed.
    """
    def __init__(self, c_spectral: int, c_boundary: int, dim: int = 32):
        super().__init__()
        self.q = nn.Conv2d(c_spectral, dim, 1)
        self.k = nn.Conv2d(c_boundary, dim, 1)
        self.v = nn.Conv2d(c_boundary, dim, 1)
        self.out = nn.Conv2d(dim, c_spectral, 1)
        self.scale = dim ** -0.5

    def forward(self, s: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        B, _, H, W = s.shape
        q = self.q(s).flatten(2)                          # (B, dim, HW)
        k = self.k(b).flatten(2)
        v = self.v(b).flatten(2)
        attn = torch.softmax((q.transpose(1, 2) @ k) * self.scale, dim=-1)  # (B, HW, HW)
        z = (attn @ v.transpose(1, 2)).transpose(1, 2)    # (B, dim, HW)
        z = z.view(B, -1, H, W)
        return s + self.out(z)


class DecoupleFuse(nn.Module):
    """Decouple spectral-anomaly vs spatial-boundary evidence, fuse.

    Args:
        c_spectral:  channels of the spectral-evidence feature (SOM+LPSI projection)
        c_boundary:  channels of the spatial-boundary feature (from backbone)
        c_out:       output channels
    The Sobel edge (1 ch) is concatenated to the boundary feature and projected
    back to c_boundary, keeping all channel bookkeeping self-consistent.
    """
    def __init__(self, c_spectral: int, c_boundary: int, c_out: int):
        super().__init__()
        self.edge = SobelEdge()
        self.bnd_proj = nn.Conv2d(c_boundary + 1, c_boundary, 1)
        self.fuse = CrossAttentionFuse(c_spectral, c_boundary)
        self.proj = nn.Conv2d(c_spectral + c_boundary, c_out, 1)

    def forward(self, spectral_feat: torch.Tensor, gray: torch.Tensor,
                boundary_feat: torch.Tensor) -> torch.Tensor:
        size = spectral_feat.shape[-2:]
        bf = F.interpolate(boundary_feat, size=size, mode="bilinear", align_corners=False)
        edge = F.interpolate(self.edge(gray), size=size, mode="bilinear", align_corners=False)
        bnd = self.bnd_proj(torch.cat([bf, edge], dim=1))      # (B, c_boundary, H, W)
        fused = self.fuse(spectral_feat, bnd)                   # (B, c_spectral, H, W)
        return self.proj(torch.cat([fused, bnd], dim=1))        # (B, c_out, H, W)
