"""SGFD — Spectral-Guided Frequency Decomposition.

A DIFFERENT-KIND, NON-REDUNDANT physics guide that fuses two paradigms:
  - frequency decomposition (FEDER-style: camouflage exposes in the frequency domain)
  - multispectral guidance (the camouflage spectral leak)

THE NON-REDUNDANCY MECHANISM (the lesson from 10 absorbed attempts + HyperCOD):
  The backbone processes a COMPRESSED 3-channel input (8 bands -> 3), so it is
  SPECTRAL-DETAIL-BLIND. The multispectral guidance, computed from the full 8 bands,
  re-injects the spectral detail the backbone lost -> non-redundant (modality bridge
  via compression, distinct from HyperCOD's RGB-only-SAM bridge).

THE DEEP FUSION (not stacking):
  The spectral guidance does not just add a feature. It MODULATES the frequency
  decomposition: where the spectral leak (divergence) is strong (target boundary),
  the high-frequency (detail) bands are emphasized. Spectral info controls HOW the
  frequency decomposition contributes -> spectral x frequency coupling.

Pipeline at a feature scale:
  F -> DWT -> {LL, LH, HL, HH};  detail bands modulated by spectral guidance g;
  IDWT -> refined F.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

V_BANDS = [0, 1, 2, 3, 4]
NV_BANDS = [5, 6, 7]


def dwt_haar(x: torch.Tensor):
    """Haar DWT: (B,C,H,W) -> 4 x (B,C,H/2,W/2) [LL, LH, HL, HH]. H,W must be even."""
    a = x[..., 0::2, 0::2]; b = x[..., 0::2, 1::2]
    c = x[..., 1::2, 0::2]; d = x[..., 1::2, 1::2]
    LL = (a + b + c + d) * 0.5
    LH = (a + b - c - d) * 0.5    # horizontal detail
    HL = (a - b + c - d) * 0.5    # vertical detail
    HH = (a - b - c + d) * 0.5    # diagonal detail
    return LL, LH, HL, HH


def idwt_haar(LL, LH, HL, HH, out_hw):
    """Haar inverse DWT -> (B,C,H,W) at out_hw=(H,W)."""
    a = (LL + LH + HL + HH) * 0.5
    b = (LL + LH - HL - HH) * 0.5
    c = (LL - LH + HL - HH) * 0.5
    d = (LL - LH - HL + HH) * 0.5
    B, C, h, w = a.shape
    out = torch.zeros(B, C, out_hw[0], out_hw[1], device=a.device, dtype=a.dtype)
    out[..., 0::2, 0::2] = a; out[..., 0::2, 1::2] = b
    out[..., 1::2, 0::2] = c; out[..., 1::2, 1::2] = d
    return out


class SGFD(nn.Module):
    """Spectral-Guided Frequency Decomposition block (plug-and-play on a feature).

    Args:
        c_feat: feature channels.
        n_bands: input spectral bands (8).
        guide_source: "spectral" (physics, Ours) | "feature" (control, non-spectral).
    """

    def __init__(self, c_feat: int, n_bands: int = 8, mid: int = 16, guide_source: str = "spectral"):
        super().__init__()
        self.guide_source = guide_source
        # spectral guidance: from the full 8 bands -> 1ch spectral-leak map
        self.spec_guide = nn.Sequential(
            nn.Conv2d(n_bands, mid, 1, bias=False), nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
            nn.Conv2d(mid, 1, 1), nn.Sigmoid())
        # control guidance: from the feature itself (same param budget, no spectral)
        self.feat_guide = nn.Sequential(
            nn.Conv2d(c_feat, mid, 1, bias=False), nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
            nn.Conv2d(mid, 1, 1), nn.Sigmoid())
        # low-freq branch
        self.ll_proc = nn.Sequential(nn.Conv2d(c_feat, c_feat, 3, padding=1), nn.BatchNorm2d(c_feat), nn.ReLU(inplace=True))
        # detail (high-freq) branches share a processor
        self.h_proc = nn.Sequential(nn.Conv2d(3 * c_feat, 3 * c_feat, 3, padding=1), nn.BatchNorm2d(3 * c_feat), nn.ReLU(inplace=True))
        # spectral-guided detail gate: [detail, guidance] -> per-channel gate on detail
        self.gate = nn.Sequential(
            nn.Conv2d(3 * c_feat + 1, 3 * c_feat, 1, bias=False), nn.BatchNorm2d(3 * c_feat), nn.ReLU(inplace=True),
            nn.Conv2d(3 * c_feat, 3 * c_feat, 1), nn.Sigmoid())
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.last_guide = None

    def forward(self, feat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # feat: (B,c,h,w); x: (B,8,H,W) full spectral input
        H, W = feat.shape[-2:]
        LL, LH, HL, HH = dwt_haar(feat)                       # each (B,c,h/2,w/2)
        ll = self.ll_proc(LL)
        h_cat = self.h_proc(torch.cat([LH, HL, HH], dim=1))   # (B,3c,h/2,w/2)
        # guidance map at the detail resolution
        if self.guide_source == "spectral":
            g = self.spec_guide(x)                            # (B,1,H,W)
        else:
            g = self.feat_guide(feat)                         # control: from feature
        g = F.adaptive_avg_pool2d(g, h_cat.shape[-2:])        # (B,1,h/2,w/2)
        self.last_guide = g
        gate = self.gate(torch.cat([h_cat, g], dim=1))        # spectral-guided detail gate
        h_mod = h_cat * gate                                  # deep fusion: guidance modulates detail
        LL2, LH2, HL2, HH2 = ll, h_mod[:, :LL.shape[1]], h_mod[:, LL.shape[1]:2*LL.shape[1]], h_mod[:, 2*LL.shape[1]:]
        out = idwt_haar(LL2, LH2, HL2, HH2, (H, W))
        return feat + self.alpha * out
