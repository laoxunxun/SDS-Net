"""Cross-Spectral Bridge — the NOVEL core module of SDS-Net.

Fuses visible (V) and non-visible (NV) feature streams via CROSS-SPECTRAL
modulation + explicit divergence. Unlike concat/add (redundant, backbone ignores),
this STRUCTURALLY forces V↔NV reasoning at every scale:
  1. V→NV channel attention: V's global context modulates NV (what NV channels matter given V)
  2. NV→V channel attention: NV modulates V (reciprocal)
  3. Explicit divergence |V−NV|: the spectral-leakage signal (where V and NV disagree)
Output = fuse(V_gated, NV_gated, divergence) — a feature that encodes the V/NV relationship.

Physics: the divergence = camouflage's spectral leak (visible matches, non-visible diverges).
Bionic: parallel spectral pathways + cortical cross-channel integration.
Lightweight (channel attention, not O(HW²) spatial) — applies at all 4 scales.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CrossSpectralBridge(nn.Module):
    def __init__(self, c: int, reduction: int = 4):
        super().__init__()
        # V → NV channel attention (V's context gates NV)
        self.v2nv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(c, c // reduction), nn.ReLU(inplace=True),
            nn.Linear(c // reduction, c), nn.Sigmoid())
        # NV → V channel attention (NV's context gates V)
        self.nv2v = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(c, c // reduction), nn.ReLU(inplace=True),
            nn.Linear(c // reduction, c), nn.Sigmoid())
        # divergence processing
        self.div = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(inplace=True))
        # fuse the three signals
        self.fuse = nn.Sequential(nn.Conv2d(3 * c, c, 1), nn.BatchNorm2d(c), nn.ReLU(inplace=True))
        # learnable residual balance (α·fused + β·v + γ·nv) — network learns optimal
        # trade-off between cross-spectral enhancement and alignment preservation
        self.alpha = nn.Parameter(torch.tensor(0.5))  # fusion weight
        self.beta = nn.Parameter(torch.tensor(0.5))   # V residual (alignment)
        self.gamma = nn.Parameter(torch.tensor(0.5))  # NV residual (alignment)

    def forward(self, v: torch.Tensor, nv: torch.Tensor) -> torch.Tensor:
        # v, nv: (B, C, H, W)
        nv_gated = nv * self.v2nv(v).view(v.shape[0], v.shape[1], 1, 1)  # NV modulated by V
        v_gated = v * self.nv2v(nv).view(nv.shape[0], nv.shape[1], 1, 1)  # V modulated by NV
        div = self.div(torch.abs(v - nv))                                 # spectral-leakage divergence
        fused = self.fuse(torch.cat([v_gated, nv_gated, div], dim=1))
        return self.alpha * fused + self.beta * v + self.gamma * nv  # learnable residual balance
