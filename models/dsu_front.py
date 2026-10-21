"""DSU front-end (differentiable spectral unmixing) — the structurally-NON-redundant
physics option. Unlike CS-SO/LPSI/derivative (linear band combos a conv already
computes), unmixing is a constrained latent decomposition:
    pixel(p) ≈ Σ_k A_k(p) · E_k,  with A_k>=0, Σ_k A_k = 1 (softmax), E_k>=0 (softplus).
Hypothesis: abundance maps (esp. the camouflage-material endmember) are a feature a
plain backbone does NOT compute, so they may add value where linear indices did not.
Returns abundance maps A (B,k,H,W); recon (B,in_bands,H,W) drives the consistency loss.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DSUFront(nn.Module):
    def __init__(self, in_bands: int = 8, k: int = 5):
        super().__init__()
        self.in_bands = in_bands
        self.k = k
        self.out_channels = k
        self.E_raw = nn.Parameter(torch.randn(k, in_bands) * 0.1)         # endmembers (softplus>0)
        self.abundance_head = nn.Sequential(
            nn.Conv2d(in_bands, 32, 1), nn.ReLU(inplace=True), nn.Conv2d(32, k, 1))

    @property
    def endmembers(self) -> torch.Tensor:
        return F.softplus(self.E_raw)                                     # (k, in_bands), >=0

    def forward(self, x: torch.Tensor):
        A = torch.softmax(self.abundance_head(x), dim=1)                  # (B,k,H,W) non-neg, sum-to-1
        recon = torch.einsum("ec,behw->bchw", self.endmembers, A)         # (B,in_bands,H,W)
        return A, recon
