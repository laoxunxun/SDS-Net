"""EDM — Endmember Decomposition Module (deep multispectral physics).

Refined version: data-driven endmember init + the target endmember is index 0.
The linear spectral-mixture physics (pixel ~= sum_k a_k * e_k) is enforced by a
hard reconstruction loss; abundance-target alignment ties endmember 0 to the
camouflage target so the physical representation becomes a detection signal.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EndmemberDecomposition(nn.Module):
    def __init__(self, n_bands: int = 8, K: int = 6, work_size: int = 96, mid: int = 32,
                 target_spec: torch.Tensor | None = None, bg_spec: torch.Tensor | None = None):
        super().__init__()
        self.K, self.n_bands, self.work_size = K, n_bands, work_size
        self.endmembers = nn.Parameter(self._init_endmembers(K, n_bands, target_spec, bg_spec))
        self.abund_est = nn.Sequential(
            nn.Conv2d(n_bands, mid, 1), nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
            nn.Conv2d(mid, K, 1))
        self.last_x = None
        self.last_xhat = None
        self.last_a = None

    @staticmethod
    def _init_endmembers(K, n_bands, target_spec, bg_spec):
        """endmember 0 = target material; 1..K-1 = background variants (spread)."""
        E = torch.zeros(K, n_bands)
        if target_spec is not None:
            E[0] = target_spec.detach().to(torch.float32)
        else:
            E[0] = 0.4
        # background endmembers: spread around the bg spectrum (or a default)
        base = bg_spec.detach().to(torch.float32) if bg_spec is not None else torch.full((n_bands,), 0.4)
        for k in range(1, K):
            sig = 0.2 * torch.cos(torch.linspace(0, k * 3.0, n_bands))
            E[k] = (base + sig).clamp(min=0.05)
        return E

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xs = F.adaptive_avg_pool2d(x, self.work_size)               # (B,8,ws,ws)
        a_logits = self.abund_est(xs)                                # (B,K,ws,ws)
        a = F.softmax(a_logits, dim=1)                               # abundances, sum_k=1, >=0
        x_hat = torch.einsum("bkhw,ke->behw", a, self.endmembers.to(a.dtype))  # (B,8,ws,ws)
        self.last_x, self.last_xhat, self.last_a = xs, x_hat, a
        return a                                                     # (B,K,ws,ws)

    def recon_loss(self) -> torch.Tensor:
        if self.last_x is None:
            return torch.tensor(0.0, device=self.endmembers.device)
        return F.mse_loss(self.last_xhat, self.last_x)

    def endmember_diversity(self) -> torch.Tensor:
        e = self.endmembers
        pd = torch.cdist(e, e)
        eye = torch.eye(self.K, device=e.device).bool()
        return -pd[~eye].mean()
