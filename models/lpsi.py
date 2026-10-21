"""Learnable Physical Spectral Index (LPSI).

Generalises fixed remote-sensing indices (NDVI = (NIR-R)/(NIR+R)) to a LEARNABLE,
band-selection form:

    P = softmax(theta_p),  Q = softmax(theta_q)        # differentiable band selection
    n = sum_i P_i * B_i   (numerator band)
    d = sum_j Q_j * B_j   (denominator band)
    idx = (n - d) / (|n| + |d| + eps)                   # bounded in ~[-1, 1]

Multiple heads => multiple learned indices. Each idx is a physically-grounded
contrast map (material-class discriminator) used as an injected prior.

TODO(research): seed theta_p/theta_q from the best hand-crafted indices reported
in "Optimized spectral indices for camouflage detection in multispectral imagery"
(in the user's IMA library) once band→wavelength mapping is confirmed.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LPSI(nn.Module):
    def __init__(self, in_bands: int = 8, num_indices: int = 4, eps: float = 1e-3,
                 init_pairs: list | None = None):
        super().__init__()
        self.eps = eps
        tp = torch.randn(num_indices, in_bands) * 0.3
        tq = torch.randn(num_indices, in_bands) * 0.3
        # Optionally seed indices as NDVI-like normalised differences.
        # init_pairs: list of (numerator_band_idx, denominator_band_idx); softmax
        # of a peaked logit => each index starts near a clean band difference.
        if init_pairs:
            for i, (nb, db) in enumerate(init_pairs[:num_indices]):
                tp[i] = -1.5; tp[i, nb] = 1.5   # soft NDVI bias (NOT saturated -> can learn)
                tq[i] = -1.5; tq[i, db] = 1.5
        self.theta_p = nn.Parameter(tp)
        self.theta_q = nn.Parameter(tq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_bands, H, W)  (use raw normalised cube, not z-scored if you want
        # the index to stay physically meaningful; caller decides).
        P = torch.softmax(self.theta_p, dim=1)           # (k, in)
        Q = torch.softmax(self.theta_q, dim=1)           # (k, in)
        n = torch.einsum("ki,bihw->bkhw", P, x)
        d = torch.einsum("ki,bihw->bkhw", Q, x)
        idx = (n - d) / (n.abs() + d.abs() + self.eps)   # bounded ~[-1,1]
        return idx

    def selected_bands(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Argmax band per index (for interpretability / reporting)."""
        return self.theta_p.argmax(dim=1), self.theta_q.argmax(dim=1)
