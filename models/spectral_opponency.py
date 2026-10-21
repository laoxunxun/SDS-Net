"""Stomatopod (mantis-shrimp)-inspired Spectral Binning & Opponency Module (SBOM).

Grounded in behavioural evidence that stomatopods combine SPECTRAL BINNING and
SPECTRAL OPPONENCY across their ~12 photoreceptor classes (Wang 2025, J Exp Biol,
jeb247952). Two stages:

1. BINNING  : K learnable receptor tuning curves over the 8 bands (non-negative,
              sum-to-1) pool the spectrum into K 'receptors' — analogous to the
              mantis shrimp's many narrow, evenly-spaced photoreceptors. A
              DIVERSITY regularizer keeps receptors spectrally spread.
2. OPPONENCY: pairwise differences r_i - r_j between receptors give opponent
              channels that amplify material spectral contrast (illumination-
              robust). A learnable OPPONENT GATE selects which opponencies are
              salient for camouflage (efficiency + interpretability).

Output: P = K*(K-1)/2 opponent channels. The gate weights and receptor tuning
curves are directly inspectable -> interpretability.

Novelty: stomatopod inspiration in the literature is almost entirely hardware
sensors (SIMPOL, organic photodetectors) and neuroscience; spectral opponency in
DL for multispectral segmentation is an open gap. This module operationalises
both for camouflaged object detection.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SBOM(nn.Module):
    def __init__(self, in_bands: int = 8, K: int = 6, gate_tau: float = 1.0):
        super().__init__()
        self.in_bands = in_bands
        self.K = K
        # receptor tuning logits -> softmax => non-negative, sum-to-1 per receptor
        self.tuning_logits = nn.Parameter(torch.randn(K, in_bands) * 0.5)
        self.gate_tau = gate_tau
        # opponent pairs (i<j)
        pairs = [(i, j) for i in range(K) for j in range(i + 1, K)]
        self.register_buffer("pi", torch.tensor([a for a, _ in pairs], dtype=torch.long))
        self.register_buffer("pj", torch.tensor([b for _, b in pairs], dtype=torch.long))
        self.P = len(pairs)
        # learnable opponent saliency gate (which opponencies matter for camouflage)
        self.gate_logits = nn.Parameter(torch.zeros(self.P))

    @property
    def tuning(self) -> torch.Tensor:           # (K, in_bands) receptor sensitivities
        return torch.softmax(self.tuning_logits, dim=1)

    @property
    def gate(self) -> torch.Tensor:             # (P,) opponent saliency, sums to 1
        return torch.softmax(self.gate_logits / self.gate_tau, dim=0)

    def forward(self, x: torch.Tensor):
        """
        x: (B, in_bands, H, W)
        returns: opp (B, P, H, W) gated opponent channels, receptors (B, K, H, W)
        """
        r = torch.einsum("ki,bihw->bkhw", self.tuning, x)        # receptors (B,K,H,W)
        opp = r[:, self.pi] - r[:, self.pj]                       # (B,P,H,W)
        g = (self.gate * self.P).view(1, -1, 1, 1)                # rescaled gate (mean~1)
        return opp * g, r

    def reg_loss(self) -> torch.Tensor:
        """Receptor diversity (spread across spectrum) + sparse opponent gate."""
        T = self.tuning                                            # (K, in_bands)
        gram = T @ T.t()
        off = gram - torch.diag(torch.diagonal(gram))
        overlap = (off ** 2).mean()                               # penalise tuning overlap
        mean_t = T.mean(dim=0)
        spread = -(mean_t * mean_t.clamp(min=1e-8).log()).sum()    # entropy of mean tuning (maximize)
        gate_ent = -(self.gate * self.gate.clamp(min=1e-8).log()).sum()   # encourage sparse gate
        return overlap - 0.5 * spread + 0.01 * gate_ent


def opponency_reg_loss(sbom: "SBOM") -> torch.Tensor:
    return sbom.reg_loss()


# backward-compatible alias
SpectralOpponency = SBOM
