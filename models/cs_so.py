"""Center-Surround Spectral Opponency (CS-SO): a DOUBLE-OPPONENT operator.

Novel architectural operator of MSO-Net. Couples spectral opponency with a spatial
receptive field, non-trivially. See mso-net/README.md.

DO_k(p) = [N_k(p) - blur(N_k)(p)] - [D_k(p) - blur(D_k)(p)]
        = local contrast(numerator band-set) - local contrast(denominator band-set)

Bio basis: double-opponent neurons (V1/avian). Physics basis: local spectral
contrast (the saliency prior); surround = Gaussian (DoG) receptive field.

PERF: the surround Gaussian blur uses SEPARABLE 1D convs (horizontal then
vertical) instead of one large 2D kernel — mathematically identical 2D Gaussian,
but ~10x faster with far better GPU utilisation (large-kernel depthwise 2D convs
are pathologically slow on GPU).

Params (all physically grounded):
  - A, B : K x 8 softmax band selections (numerator/denominator), NDVI-family init.
  - 1D Gaussian surrounds : FIXED, multiple sigmas (multi-scale receptive fields).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gauss1d(sigma: float) -> torch.Tensor:
    k = max(int(2 * round(2 * sigma) + 1), 3)            # odd, ~4*sigma diameter
    ax = torch.arange(k, dtype=torch.float32) - (k - 1) / 2
    g = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
    return g / g.sum()                                    # (k,), sums to 1


class CSSO(nn.Module):
    def __init__(self, in_bands: int = 8, K: int = 6,
                 sigmas=(1.5, 3.0, 5.0), init_pairs: list | None = None):
        super().__init__()
        self.in_bands = in_bands
        self.K = K
        self.sigmas = tuple(sigmas)
        self.S = len(self.sigmas)
        self.out_channels = K * self.S

        A = torch.randn(K, in_bands) * 0.3
        B = torch.randn(K, in_bands) * 0.3
        if init_pairs:
            for i, (nb, db) in enumerate(init_pairs[:K]):
                A[i] = -1.5; A[i, nb] = 1.5   # soft NDVI bias (not saturated -> can learn)
                B[i] = -1.5; B[i, db] = 1.5
        self.A = nn.Parameter(A)
        self.B = nn.Parameter(B)

        for i, s in enumerate(self.sigmas):               # fixed separable 1D surrounds
            self.register_buffer(f"_g{i}", _gauss1d(s))

    def _blur(self, x: torch.Tensor, g1d: torch.Tensor) -> torch.Tensor:
        """Separable Gaussian blur on x:(B,C,H,W)."""
        B, C, H, W = x.shape
        k = g1d.shape[0]
        xc = x.reshape(B * C, 1, H, W)
        y = F.conv2d(xc, g1d.view(1, 1, k, 1), padding=(k // 2, 0))   # horizontal
        y = F.conv2d(y, g1d.view(1, 1, 1, k), padding=(0, k // 2))    # vertical
        return y.view(B, C, H, W)

    def _surround(self, x: torch.Tensor) -> torch.Tensor:    # (B,C,H,W) -> (B,C,S,H,W)
        outs = [self._blur(x, getattr(self, f"_g{i}")) for i in range(self.S)]
        return torch.stack(outs, dim=2)

    def band_selection(self):
        return torch.softmax(self.A, dim=1), torch.softmax(self.B, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:      # (B,in_bands,H,W) -> (B,K*S,H,W)
        A, B = self.band_selection()
        N = torch.einsum("ki,bihw->bkhw", A, x)              # numerator spectrum (B,K,H,W)
        D = torch.einsum("ki,bihw->bkhw", B, x)              # denominator spectrum
        Ns = self._surround(N)                               # (B,K,S,H,W)
        Ds = self._surround(D)
        CN = N[..., None, :, :] - Ns
        CD = D[..., None, :, :] - Ds
        DO = CN - CD                                         # double-opponent response
        return DO.reshape(x.shape[0], self.out_channels, x.shape[2], x.shape[3])

    def reg_loss(self) -> torch.Tensor:
        A, _ = self.band_selection()
        mean_t = A.mean(dim=0)
        ent = -(mean_t * mean_t.clamp(min=1e-8).log()).sum()
        return -ent
