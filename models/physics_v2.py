"""Physics-guided components v2 — designed after PHYSICS_GUIDED_ANALYSIS.md showed that
loss-level soft constraints on input-derived quantities are learned by the Swin backbone
and contribute nothing. The v2 principle: physics must CHANGE THE INFORMATION THE NETWORK
SEES (normalization / prior injection / loss weighting structure), not merely add a
regularizer the backbone can replicate.

Three independent, individually-ablatable mechanisms:

1. PhysicsNorm   — hard inductive bias BEFORE the encoders. Maps z-scored pixels to
                   physically meaningful local quantities: Weber contrast (band response
                   relative to local surround, the quantity early vision actually encodes)
                   and Stevens-gamma compression (perceptual response ~ stimulus^gamma with
                   a learnable per-band gamma). This is a normalization the backbone cannot
                   "learn back": every downstream feature is computed in physics space.

2. SpectralPrior — explicit physics prior maps (V/NV divergence, spectral angle, band-ratio
                   NDVI-like indices) injected into EVERY decoder scale via FiLM gating.
                   Unlike FROB (which only reaches the refinement head), the prior shapes
                   the whole decoding path, so removing it provably changes the function.

3. XMC loss      — cross-modal contrast asymmetry. Camouflage in MCOD is physically
                   "low contrast in visible, radiometrically distinct in NIR". The loss
                   re-weights BCE per-pixel by this asymmetry, up-weighting exactly the
                   pixels where the two physics regimes disagree — a weighting pattern the
                   backbone cannot self-produce because it depends on the paired geometry.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

V_BANDS = [0, 1, 2, 3, 4]
NV_BANDS = [5, 6, 7]

# The dataloader z-scores the cube with these train-set stats; physics quantities need the
# [0,1] reflectance domain, so we invert here (single source of truth = shared.dataset).
from shared.dataset import BAND_MEAN, BAND_STD
_MEAN = torch.tensor(BAND_MEAN).view(1, 8, 1, 1)
_STD = torch.tensor(BAND_STD).view(1, 8, 1, 1)


def to_reflectance(x_z: torch.Tensor) -> torch.Tensor:
    """z-scored cube (B,8,H,W) -> [0,1] reflectance (same device as input)."""
    return (x_z * _STD.to(x_z.device) + _MEAN.to(x_z.device)).clamp(0.0, 1.0)


class PhysicsNorm(nn.Module):
    """Input-level physics normalization (mechanism 1).

    Two operating regimes:

    mode in ("weber","stevens","both") — HARD REPLACEMENT (exp36, FAILED):
      output replaces the z-scored cube. Measured failure causes (2026-08-30, see
      paper/PHYSICS_GUIDED_ANALYSIS.md): Weber kurtosis 20.6 and Stevens std 0.07 enter
      a pretrained stem calibrated for ~unit variance -> mid-training re-calibration
      plateau (ep59-104 stuck at 0.834); Weber keeps only local contrast (corr 0.069
      with intensity), deleting absolute radiometry. Retained for reproducing exp36.

    mode == "residual" — RESIDUAL GATED INJECTION (exp39, the fix):
      delta = g * W [standardized Weber; standardized Stevens],  g zero-init scalar.
      v_in = x_zscored + delta  -> step 0 is EXACTLY the baseline (pretrained calibration
      preserved); the network learns how much physics to admix. Physics channels are
      standardized (weber/0.2, stevens affine to ~unit variance) for good conditioning.
    """

    # Standardization constants measured on 24 cached train cubes (2026-08-30).
    WEBER_STD = 0.20
    STEVENS_MV = (0.593, 0.072)   # V pathway mean/std of I^gamma at gamma=1
    STEVENS_MN = (0.586, 0.037)   # NV pathway

    def __init__(self, v_arch_out: int = 5, nv_arch_out: int = 3, mode: str = "residual", k: int = 31):
        super().__init__()
        assert mode in ("weber", "stevens", "both", "residual")
        self.mode = mode
        self.k = k
        # gamma init 1.0 -> identity at start, learnable through sigmoid reparam
        self.gamma_v = nn.Parameter(torch.zeros(len(V_BANDS)))
        self.gamma_nv = nn.Parameter(torch.zeros(len(NV_BANDS)))
        if mode == "residual":
            cv, cnv = 2 * len(V_BANDS), 2 * len(NV_BANDS)
            self.compress_v = nn.Conv2d(cv, v_arch_out, 1)
            self.compress_nv = nn.Conv2d(cnv, nv_arch_out, 1)
            # zero-init gate: step 0 == baseline exactly (protects pretrained transfer)
            self.gate_v = nn.Parameter(torch.zeros(1))
            self.gate_nv = nn.Parameter(torch.zeros(1))
            return
        cv = 2 * len(V_BANDS) if mode == "both" else len(V_BANDS)
        cnv = 2 * len(NV_BANDS) if mode == "both" else len(NV_BANDS)
        self.compress_v = nn.Conv2d(cv, v_arch_out, 1)
        self.compress_nv = nn.Conv2d(cnv, nv_arch_out, 1)

    def _gamma(self, g: torch.Tensor) -> torch.Tensor:
        return 1.0 + torch.sigmoid(g) - 0.5          # range [0.5, 1.5], init = 1.0

    def _physics(self, img: torch.Tensor, gam: torch.Parameter, which: str) -> torch.Tensor:
        """Physics channels for one pathway, standardized. which: 'v' | 'nv'."""
        sm, ss = self.STEVENS_MV if which == "v" else self.STEVENS_MN
        feats = []
        if self.mode in ("weber", "both", "residual"):
            mu = F.avg_pool2d(img, self.k, stride=1, padding=self.k // 2) + 1e-4
            feats.append(((img - mu) / mu).clamp(-4.0, 4.0) / self.WEBER_STD)
        if self.mode in ("stevens", "both", "residual"):
            st = img.clamp(1e-4, 1.0) ** self._gamma(gam).view(1, -1, 1, 1)
            feats.append(((st - sm) / ss).clamp(-4.0, 4.0))
        return torch.cat(feats, dim=1)

    def delta(self, x_z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Residual physics deltas (B,5)/(B,3) for the residual mode."""
        x01 = to_reflectance(x_z)
        # tanh-bounded gate: contribution stays within [-1,1]x conv output, so physics
        # can never re-break the pretrained calibration however large the raw gate grows.
        gv, gn = torch.tanh(self.gate_v), torch.tanh(self.gate_nv)
        dv = gv * self.compress_v(self._physics(x01[:, V_BANDS], self.gamma_v, "v"))
        dn = gn * self.compress_nv(self._physics(x01[:, NV_BANDS], self.gamma_nv, "nv"))
        return dv, dn

    def forward(self, x_z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x_z: (B,8,H,W) z-scored cube (the dataloader's output). Returns (v_in, nv_in)."""
        if self.mode == "residual":
            dv, dn = self.delta(x_z)
            return x_z[:, V_BANDS] + dv, x_z[:, NV_BANDS] + dn
        x01 = to_reflectance(x_z)
        v, nv = x01[:, V_BANDS], x01[:, NV_BANDS]
        outs = []
        for img, gam, compress, which in ((v, self.gamma_v, self.compress_v, "v"),
                                          (nv, self.gamma_nv, self.compress_nv, "nv")):
            y = self._physics(img, gam, which).clamp(-4.0, 4.0)
            outs.append(compress(y))
        return outs[0], outs[1]


class SpectralPrior(nn.Module):
    """Explicit physics prior maps injected at every decoder scale (mechanism 2).

    Physics indices computed from the [0,1] cube:
      - div:   |mean(V) - mean(NV)|                       spectral divergence
      - ratio: normalized difference (V-NV)/(V+NV)        NDVI-like band-ratio index
      - sam:   spectral angle between each pixel's 5-band V vector and 3-band NV vector
               (flattened angle between the two spectral response profiles)
      - grad:  Sobel magnitude of div                     spectral edge strength
    Each index is refined by a small conv and FiLM-modulated (scale+shift) onto each
    decoder feature level. Zero-init of the FiLM shift/scale means the module starts as
    identity and LEARNS how much physics to inject — no random noise at init.
    """

    N_IDX = 4  # div, ratio, sam, grad

    def __init__(self, decoder_channels: tuple[int, ...]):
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(self.N_IDX, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, self.N_IDX, 1))
        self.film = nn.ModuleList([nn.Conv2d(self.N_IDX, 2 * c, 1) for c in decoder_channels])
        for m in self.film:
            nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)

    def indices(self, x01: torch.Tensor) -> torch.Tensor:
        v, nv = x01[:, V_BANDS], x01[:, NV_BANDS]
        vm, nm = v.mean(1, keepdim=True), nv.mean(1, keepdim=True)
        div = (vm - nm).abs()
        ratio = (vm - nm) / (vm + nm + 1e-4)
        # spectral angle between the V and NV spectral profiles of each pixel
        vN = F.normalize(v, dim=1); nN = F.normalize(nv, dim=1) if nv.shape[1] == v.shape[1] \
            else F.normalize(F.pad(nv, (0, 0, 0, 0, 0, v.shape[1] - nv.shape[1])), dim=1)
        # align lengths by padding NV to 5 bands with its mean (angle uses direction only)
        nv_pad = nv.mean(1, keepdim=True).expand(-1, v.shape[1], -1, -1)
        cos = (vN * F.normalize(nv_pad, dim=1)).sum(1, keepdim=True).clamp(-1 + 1e-6, 1 - 1e-6)
        sam = torch.acos(cos) / torch.pi
        kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                          device=x01.device).view(1, 1, 3, 3)
        grad = torch.sqrt(F.conv2d(div, kx, padding=1) ** 2 +
                          F.conv2d(div, kx.transpose(2, 3), padding=1) ** 2 + 1e-6)
        return torch.cat([div, ratio, sam, grad], dim=1)

    def forward(self, feats: list[torch.Tensor], x_z: torch.Tensor) -> list[torch.Tensor]:
        pri = self.refine(self.indices(to_reflectance(x_z)))
        out = []
        for i, f in enumerate(feats):
            p = F.interpolate(pri, size=f.shape[-2:], mode="bilinear", align_corners=False)
            scale, shift = torch.chunk(self.film[i](p), 2, dim=1)
            out.append(f * (1.0 + scale) + shift)      # FiLM; zero-init -> identity at start
        return out


def xmc_weight(cube: torch.Tensor, mask: torch.Tensor, k: int = 31,
               tau: float = 2.0, w_max: float = 5.0) -> torch.Tensor:
    """Cross-modal contrast asymmetry pixel weights (mechanism 3).

    Camouflage physics: |local contrast| low in V but high in NV. The weight
      w = 1 + tau * NV_contrast * (1 - V_contrast)
    amplifies the loss exactly on the pixels whose two physics regimes disagree —
    including FALSE-negative-prone background (V-flat NV-flat gets ~1) — and is
    computed from the paired multispectral cube, which no RGB-only reweighting can mimic.

    cube: (B,8,H,W) in [0,1]; mask unused (kept for API symmetry). Returns (B,1,H,W).
    """
    vm = cube[:, V_BANDS].mean(1, keepdim=True)
    nm = cube[:, NV_BANDS].mean(1, keepdim=True)

    def local_c(t: torch.Tensor) -> torch.Tensor:
        mu = F.avg_pool2d(t, k, stride=1, padding=k // 2)
        return (t - mu).abs().clamp(0, 1)

    c_v, c_nv = local_c(vm), local_c(nm)
    w = 1.0 + tau * c_nv * (1.0 - c_v)
    return w.clamp(1.0, w_max)
