"""SDS-Net — Spectral-Disentangled Swin Transformer for Multispectral COD.

Architecture innovations:
1. Asymmetric dual-pathway Swin (V=visible / NV=non-visible) + Cross-Spectral Bridge
2. Full-Resolution Double-Opponent Boundary Guidance (FROB) — computes V/NV spectral
   divergence at FULL INPUT RESOLUTION (pixel-level), bypassing the Swin patchification
   that loses fine boundary detail. Inspired by double-opponent cells in V1 cortex
   (Yang 2015, Gao 2015), generalized to multispectral camouflage detection. The FROB
   map provides pixel-level spectral boundary information that the encoder CANNOT recover,
   guiding the decoder's boundary refinement head directly.

Physics: V/NV divergence at pixel level = camouflage spectral-leak boundary.
Bionic: double-opponent cells = spectral edge detectors in V1 (cite Yang/Gao).
Multispectral: requires V/NV spectral group separation (RGB cannot compute this).
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cross_spectral_bridge import CrossSpectralBridge
from decoders.strong_decoder import StrongDecoder
from physics_v2 import PhysicsNorm, SpectralPrior, to_reflectance

V_BANDS = [0, 1, 2, 3, 4]      # visible: Violet,Blue,Green,Orange,Red
NV_BANDS = [5, 6, 7]           # non-visible: RedEdge,NIR1,NIR2


class FullResDoubleOpponent(nn.Module):
    """Full-Resolution Double-Opponent Boundary map.

    Computes V/NV spectral divergence at FULL INPUT RESOLUTION (pixel-level),
    bypassing Swin's 4x4 patchification. This recovers fine boundary detail that
    the encoder loses. Inspired by double-opponent cells in V1 cortex that detect
    spectral edges (Yang 2015, Gao 2015).

    Pipeline (all at input resolution, e.g. 384x384):
    1. V_mean = mean of visible bands (channel-wise average of V{0-4})
    2. NV_mean = mean of non-visible bands (channel-wise average of NV{5-7})
    3. DO_map = |V_mean - NV_mean| (absolute spectral divergence = spectral edge)
    4. Multi-scale smoothing: 3x3 + 5x5 avg pools for boundary band extraction
    5. Learnable enhancement: 3-layer conv to refine the boundary signal
    Output: 1-channel boundary prior at full resolution.

    Physics: camouflage matches V but leaks NV -> DO_map is HIGH at target boundaries.
    This is the pixel-level spectral boundary that Swin's patchification CANNOT see.
    """

    def __init__(self, in_bands: int = 8, mid: int = 16):
        super().__init__()
        # Learnable refinement on top of the physics-derived DO map
        # Input: the raw DO map (1ch) + multi-scale context (2ch) = 3ch
        self.refine = nn.Sequential(
            nn.Conv2d(3, mid, 3, padding=1), nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, 3, padding=1), nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
            nn.Conv2d(mid, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 8, H, W) z-scored multispectral cube
        Returns: (B, 1, H, W) full-resolution double-opponent boundary prior.
        """
        # Pixel-level V/NV spectral divergence (double-opponent response)
        v_mean = x[:, V_BANDS].mean(dim=1, keepdim=True)    # (B,1,H,W)
        nv_mean = x[:, NV_BANDS].mean(dim=1, keepdim=True)   # (B,1,H,W)
        do_raw = torch.abs(v_mean - nv_mean)                  # (B,1,H,W) — spectral edge

        # Multi-scale boundary band: difference between fine and coarse DO response
        # This extracts the BOUNDARY BAND (where DO transitions from low to high)
        do_fine = F.avg_pool2d(do_raw, 3, stride=1, padding=1)
        do_coarse = F.avg_pool2d(do_raw, 5, stride=1, padding=2)
        do_band = do_fine - do_coarse                          # (B,1,H,W) — boundary band

        # Learnable refinement: raw DO + boundary band + coarse context -> refined prior
        out = self.refine(torch.cat([do_raw, do_band, do_coarse], dim=1))
        return torch.sigmoid(out)                              # (B,1,H,W) in [0,1]


class SDSNet(nn.Module):
    def __init__(self, v_arch: str = "swin_small_patch4_window7_224",
                 nv_arch: str = "swin_tiny_patch4_window7_224", pretrained: bool = True,
                 img_size: int = 384, use_bridge: bool = True, use_frob: bool = True,
                 physics_norm: str = "none", use_prior: bool = False,
                 grad_ckpt: bool = False,
                 v_pretrained=None, nv_pretrained=None):
        super().__init__()
        self.use_bridge = use_bridge    # ablation: cross-spectral bridge on/off
        self.use_frob = use_frob        # ablation: full-res double-opponent prior on/off
        self.use_norm = physics_norm in ("weber", "stevens", "both", "residual")
        self.use_prior = use_prior      # ablation: spectral prior FiLM injection on/off
        self.grad_ckpt = grad_ckpt      # recompute activations instead of spilling into
                                        # shared GPU memory (8GB cards: spill = 10x slowdown)
        from shared.swin_backbone import SwinBackbone
        # v/nv_pretrained: per-stream override (None -> follow `pretrained`); used by the
        # N-group from-scratch control (does ImageNet RGB init of the 3-band NV stem matter?)
        if v_pretrained is None: v_pretrained = pretrained
        if nv_pretrained is None: nv_pretrained = pretrained
        self.v_enc = SwinBackbone(in_bands=len(V_BANDS), arch=v_arch, pretrained=v_pretrained,
                                  img_size=img_size, grad_ckpt=grad_ckpt)
        self.nv_enc = SwinBackbone(in_bands=len(NV_BANDS), arch=nv_arch, pretrained=nv_pretrained,
                                   img_size=img_size, grad_ckpt=grad_ckpt)
        c = self.v_enc.channels
        self.bridges = nn.ModuleList([CrossSpectralBridge(ci) for ci in c])
        self.decoder = StrongDecoder(tuple(reversed(c)))

        # Full-Resolution Double-Opponent boundary prior (pixel-level, bypasses Swin).
        # Built only when use_frob, so the FROB-off ablation genuinely removes its params.
        if self.use_frob:
            self.frod = FullResDoubleOpponent(in_bands=8, mid=16)

        # Physics v2 (only built when enabled, so off-ablations genuinely remove parameters)
        if self.use_norm:
            self.pnorm = PhysicsNorm(mode=physics_norm)
        if self.use_prior:
            self.sprior = SpectralPrior(tuple(c))   # fused features are ordered shallow->deep
        self._physics_norm_mode = physics_norm

        # Boundary-Guided Refinement Head: fuses coarse mask + (FROB prior if used) + raw input.
        # Input channels = 1 (coarse mask) + (1 if FROB) + 8 (raw cube). When FROB is off the
        # prior channel is dropped so the head cannot relearn a spectral-edge prior from raw x.
        refine_in = 1 + (1 if self.use_frob else 0) + 8
        self.boundary_refine = nn.Sequential(
            nn.Conv2d(refine_in, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1))

        self.last_auxs = None
        self.last_edge = None
        self.last_aux = None
        self.last_frod = None    # cached for visualization
        self.last_cube01 = None  # cached [0,1] cube for physics losses (xmc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Full-resolution double-opponent boundary prior (pixel-level, physics-derived)
        frod_map = self.frod(x) if self.use_frob else None    # (B,1,H,W) spectral boundary prior
        if frod_map is not None:
            self.last_frod = frod_map
        self.last_cube01 = to_reflectance(x)                  # physics-domain copy for xmc loss

        # Mechanism 1 (physics_norm): encoders consume Weber/Stevens physics coordinates
        # instead of z-scored pixels — a hard inductive bias the backbone cannot learn back.
        if self.use_norm:
            v_in, nv_in = self.pnorm(x)
        else:
            v_in, nv_in = x[:, V_BANDS], x[:, NV_BANDS]

        # Spectral-disentangled dual-pathway encoding (feature-level, stride-4)
        v_feats = self.v_enc(v_in)
        nv_feats = self.nv_enc(nv_in)
        if self.use_bridge:
            fused = [self.bridges[i](v_feats[i], nv_feats[i]) for i in range(len(v_feats))]
        else:
            # Bridge-off null: V and NV share channel dims [96,192,384,768], so element-wise
            # add is parameter-free, preserves the decoder's channel contract, and removes ALL
            # cross-spectral interaction — the fairest "no bridge" baseline (not a strawman).
            fused = [v_feats[i] + nv_feats[i] for i in range(len(v_feats))]

        # Mechanism 2 (spectral prior): FiLM-inject physics indices at every decoder scale
        if self.use_prior:
            fused = self.sprior(fused, x)

        # Decoder (multi-scale fusion + deep supervision + boundary branch)
        mask, auxs, edge = self.decoder(fused)
        self.last_auxs = auxs
        self.last_edge = F.interpolate(edge, size=x.shape[-2:], mode="bilinear", align_corners=False)

        # Upsample coarse mask to full resolution
        mask = F.interpolate(mask, size=x.shape[-2:], mode="bilinear", align_corners=False)

        # Boundary refinement: coarse_mask + (FROB prior if used) + raw_input. The FROB map
        # tells the head WHERE the pixel-level spectral edges are (info Swin's patchification
        # LOST); without FROB the head only sees the coarse mask + raw cube.
        refine_in = torch.cat([mask, frod_map, x], dim=1) if self.use_frob else torch.cat([mask, x], dim=1)
        mask = mask + self.boundary_refine(refine_in)
        return mask
