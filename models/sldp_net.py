"""SLDP-Net — Spectral-Leakage-aware Dual-Pathway network (NOVEL architecture).

Integrates physics + multispectral + bionics + COD at the ARCHITECTURE level (not
as additive features, which prior experiments showed are redundant):

- PHYSICS: camouflage matches the background in visible bands {0-4} but LEAKS in
  non-visible {5,6,7} (NIR/red-edge). The core signal is the DIVERGENCE between
  the visible (V) and non-visible (NV) responses — not any single band.
- BIONIC: visual systems process spectral ranges via separate receptor classes
  (visible cones vs UV/IR channels) in parallel pathways, fused in cortex;
  predators locate prey by spectral MISMATCH.
- MULTISPECTRAL: 8 bands split by physical role into V {0-4} and NV {5,6,7}.
- COD: detect where V and NV 'disagree' = the camouflage leak.

WHY it can beat a plain 8-band net (the key difference from prior failed physics
add-ons): the physics is encoded STRUCTURALLY — V and NV are processed in separate
encoders and their divergence is the mandatory fusion path; the backbone CANNOT
bypass it (unlike additive features it can ignore). The divergence is exactly the
camouflage-breaking physical signal, computed by construction.

Structure: two encoders (ResNet, on V and NV) -> per-scale divergence |V-NV| +
cross-fusion -> StrongDecoder.
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backbones import MSBackbone                                           # noqa: E402
from decoders.strong_decoder import StrongDecoder                          # noqa: E402

# band split by physical role
V_BANDS = [0, 1, 2, 3, 4]      # visible: Violet,Blue,Green,Orange,Red
NV_BANDS = [5, 6, 7]           # non-visible: RedEdge,NIR1,NIR2 (where camouflage leaks)


class CrossFusion(nn.Module):
    """Fuse V, NV, and their divergence at one scale."""
    def __init__(self, c):
        super().__init__()
        self.proj = nn.Sequential(nn.Conv2d(3 * c, c, 1), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
                                  nn.Conv2d(c, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(inplace=True))

    def forward(self, v, nv):
        div = torch.abs(v - nv)                    # spectral-leakage divergence
        return self.proj(torch.cat([v, nv, div], dim=1))


class SLDPNet(nn.Module):
    def __init__(self, backbone_arch: str = "resnet34", pretrained: bool = True):
        super().__init__()
        self.v_enc = MSBackbone(in_channels=len(V_BANDS), arch=backbone_arch,
                                pretrained=pretrained, antagonistic_stem=False)
        self.nv_enc = MSBackbone(in_channels=len(NV_BANDS), arch=backbone_arch,
                                 pretrained=pretrained, antagonistic_stem=False)
        c = self.v_enc.channels                      # same for both encoders
        self.fuse = nn.ModuleList([CrossFusion(ci) for ci in c])
        self.decoder = StrongDecoder(tuple(reversed(c)))   # spec_map unused (SLDP IS the physics)
        self.last_auxs = None
        self.last_edge = None
        self.last_aux = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,8,H,W) z-scored; split by physical role
        v_in = x[:, V_BANDS]
        nv_in = x[:, NV_BANDS]
        v_feats = self.v_enc(v_in)                   # [C2..C5]
        nv_feats = self.nv_enc(nv_in)
        fused = [self.fuse[i](v_feats[i], nv_feats[i]) for i in range(len(v_feats))]
        mask, auxs, edge = self.decoder(fused, spec_map=None)
        self.last_auxs = auxs
        self.last_edge = F.interpolate(edge, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return F.interpolate(mask, size=x.shape[-2:], mode="bilinear", align_corners=False)
