"""Strong COD decoder — fixes the 'toy 0.32M decoder' bottleneck.
Multi-scale top-down fusion + residual refinement at every level + DEEP SUPERVISION
(aux heads at C5/C4/C3) + a boundary branch fused into the main head. ~2.4M params
(vs 0.32M before). This is the standard recipe that lets COD decoders turn strong
backbone features into accurate masks (the previous tiny FPN could not).

Returns (main_logits, [aux5, aux4, aux3], edge_logits) — aux list drives deep
supervision, edge drives boundary loss. Peripheral evidence (spec_map) injected at P2.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RB(nn.Module):  # residual refine block (capacity)
    def __init__(self, c):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, padding=1, bias=False); self.bn1 = nn.BatchNorm2d(c)
        self.c2 = nn.Conv2d(c, c, 3, padding=1, bias=False); self.bn2 = nn.BatchNorm2d(c)
    def forward(self, x):
        return F.relu(x + self.bn2(self.c2(F.relu(self.bn1(self.c1(x))))), inplace=True)


class StrongDecoder(nn.Module):
    def __init__(self, chans=(2048, 1024, 512, 256), c_spec: int = 32, c: int = 128):
        super().__init__()
        self.lat5 = nn.Conv2d(chans[0], c, 1); self.lat4 = nn.Conv2d(chans[1], c, 1)
        self.lat3 = nn.Conv2d(chans[2], c, 1); self.lat2 = nn.Conv2d(chans[3], c, 1)
        self.spec = nn.Conv2d(c_spec, c, 1)
        self.ref5, self.ref4, self.ref3, self.ref2 = RB(c), RB(c), RB(c), RB(c)
        self.aux5 = nn.Conv2d(c, 1, 1); self.aux4 = nn.Conv2d(c, 1, 1); self.aux3 = nn.Conv2d(c, 1, 1)
        self.bnd_feat = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(inplace=True))
        self.bnd_head = nn.Conv2d(c, 1, 1)
        self.main = nn.Sequential(nn.Conv2d(2 * c, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(inplace=True), RB(c))
        self.head = nn.Conv2d(c, 1, 1)

    def forward(self, feats, spec_map=None):
        c2, c3, c4, c5 = feats
        d5 = self.ref5(self.lat5(c5))
        d4 = self.ref4(self.lat4(c4) + F.interpolate(d5, size=c4.shape[-2:], mode="bilinear", align_corners=False))
        d3 = self.ref3(self.lat3(c3) + F.interpolate(d4, size=c3.shape[-2:], mode="bilinear", align_corners=False))
        p2 = self.lat2(c2) + F.interpolate(d3, size=c2.shape[-2:], mode="bilinear", align_corners=False)
        if spec_map is not None:
            p2 = p2 + self.spec(F.interpolate(spec_map, size=p2.shape[-2:], mode="bilinear", align_corners=False))
        d2 = self.ref2(p2)
        bfeat = self.bnd_feat(d2)
        edge = self.bnd_head(bfeat)
        m = self.main(torch.cat([d2, bfeat], dim=1))
        return self.head(m), [self.aux5(d5), self.aux4(d4), self.aux3(d3)], edge
