"""Plain FPN+attention decoder (no boundary head). Returns mask logits only.
forward(feats, spec_map=None): if spec_map given, inject peripheral evidence at the
finest level (used by MSO-Net physics runs); else plain (used by the no-physics baseline)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PlainFPNDecoder(nn.Module):
    def __init__(self, chans=(2048, 1024, 512, 256), c_spec: int = 32, c_mid: int = 64):
        super().__init__()
        self.l5 = nn.Conv2d(chans[0], c_mid, 1)
        self.l4 = nn.Conv2d(chans[1], c_mid, 1)
        self.l3 = nn.Conv2d(chans[2], c_mid, 1)
        self.l2 = nn.Conv2d(chans[3], c_mid, 1)
        self.spec = nn.Conv2d(c_spec, c_mid, 1)
        self.refine = nn.Sequential(
            nn.Conv2d(c_mid, c_mid, 3, padding=1), nn.BatchNorm2d(c_mid), nn.ReLU(inplace=True),
            nn.Conv2d(c_mid, c_mid, 3, padding=1), nn.BatchNorm2d(c_mid), nn.ReLU(inplace=True),
        )
        self.att = nn.Conv2d(c_mid, 1, 1)
        self.head = nn.Conv2d(c_mid, 1, 1)

    def forward(self, feats, spec_map=None):
        c2, c3, c4, c5 = feats
        p5 = self.l5(c5)
        p4 = self.l4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="bilinear", align_corners=False)
        p3 = self.l3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="bilinear", align_corners=False)
        p2 = self.l2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="bilinear", align_corners=False)
        if spec_map is not None:
            s = F.interpolate(self.spec(spec_map), size=p2.shape[-2:], mode="bilinear", align_corners=False)
            p2 = p2 + s
        x = self.refine(p2)
        a = torch.sigmoid(self.att(x))
        return self.head(x * (1.0 + a))
