"""Boundary-Guided Decoder (BGNet-inspired, not copied).

Addresses the diagnosed 'blurry boundary' weakness of the placeholder FPN. Three
principled, effective pieces:
  1. Top-down FPN fusion of {C2..C5} + injection of the peripheral evidence.
  2. Explicit EDGE HEAD predicting boundaries (supervised by Sobel(GT) in the loss)
     — directly targets boundary precision (the BGNet finding: edge-supervision
     decoders materially sharpen COD masks).
  3. EDGE-GUIDED REFINEMENT: the predicted edge gates the mask features
     (base * (1 + sigmoid(edge))), enhancing responses at boundaries, followed by
     a spatial (search) attention gate.

Returns (mask_logits, edge_logits). The model upsamples both to input resolution,
returns mask for the engine, and caches edge for the explicit edge-supervision loss.
Integration with our hub (not a vanilla BGNet): the peripheral spectral-opponency
evidence (CS-SO/LPSI/derivative) is injected at the finest level, so the decoder
refines physics-grounded evidence rather than generic backbone features.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundaryGuidedDecoder(nn.Module):
    def __init__(self, chans=(512, 256, 128, 64), c_spec: int = 32, c_mid: int = 64):
        super().__init__()
        self.l5 = nn.Conv2d(chans[0], c_mid, 1)   # C5
        self.l4 = nn.Conv2d(chans[1], c_mid, 1)
        self.l3 = nn.Conv2d(chans[2], c_mid, 1)
        self.l2 = nn.Conv2d(chans[3], c_mid, 1)   # C2
        self.spec = nn.Conv2d(c_spec, c_mid, 1)
        self.edge_head = nn.Sequential(
            nn.Conv2d(c_mid, c_mid, 3, padding=1), nn.BatchNorm2d(c_mid), nn.ReLU(inplace=True),
            nn.Conv2d(c_mid, 1, 1),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(c_mid, c_mid, 3, padding=1), nn.BatchNorm2d(c_mid), nn.ReLU(inplace=True),
            nn.Conv2d(c_mid, c_mid, 3, padding=1), nn.BatchNorm2d(c_mid), nn.ReLU(inplace=True),
        )
        self.att = nn.Conv2d(c_mid, 1, 1)         # spatial (search) attention
        self.mask_head = nn.Conv2d(c_mid, 1, 1)

    def forward(self, feats, spec_map):
        c2, c3, c4, c5 = feats
        p5 = self.l5(c5)
        p4 = self.l4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="bilinear", align_corners=False)
        p3 = self.l3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="bilinear", align_corners=False)
        p2 = self.l2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="bilinear", align_corners=False)
        s = F.interpolate(self.spec(spec_map), size=p2.shape[-2:], mode="bilinear", align_corners=False)
        base = p2 + s                                 # inject peripheral evidence
        edge = self.edge_head(base)                   # boundary logits
        g = self.refine(base * (1.0 + torch.sigmoid(edge)))   # edge-guided refinement
        a = torch.sigmoid(self.att(g))                # search attention
        mask = self.mask_head(g * (1.0 + a))
        return mask, edge
