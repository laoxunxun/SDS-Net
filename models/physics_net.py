"""PhysicsNet — physics-as-load-bearing-structure dual-pathway (feasibility test).

PURPOSE
-------
Tests the central concern: with a STRONG dual-Swin backbone, does the V/NV
physics design give a measurable performance Δ over a capacity-matched control
that LACKS the physics? If Δ>0 the physics is load-bearing (not absorbed); if
Δ≈0 it is absorbed MSO-style and we redesign.

Two boolean-ish flags isolate the physics, and ONLY the physics:

  band_split  : "vn"     = visible{0-4} / non-visible{5-7}        (PHYSICS split)
                "random" = a fixed meaningless 5/3 band mix        (CONTROL split)

  gate_source : "div"    = fusion gate driven by |V-NV| spectral divergence (PHYSICS)
                "v"      = fusion gate driven by group-1 features alone     (CONTROL)

  Ours = (vn, div);  B0 = (random, v).

The two share IDENTICAL architecture, IDENTICAL parameter count, IDENTICAL
training. The ONLY difference is whether the physically-meaningful structure
(V/NV grouping + divergence-driven gating) organizes the computation. So any
Δ is attributable to the physics design, not to capacity or backbone strength.

This directly answers "is the architecture just strong, with nothing to do with
your design?" — B0 IS the "just strong" baseline; Ours must beat it.

DESIGN PRINCIPLE
----------------
The physics is LOAD-BEARING: removing it does not revert to a complete generic
backbone, it changes what the network computes. The decoder cannot bypass the
fusion (it consumes fused features), and the fusion's coupling is steered by
the (physics or control) gate. If the physics gate steers better than the
control gate, Δ>0.
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decoders.strong_decoder import StrongDecoder

# Two 5/3 band groupings. "vn" is the physics split; "random" is a fixed
# meaningless mix (interleaves visible & non-visible so neither group is physical).
SPLITS = {
    "vn":     ([0, 1, 2, 3, 4], [5, 6, 7]),               # visible / non-visible
    "random": ([0, 2, 4, 5, 7], [1, 3, 6]),               # meaningless mix (5/3)
}


class PhysicsAnchoredFusion(nn.Module):
    """Per-scale gated residual fusion of two pathway features.

    coupled = MLP([g1, g2])                       # cross-group content coupling
    gate    = sigmoid(MLP( SOURCE ))              # where to apply coupling
    out     = g1 + alpha * gate * (coupled - g1)   # gated residual

    SOURCE = |g1 - g2| (divergence, PHYSICS)  OR  g1 (control). Same MLP shape,
    same params -> capacity matched; only the signal that drives the gate differs.
    """

    def __init__(self, c: int, gate_source: str = "div", reduction: int = 4):
        super().__init__()
        self.gate_source = gate_source
        self.cross = nn.Sequential(
            nn.Conv2d(2 * c, c, 1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True))
        hidden = max(c // reduction, 8)
        self.gate = nn.Sequential(
            nn.Conv2d(c, hidden, 1, bias=False), nn.BatchNorm2d(hidden), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 1), nn.Sigmoid())
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.last_gate = None   # cached for diagnostics

    def forward(self, g1: torch.Tensor, g2: torch.Tensor) -> torch.Tensor:
        coupled = self.cross(torch.cat([g1, g2], dim=1))
        src = torch.abs(g1 - g2) if self.gate_source == "div" else g1
        gate = self.gate(src)
        self.last_gate = gate
        return g1 + self.alpha * gate * (coupled - g1)


class PhysicsNet(nn.Module):
    def __init__(self, band_split: str = "vn", gate_source: str = "div",
                 v_arch: str = "swin_small_patch4_window7_224",
                 nv_arch: str = "swin_tiny_patch4_window7_224",
                 pretrained: bool = True, img_size: int = 384,
                 alignment: str | None = None, grad_ckpt: bool = False):
        super().__init__()
        from shared.swin_backbone import SwinBackbone
        if band_split not in SPLITS:
            raise ValueError(f"band_split must be 'vn' or 'random', got {band_split}")
        self.g1_bands, self.g2_bands = SPLITS[band_split]
        self.band_split = band_split
        self.gate_source = gate_source

        self.enc1 = SwinBackbone(in_bands=len(self.g1_bands), arch=v_arch,
                                 pretrained=pretrained, img_size=img_size)
        self.enc2 = SwinBackbone(in_bands=len(self.g2_bands), arch=nv_arch,
                                 pretrained=pretrained, img_size=img_size)
        if grad_ckpt:   # enable on both Swin encoders so bs=8 + alignment fits in 8GB
            for enc in (self.enc1, self.enc2):
                if hasattr(enc.encoder, "set_grad_checkpointing"):
                    enc.encoder.set_grad_checkpointing(enable=True)
        c = self.enc1.channels                      # [96,192,384,768]
        self.fusions = nn.ModuleList([
            PhysicsAnchoredFusion(ci, gate_source=gate_source) for ci in c])
        self.decoder = StrongDecoder(tuple(reversed(c)))

        # Optional spectral-divergence-guided boundary alignment (the novel op)
        self.alignment = None
        if alignment is not None:
            from boundary_alignment import SpectralBoundaryAlignment
            self.alignment = SpectralBoundaryAlignment(offset_source=alignment)

        self.last_auxs = None
        self.last_edge = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[-2:]
        f1 = self.enc1(x[:, self.g1_bands])
        f2 = self.enc2(x[:, self.g2_bands])
        fused = [self.fusions[i](f1[i], f2[i]) for i in range(len(f1))]
        mask, auxs, edge = self.decoder(fused)
        self.last_auxs = auxs
        self.last_edge = F.interpolate(edge, size=(H, W), mode="bilinear", align_corners=False)
        mask = F.interpolate(mask, size=(H, W), mode="bilinear", align_corners=False)
        if self.alignment is not None:
            from boundary_alignment import spectral_divergence
            div = spectral_divergence(x)
            mask = mask + self.alignment(mask, div)
        return mask


if __name__ == "__main__":
    # capacity-match sanity: Ours and B0 must have ~identical param counts
    for name, bs, gs in [("Ours (vn,div)", "vn", "div"), ("B0 (random,v)", "random", "v")]:
        m = PhysicsNet(band_split=bs, gate_source=gs, pretrained=False, img_size=384)
        n = sum(p.numel() for p in m.parameters())
        x = torch.randn(2, 8, 384, 384)
        y = m(x)
        print(f"{name:18s}: params={n/1e6:6.2f}M  out={tuple(y.shape)}")
