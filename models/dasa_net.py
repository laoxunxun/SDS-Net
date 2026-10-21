"""DASA-Net — Depth-Asymmetric Spectral Architecture for Multispectral COD.

Design principle (physics-driven role allocation):
  The camouflage spectral-leak asymmetry, measured on MCOD, dictates an
  ARCHITECTURAL asymmetry that symmetric two-stream methods cannot express.

  * Visible (V) boundary is low-contrast (mean |Delta| ~= 0.002 at 660nm):
    there is NO sharp boundary to resolve, so spatial resolution is wasted on
    V. But V's appearance is COMPLEX (texture/colour/shape), so V needs DEPTH.
    -> V pathway = deep Swin-Small (stride-4 start, 50M params).

  * Non-visible (NV) boundary is high-contrast (mean |Delta| ~= 0.441 at 785nm):
    there IS a sharp boundary, and it is SIMPLE (high vs low reflectance), so
    NV needs RESOLUTION (to keep that sharp edge pixel-level) but NOT depth.
    -> NV pathway = shallow full-resolution CNN (stride-1 start, ~1M params).

  V needs depth; NV needs resolution. No RGB method can express this, because
  RGB has no V/NV split — all channels are visible and equally affected by
  camouflage. The asymmetry exists ONLY because of multispectral physics.

The headline mechanism: Cross-Spectral Sharpening (CSS) at every decoder scale.
NV (sharp, reliable boundary) sharpens V (deep, fuzzy boundary) via a
cross-modal lateral-inhibition gate. See cross_spectral_sharpening.py.

Bionic correspondence (deep, not surface):
  V pathway  <-> ventral "What" stream  (deep, large RF, low spatial freq, identity)
  NV pathway <-> dorsal  "Where" stream (shallow, small RF, high spatial freq, boundary)
  The depth-vs-resolution split follows the SAME principle the primate visual
  system uses to resolve the resolution/complexity trade-off, and the spectral
  assignment mirrors P(parvocellular, high spectral acuity) vs M(magnocellular,
  high spatial frequency) retinal ganglion cells.

Keeps the double-opponent physics prior (FROB, Gao 2015) as a SUPPORTING module
(not the novelty): pixel-level |V - NV| spectral-edge map guiding the
full-resolution boundary refinement head.
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cross_spectral_sharpening import CrossSpectralSharpening
from decoders.strong_decoder import StrongDecoder

V_BANDS = [0, 1, 2, 3, 4]      # visible: Violet,Blue,Green,Orange,Red
NV_BANDS = [5, 6, 7]           # non-visible: RedEdge,NIR1,NIR2


class NVFullResEncoder(nn.Module):
    """Lightweight NON-VISIBLE pathway: shallow, stride-1, FULL RESOLUTION.

    Produces a full-resolution boundary feature (stride-1, pixel-level) plus a
    ladder of features at the same 4 scales as the V (Swin) backbone, so the
    Cross-Spectral Sharpening gates can be computed at matching resolutions.

    Channels are deliberately SMALL (32-128) vs V's 96-768: NV's signal is
    simple (spectral-edge contrast), so it needs resolution, not capacity.
    """

    def __init__(self, in_bands: int = 3, base: int = 32,
                 nv_channels=(32, 64, 96, 128)):
        super().__init__()
        c2, c3, c4, c5 = nv_channels
        # Full-resolution stem (stride-1): preserve pixel-level boundary at H x W
        self.stem = nn.Sequential(
            nn.Conv2d(in_bands, base, 3, padding=1, bias=False),
            nn.BatchNorm2d(base), nn.ReLU(inplace=True),
            nn.Conv2d(base, base, 3, padding=1, bias=False),
            nn.BatchNorm2d(base), nn.ReLU(inplace=True),
        )
        # Progressive stride-2 downsampling to V's 4 scales (strides 4/8/16/32)
        self.d1 = self._block(base, base)        # -> stride 2
        self.d2 = self._block(base, c2)          # -> stride 4  (matches V C2)
        self.d3 = self._block(c2, c3)            # -> stride 8  (matches V C3)
        self.d4 = self._block(c3, c4)            # -> stride 16 (matches V C4)
        self.d5 = self._block(c4, c5)            # -> stride 32 (matches V C5)
        self.nv_channels = list(nv_channels)
        self.full_channels = base

    @staticmethod
    def _block(cin, cout):
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor):
        f_full = self.stem(x)        # (B, base, H, W)   — full-res boundary feature
        f1 = self.d1(f_full)          # stride 2
        f2 = self.d2(f1)              # stride 4
        f3 = self.d3(f2)              # stride 8
        f4 = self.d4(f3)              # stride 16
        f5 = self.d5(f4)              # stride 32
        return f_full, [f2, f3, f4, f5]


class DASANet(nn.Module):
    """Depth-Asymmetric Spectral Architecture.

    Args:
        v_arch: timm name for the V (deep) pathway. Default Swin-Small (~50M).
        pretrained: ImageNet pretrain for the V Swin (NV CNN trains from scratch).
        img_size: input resolution (must match Swin's img_size, e.g. 384).
        nv_base: full-resolution NV stem width.
    """

    def __init__(self, v_arch: str = "swin_small_patch4_window7_224",
                 pretrained: bool = True, img_size: int = 384, nv_base: int = 32):
        super().__init__()
        from shared.swin_backbone import SwinBackbone
        from sds_net import FullResDoubleOpponent

        # --- V pathway: DEEP Swin (stride-4, semantics) ---
        self.v_enc = SwinBackbone(in_bands=len(V_BANDS), arch=v_arch,
                                  pretrained=pretrained, img_size=img_size)
        c = self.v_enc.channels                       # [96,192,384,768]

        # --- NV pathway: SHALLOW full-res CNN (stride-1, boundary precision) ---
        self.nv_enc = NVFullResEncoder(in_bands=len(NV_BANDS), base=nv_base,
                                       nv_channels=(32, 64, 96, 128))

        # --- Cross-Spectral Sharpening at every decoder scale (NV sharpens V) ---
        self.css = nn.ModuleList([
            CrossSpectralSharpening(c[i], self.nv_enc.nv_channels[i]) for i in range(len(c))
        ])

        # --- Double-opponent physics prior (Gao 2015) — supporting module, not novelty ---
        self.frod = FullResDoubleOpponent(in_bands=8, mid=16)

        # --- Decoder on the SHARPENED V features ---
        self.decoder = StrongDecoder(tuple(reversed(c)))

        # --- Full-resolution boundary refinement head ---
        # coarse mask + FROB prior + NV full-res feature + raw 8-band input
        self.boundary_refine = nn.Sequential(
            nn.Conv2d(1 + 1 + nv_base + 8, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
        )

        # exposed for the loss (deep supervision + edge supervision)
        self.last_auxs = None
        self.last_edge = None
        self.last_frod = None
        self.last_gates = None   # NV sharpening gates, cached for visualization

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[-2:]

        # Physics prior (full-res double-opponent spectral edge)
        frod = self.frod(x)
        self.last_frod = frod

        # Dual asymmetric encoding
        v_feats = self.v_enc(x[:, V_BANDS])              # [C2@96, C3@48, C4@24, C5@12]
        nv_full, nv_feats = self.nv_enc(x[:, NV_BANDS])  # f_full@full-res, [f2,f3,f4,f5]

        # Cross-Spectral Sharpening: NV gate sharpens V at every scale
        sharpened = []
        gates = []
        for i in range(len(v_feats)):
            g = self.css[i].gate(nv_feats[i])
            gates.append(g)
            surround = F.avg_pool2d(v_feats[i], kernel_size=self.css[i].ks, stride=1,
                                     padding=self.css[i].ks // 2)
            sharpened.append(v_feats[i] + self.css[i].alpha * g * (v_feats[i] - surround))
        self.last_gates = gates

        # Decoder (deep-supervision auxs + boundary edge)
        mask, auxs, edge = self.decoder(sharpened)
        self.last_auxs = auxs
        self.last_edge = F.interpolate(edge, size=(H, W), mode="bilinear", align_corners=False)

        # Upsample coarse mask to full resolution
        mask = F.interpolate(mask, size=(H, W), mode="bilinear", align_corners=False)

        # Full-resolution boundary refinement
        mask = mask + self.boundary_refine(torch.cat([mask, frod, nv_full, x], dim=1))
        return mask


if __name__ == "__main__":
    # Smoke test: forward pass + parameter accounting
    m = DASANet(pretrained=False, img_size=384)
    n_all = sum(p.numel() for p in m.parameters())
    n_v = sum(p.numel() for p in m.v_enc.parameters())
    n_nv = sum(p.numel() for p in m.nv_enc.parameters())
    n_css = sum(p.numel() for p in m.css.parameters())
    n_dec = sum(p.numel() for p in m.decoder.parameters())
    x = torch.randn(2, 8, 384, 384)
    y = m(x)
    print(f"output: {tuple(y.shape)}  (expected (2,1,384,384))")
    print(f"params total:  {n_all/1e6:6.2f}M")
    print(f"  V (Swin-S):  {n_v/1e6:6.2f}M")
    print(f"  NV (CNN):    {n_nv/1e6:6.2f}M")
    print(f"  CSS x4:      {n_css/1e6:6.2f}M")
    print(f"  decoder:     {n_dec/1e6:6.2f}M")
