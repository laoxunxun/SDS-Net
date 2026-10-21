"""SFCBNet — frozen spectral-blind Swin + Spectral-Frequency Coupled Bridge (4-scale)."""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sfc_bridge import SFCBridge
from decoders.strong_decoder import StrongDecoder

RGB_BANDS = [4, 2, 1]


class SFCBNet(nn.Module):
    def __init__(self, arch: str = "swin_small_patch4_window7_224", pretrained: bool = True, img_size: int = 384):
        super().__init__()
        from shared.swin_backbone import SwinBackbone
        self.rgb_backbone = SwinBackbone(in_bands=3, arch=arch, pretrained=pretrained, img_size=img_size)
        for p in self.rgb_backbone.parameters():                 # FROZEN -> spectral-blind
            p.requires_grad = False
        c = self.rgb_backbone.channels                           # [96,192,384,768]
        self.bridge = SFCBridge(n_bands=8, chans=tuple(c))
        # per-scale injection strength (init small -> starts near frozen features)
        self.inject = nn.ParameterList([nn.Parameter(torch.tensor(0.1)) for _ in c])
        self.decoder = StrongDecoder(tuple(reversed(c)))
        self.last_auxs = None
        self.last_edge = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rgb = x[:, RGB_BANDS]
        feats = self.rgb_backbone(rgb)                           # frozen features [4 scales]
        bridge_feats, anomaly = self.bridge(x)                   # spectral-freq coupled, 4 scales
        feats = [feats[i] + self.inject[i] * bridge_feats[i] for i in range(len(feats))]
        mask, auxs, edge = self.decoder(feats)
        self.last_auxs = auxs
        self.last_edge = F.interpolate(edge, size=x.shape[-2:], mode="bilinear", align_corners=False)
        mask = F.interpolate(mask, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return mask


if __name__ == "__main__":
    import torch
    m = SFCBNet(pretrained=False, img_size=384)
    n_tr = sum(p.numel() for p in m.parameters() if p.requires_grad)
    x = torch.randn(2, 8, 384, 384)
    print("out", tuple(m(x).shape), "trainable", f"{n_tr/1e6:.2f}M")
