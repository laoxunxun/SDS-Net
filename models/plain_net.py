"""Plain no-physics baseline: 8-channel ResNet (pretrained conv1 patched) + decoder.
decoder: 'plain' (tiny FPN) | 'strong' (multi-scale + deep-sup + boundary, ~2.4M).
Used for the decisive ablation (does the physics hub add value?) AND to isolate the
architecture fix (strong decoder). No CS-SO/LPSI/deriv, no DOC stem, no physics-consistency."""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backbones import MSBackbone                  # noqa: E402
from decoders.plain_fpn import PlainFPNDecoder    # noqa: E402
from decoders.strong_decoder import StrongDecoder  # noqa: E402


class PlainNet(nn.Module):
    def __init__(self, in_bands: int = 8, backbone_arch: str = "resnet50",
                 pretrained: bool = True, decoder: str = "plain", rgb_bands=None,
                 backbone_type: str = "resnet", img_size: int = 352,
                 grad_ckpt: bool = False):
        super().__init__()
        self.decoder_kind = decoder
        self.rgb_bands = rgb_bands                     # if set (e.g. [4,2,1]=R,G,B), slice 8-band cube to RGB
        if backbone_type == "swin":
            from shared.swin_backbone import SwinBackbone
            self.backbone = SwinBackbone(in_bands=in_bands, arch=backbone_arch,
                                         pretrained=pretrained, img_size=img_size,
                                         grad_ckpt=grad_ckpt)
        else:
            self.backbone = MSBackbone(in_channels=in_bands, arch=backbone_arch,
                                       pretrained=pretrained, antagonistic_stem=False)
        chans = tuple(reversed(self.backbone.channels))
        self.decoder = StrongDecoder(chans) if decoder == "strong" else PlainFPNDecoder(chans)
        self.last_auxs = None
        self.last_edge = None
        self.last_aux = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.rgb_bands is not None:
            x = x[:, list(self.rgb_bands)]             # 8-band -> 3-band RGB
        feats = self.backbone(x)
        if self.decoder_kind == "strong":
            mask, auxs, edge = self.decoder(feats)
            self.last_auxs = auxs
            self.last_edge = F.interpolate(edge, size=x.shape[-2:], mode="bilinear", align_corners=False)
        else:
            mask = self.decoder(feats)
            self.last_auxs = None
            self.last_edge = None
        return F.interpolate(mask, size=x.shape[-2:], mode="bilinear", align_corners=False)
