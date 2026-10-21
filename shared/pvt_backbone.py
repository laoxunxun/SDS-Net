"""PVT-v2 backbone (via timm) — the COD-convention encoder (ZoomNeXt/CamoFormer
ladder: B2→B3 ≈ +0.8pp Sα). Mirrors shared/swin_backbone.py: patch-embedding
inflated to N spectral bands with pretrained-RGB seeding (RGB filters on bands
0-2, mean-filter seed on extras so the stem output statistics stay calibrated
with the pretrained downstream blocks).

PVT accepts arbitrary input sizes (no window padding constraint); outputs are
already (B,C,H,W) channels-first."""
from __future__ import annotations

import torch
import torch.nn as nn


class PVTBackbone(nn.Module):
    def __init__(self, in_bands: int = 5, arch: str = "pvt_v2_b3",
                 pretrained: bool = True, grad_ckpt: bool = False,
                 extra_init: str = "mean"):
        super().__init__()
        import timm
        self.encoder = timm.create_model(arch, pretrained=pretrained,
                                         features_only=True,
                                         out_indices=(0, 1, 2, 3))
        if grad_ckpt:
            # FeatureListNet exposes set_grad_checkpointing() (timm>=1.0)
            if hasattr(self.encoder, "set_grad_checkpointing"):
                self.encoder.set_grad_checkpointing()
        self.channels = list(self.encoder.feature_info.channels())  # [64,128,320,512]
        # inflate the overlapping-patch-embed proj to `in_bands` channels
        for _name, mod in self.encoder.named_modules():
            proj = getattr(mod, "proj", None)
            if isinstance(proj, nn.Conv2d):
                old = proj
                new = nn.Conv2d(in_bands, old.out_channels, kernel_size=old.kernel_size,
                                stride=old.stride, padding=old.padding)
                if pretrained and old.weight is not None:
                    with torch.no_grad():
                        new.weight[:, :3] = old.weight
                        if in_bands > 3:
                            if extra_init == "mean":
                                w_mean = old.weight.mean(dim=1, keepdim=True)
                                new.weight[:, 3:] = w_mean * (3.0 / in_bands)
                            else:
                                nn.init.kaiming_normal_(new.weight[:, 3:])
                setattr(mod, "proj", new)
                break

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return list(self.encoder(x))                    # already (B,C,H,W)
