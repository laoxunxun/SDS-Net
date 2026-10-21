"""Swin Transformer backbone (via timm) for strong MS-COD features.
Swin-Tiny: hierarchical ViT, 4 stages [96,192,384,768], ~27.5M, pretrained on
ImageNet. Much stronger than ResNet-50 for COD (matches the ViT-based MCOD SOTA
methods). Patches the patch-embedding to 8 multispectral bands. Permutes Swin's
channels-last output to (B,C,H,W) for the decoder."""
from __future__ import annotations

import torch
import torch.nn as nn


class SwinBackbone(nn.Module):
    def __init__(self, in_bands: int = 8, arch: str = "swin_tiny_patch4_window7_224",
                 pretrained: bool = True, img_size: int = 352, extra_init: str = "mean",
                 grad_ckpt: bool = False):
        super().__init__()
        import timm
        self.encoder = timm.create_model(arch, pretrained=pretrained, features_only=True,
                                         out_indices=(0, 1, 2, 3), img_size=img_size)
        if grad_ckpt:
            # timm 1.0.x Swin ignores the create-time kwarg; with features_only=True the
            # model is wrapped in FeatureListNet, so find stages by class name.
            for mod in self.encoder.modules():
                if mod.__class__.__name__ == "SwinTransformerStage":
                    mod.grad_checkpointing = True
        self.channels = list(self.encoder.feature_info.channels())  # [96,192,384,768]
        # patch the patch-embedding proj to accept `in_bands` channels
        for _name, mod in self.encoder.named_modules():
            if hasattr(mod, "proj") and isinstance(mod.proj, nn.Conv2d):
                old = mod.proj
                mod.proj = nn.Conv2d(in_bands, old.out_channels, kernel_size=old.kernel_size,
                                     stride=old.stride, padding=old.padding)
                if pretrained and old.weight is not None:
                    with torch.no_grad():
                        mod.proj.weight[:, :3] = old.weight          # seed RGB bands
                        if in_bands > 3:
                            if extra_init == "mean":
                                # Seed extra spectral bands with the MEAN of the pretrained RGB
                                # filters (scaled so the total input magnitude matches RGB):
                                # keeps patch-embed output statistics consistent with the
                                # pretrained stem instead of injecting Kaiming noise that the
                                # pretrained downstream blocks were never calibrated for.
                                w_mean = old.weight.mean(dim=1, keepdim=True)  # (out,1,1,1)
                                mod.proj.weight[:, 3:] = w_mean * (3.0 / in_bands)
                            else:  # legacy random init (kept for reproduction of old runs)
                                nn.init.kaiming_normal_(mod.proj.weight[:, 3:])
                break

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        feats = self.encoder(x)                                 # list of (B,H,W,C) channels-last
        return [f.permute(0, 3, 1, 2).contiguous() for f in feats]  # -> (B,C,H,W)
