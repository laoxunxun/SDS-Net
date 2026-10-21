"""MSO-Net = Physics-Guided Peripheral Sensor-Hub (Paper 1).
physics mode:
  'bio'   : CS-SO + LPSI + SpectralDerivative + saliency-consistency (the bionic front-end)
  'unmix' : DSU spectral unmixing -> abundance maps + reconstruction-consistency
            (structurally non-redundant with convs; the physics option being tested for
             additive value after 'bio' proved neutral vs a plain backbone)
decoder: 'boundary' | 'plain'. antagonistic_stem: DOC stem on/off.
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backbones import MSBackbone                                            # noqa: E402
from modules import CSSO, LPSI, SpectralDerivative, physics_saliency_prior, SpectralPredictiveCoder  # noqa: E402
from dsu_front import DSUFront                                               # noqa: E402
from decoders.boundary_decoder import BoundaryGuidedDecoder                 # noqa: E402
from decoders.plain_fpn import PlainFPNDecoder                              # noqa: E402
from decoders.strong_decoder import StrongDecoder                           # noqa: E402

RED, RED_EDGE, NIR1, NIR2 = 4, 5, 6, 7
NDVI_INIT = [(NIR1, RED), (NIR2, RED), (NIR1, RED_EDGE), (NIR2, RED_EDGE)]


class MSONet(nn.Module):
    def __init__(self, in_bands: int = 8, receptors: int = 6, num_indices: int = 4,
                 backbone_arch: str = "resnet34", pretrained: bool = False,
                 decoder: str = "plain", antagonistic_stem: bool = True,
                 physics: str = "bio", unmix_k: int = 5):
        super().__init__()
        self.physics = physics
        self.decoder_kind = decoder

        if physics == "unmix":
            self.phys_front = DSUFront(in_bands, k=unmix_k)
            ev_ch = self.phys_front.out_channels
        elif physics == "predcode":
            self.pc = SpectralPredictiveCoder(in_bands)
            ev_ch = self.pc.out_channels
        else:
            self.csso = CSSO(in_bands, K=receptors, init_pairs=NDVI_INIT)
            self.lpsi = LPSI(in_bands, num_indices, init_pairs=NDVI_INIT)
            self.deriv = SpectralDerivative(in_bands)
            ev_ch = self.csso.out_channels + num_indices + self.deriv.out_channels

        self.spec_proj = nn.Conv2d(ev_ch, 32, 1)
        self.aux_head = nn.Conv2d(32, 1, 1)
        self.backbone = MSBackbone(in_channels=in_bands, arch=backbone_arch,
                                   pretrained=pretrained, antagonistic_stem=antagonistic_stem)
        chans = tuple(reversed(self.backbone.channels))
        if decoder == "boundary":
            self.decoder = BoundaryGuidedDecoder(chans, c_spec=32)
        elif decoder == "strong":
            self.decoder = StrongDecoder(chans, c_spec=32)
        else:
            self.decoder = PlainFPNDecoder(chans, c_spec=32)

        self.last_cube = None
        self.last_aux = None
        self.last_auxs = None
        self.last_edge = None
        self.last_recon = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.physics == "unmix":
            ev, recon = self.phys_front(x)
            self.last_recon = recon
        elif self.physics == "predcode":
            ev = self.pc(x)
        else:
            do = self.csso(x); idx = self.lpsi(x); drv = self.deriv(x)
            ev = torch.cat([do, idx, drv], dim=1)
        spec_map = self.spec_proj(ev)
        self.last_cube = x
        self.last_aux = self.aux_head(spec_map)
        feats = self.backbone(x)
        if self.decoder_kind == "strong":
            mask, auxs, edge = self.decoder(feats, spec_map)
            self.last_auxs = auxs
            self.last_edge = F.interpolate(edge, size=x.shape[-2:], mode="bilinear", align_corners=False)
        elif self.decoder_kind == "boundary":
            mask, edge = self.decoder(feats, spec_map)
            self.last_auxs = None
            self.last_edge = F.interpolate(edge, size=x.shape[-2:], mode="bilinear", align_corners=False)
        else:
            mask = self.decoder(feats, spec_map)
            self.last_auxs = None
            self.last_edge = None
        return F.interpolate(mask, size=x.shape[-2:], mode="bilinear", align_corners=False)

    def opponency_reg(self) -> torch.Tensor:        # bio mode only (0 in unmix mode)
        if self.physics == "bio":
            return self.csso.reg_loss()
        return torch.tensor(0.0, device=self.last_cube.device)

    def physics_consistency(self) -> torch.Tensor:
        if self.physics == "unmix":
            return F.l1_loss(self.last_recon, self.last_cube)              # reconstruction = physical model fit
        if self.physics == "predcode":
            return torch.tensor(0.0, device=self.last_cube.device)         # residual is the signal; no extra phys loss
        target = physics_saliency_prior(self.last_cube)
        return F.mse_loss(torch.sigmoid(self.last_aux), target)

    def physics_loss(self) -> torch.Tensor:
        return self.opponency_reg() if self.physics == "bio" else torch.tensor(0.0, device=self.last_cube.device)
