"""Cross-Spectral Sharpening (CSS) — cross-modal lateral inhibition.

This is the headline mechanism of DASA. It is NOT a new biology, NOT a new
attention variant — it is the first CROSS-MODAL instantiation of lateral
inhibition as a feature-fusion operator, where the inhibition kernel that
sharpens one pathway is DRIVEN BY A DIFFERENT SPECTRAL PATHWAY.

Biological grounding
--------------------
Cross-modality sharpening of cortical processing is established neuroscience:
one modality's input sharpens another modality's neural representation via
interneuron-mediated inhibition (Ibrahim et al., "Cross-modality Sharpening
of Visual Cortical Processing", Neuron 2016, 338 citations). Classical lateral
inhibition itself (Hartline, Nobel 1967) sharpens a response by subtracting a
blurred surround:  out = center - k * surround  (a.k.a. unsharp masking).

Computation
-----------
Classical (intra-modal) lateral inhibition, applied to a single feature map:

        V' = V + k * (V - blur(V))              k = scalar, learned globally

The sharpening strength k is the SAME everywhere. LICNN (Cao 2018, AAAI) and
SpiLiFormer (ICCV 2025) build deep modules on exactly this intra-modal idea.

Ours (cross-spectral): the inhibition strength is a PER-PIXEL MAP produced by
a DIFFERENT spectral pathway (NV = non-visible):

        V' = V + alpha * g_NV * (V - blur(V))

where  g_NV = sigmoid(MLP(NV_feat))  in (0,1)  encodes WHERE the NV pathway
sees a sharp, high-confidence spectral edge (the camouflage leak boundary).

Key property: NV NEVER contributes its own content to V. NV only GATES V's
own high-frequency detail. This is SHARPENING, not fusion — and it is what
makes it distinct from:

  * Attention / gated fusion  out = g*A + (1-g)*B   -> MIXES two modalities'
    content (NV's values enter V). Ours never mixes content; NV only modulates
    V's spatial frequency.
  * Intra-modal lateral inhibition (LICNN, SpiLiFormer): the gate comes from
    the SAME modality's neighbors. Ours draws the gate from a different
    spectral group — the cross-modal step that has no DL precedent.
  * Double-opponent filtering (Gao 2015, color constancy): a spectral-edge
    FILTER computed on a single image. CSS is a cross-pathway OPERATOR whose
    gate comes from a separately-encoded spectral group.

Physics
-------
Camouflage matches background in V (low contrast, fuzzy boundary) but leaks in
NV (high contrast, sharp boundary). So the reliable boundary signal lives in
NV. CSS lets the reliable modality (NV) sharpen the unreliable one (V) exactly
where NV is confident — transferring NV's boundary sharpness into V's deep
semantic representation, instead of trying to recover detail V lost at stride-4.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossSpectralSharpening(nn.Module):
    """One-scale cross-spectral sharpening gate.

    Args:
        c_v:  channel count of the visible (to-be-sharpened) features.
        c_nv: channel count of the non-visible (gate-source) features.
        reduction: MLP bottleneck for the NV gate.
        ks: kernel size of the surround (blur) used for the lateral context.
    """

    def __init__(self, c_v: int, c_nv: int, reduction: int = 4, ks: int = 7):
        super().__init__()
        # NV -> per-pixel inhibition/sharpening gate. WHERE does NV see a sharp edge?
        hidden = max(c_nv // reduction, 8)
        self.gate = nn.Sequential(
            nn.Conv2d(c_nv, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 1),
            nn.Sigmoid(),                       # g_NV in (0,1)
        )
        self.ks = ks
        # Per-channel sharpening scalar (init small -> starts near identity, stable training)
        self.alpha = nn.Parameter(torch.full((1, c_v, 1, 1), 0.1))

    def forward(self, v: torch.Tensor, nv: torch.Tensor) -> torch.Tensor:
        """
        v:  (B, Cv, H, W) visible semantic features (deep, fuzzy boundary).
        nv: (B, Cnv, H, W) non-visible features (shallow, sharp boundary), same HxW.
        Returns: (B, Cv, H, W) V sharpened by the NV-driven inhibition map.
        """
        g = self.gate(nv)                                     # (B,1,H,W) NV gate
        surround = F.avg_pool2d(v, kernel_size=self.ks, stride=1,
                                padding=self.ks // 2)        # blurred V (lateral context)
        detail = v - surround                                 # V's own high-frequency detail
        return v + self.alpha * g * detail                    # cross-spectral sharpening


class CrossSpectralSharpeningGate(nn.Module):
    """Variant that ALSO exposes the NV gate for visualization / ablation.

    Returns (v_sharp, g_NV) so we can show WHERE NV decided to sharpen.
    """

    def __init__(self, c_v: int, c_nv: int, reduction: int = 4, ks: int = 7):
        super().__init__()
        self.css = CrossSpectralSharpening(c_v, c_nv, reduction, ks)

    def forward(self, v: torch.Tensor, nv: torch.Tensor):
        g = self.css.gate(nv)
        surround = F.avg_pool2d(v, kernel_size=self.css.ks, stride=1, padding=self.css.ks // 2)
        v_sharp = v + self.css.alpha * g * (v - surround)
        return v_sharp, g
