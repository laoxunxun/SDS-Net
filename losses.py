"""MSO-Net losses: BCE + IoU + edge + opponency sparsity regulariser.

L = L_bce + l_iou*L_iou + l_edge*L_edge + l_opp*L_opponency
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def bce_with_logits(logits: torch.Tensor, mask: torch.Tensor, pos_weight=None) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, mask, pos_weight=pos_weight)


def iou_loss(logits: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred = torch.sigmoid(logits)
    inter = (pred * mask).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + mask.sum(dim=(2, 3)) - inter
    return (1 - (inter + eps) / (union + eps)).mean()


def edge_loss(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Boundary consistency via Sobel of pred vs Sobel of GT."""
    kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                      device=logits.device).view(1, 1, 3, 3)
    ky = kx.transpose(2, 3)
    pred = torch.sigmoid(logits)
    gx = F.conv2d(pred, kx, padding=1); gy = F.conv2d(pred, ky, padding=1)
    hx = F.conv2d(mask, kx, padding=1); hy = F.conv2d(mask, ky, padding=1)
    return (0.5 * ((gx - hx) ** 2 + (gy - hy) ** 2)).mean()


def ssim_loss(logits: torch.Tensor, mask: torch.Tensor, win: int = 11) -> torch.Tensor:
    """1 - SSIM (structure similarity) — directly optimizes Sm/Em (alignment metrics)."""
    pred = torch.sigmoid(logits)
    pad = win // 2
    mu_p = F.avg_pool2d(pred, win, 1, pad); mu_g = F.avg_pool2d(mask, win, 1, pad)
    var_p = F.avg_pool2d(pred * pred, win, 1, pad) - mu_p * mu_p
    var_g = F.avg_pool2d(mask * mask, win, 1, pad) - mu_g * mu_g
    cov = F.avg_pool2d(pred * mask, win, 1, pad) - mu_p * mu_g
    c1, c2 = 0.01 ** 2, 0.1 ** 2
    ssim = ((2 * mu_p * mu_g + c1) * (2 * cov + c2)) / ((mu_p ** 2 + mu_g ** 2 + c1) * (var_p + var_g + c2))
    return 1.0 - ssim.mean()


def sobel_edges(mask: torch.Tensor) -> torch.Tensor:
    """Binary edge target {0,1} from a binary mask via Sobel magnitude."""
    kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                      device=mask.device).view(1, 1, 3, 3)
    ky = kx.transpose(2, 3)
    gx = F.conv2d(mask, kx, padding=1); gy = F.conv2d(mask, ky, padding=1)
    mag = torch.sqrt(gx * gx + gy * gy + 1e-6)
    return (mag > 0.5).float()


# ---- clDice (topology-preserving loss for thin structures / protrusions) ----
# Shit et al., "clDice — A Novel Topology-Preserving Loss Function", CVPR 2021.
# Soft-skeleton via iterative morphology; clDice keeps thin branches/connections
# that region losses (BCE/IoU) wash out — directly targets the boundary-detail goal.

def soft_erode(img: torch.Tensor) -> torch.Tensor:
    p1 = -F.max_pool2d(-img, (3, 1), 1, (1, 0))
    p2 = -F.max_pool2d(-img, (1, 3), 1, (0, 1))
    return torch.min(p1, p2)


def soft_dilate(img: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(img, (3, 3), 1, (1, 1))


def soft_open(img: torch.Tensor) -> torch.Tensor:
    return soft_dilate(soft_erode(img))


def soft_skel(img: torch.Tensor, iters: int = 3) -> torch.Tensor:
    img1 = soft_open(img)
    skel = F.relu(img - img1)
    for _ in range(iters):
        img = soft_erode(img)
        img1 = soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel


def cldice_loss(logits: torch.Tensor, mask: torch.Tensor, iters: int = 3, smooth: float = 1e-6) -> torch.Tensor:
    """1 - clDice. Operates on soft predictions to preserve thin topology."""
    pred = torch.sigmoid(logits)
    skel_pred = soft_skel(pred, iters)
    skel_gt = soft_skel(mask, iters)
    tprec = (skel_pred * mask).sum(dim=(2, 3)) / (skel_pred.sum(dim=(2, 3)) + smooth)
    tsens = (skel_gt * pred).sum(dim=(2, 3)) / (skel_gt.sum(dim=(2, 3)) + smooth)
    cldice = 2.0 * tprec * tsens / (tprec + tsens + smooth)
    return (1.0 - cldice).mean()



class MSOLoss(nn.Module):
    def __init__(self, l_iou: float = 1.0, l_edge: float = 1.0, l_opp: float = 0.1,
                 l_phys: float = 0.5, weighted: bool = False, l_ds: float = 1.0, l_ssim: float = 0.0,
                 l_anomaly: float = 0.0, l_recon: float = 0.0, l_align: float = 0.0, l_cldice: float = 0.0,
                 l_boundary: float | None = None, l_xmc: float = 0.0):
        super().__init__()
        self.l_iou, self.l_edge, self.l_opp = l_iou, l_edge, l_opp
        self.l_phys, self.l_ds, self.l_ssim = l_phys, l_ds, l_ssim
        # Explicit boundary-branch weight. Historically the Sobel-edge loss on logits and the
        # BCE on the model's boundary branch were BOTH scaled by l_edge (double-weighted).
        # l_boundary=None preserves the legacy behaviour (falls back to l_edge); new configs
        # should set it explicitly.
        self.l_boundary = l_edge if l_boundary is None else l_boundary
        self.l_anomaly = l_anomaly
        self.l_recon = l_recon
        self.l_align = l_align
        self.l_cldice = l_cldice      # topology-preserving loss (thin structures / protrusions)
        self.l_xmc = l_xmc            # cross-modal contrast asymmetry reweighting (physics v2)
        self.weighted = weighted       # foreground-weighted BCE (helps tiny objects, MCOD mean area 0.43%)
        self.model = None
        self._warned = set()

    def bind(self, model: nn.Module):
        """Bind the model so the engine's loss_fn(logits, mask) can reach physics losses."""
        self.model = model
        self._warned = set()

    def _expect(self, key: str, ok: bool):
        """Warn ONCE per missing term so a misconfigured weight is visible instead of
        silently training a different objective than configured."""
        if not ok and key not in self._warned:
            self._warned.add(key)
            print(f"[MSOLoss WARNING] l_{key} > 0 but model did not provide the corresponding "
                  f"output this step — this loss term is being DROPPED. Check the model/config contract.",
                  flush=True)

    def forward(self, logits: torch.Tensor, mask: torch.Tensor,
                model: nn.Module | None = None) -> torch.Tensor:
        mdl = model if model is not None else self.model
        pw = None
        if self.weighted:
            pos = float(mask.sum().clamp(min=1.0)); neg = mask.numel() - pos
            pw = torch.tensor(min(max(neg / pos, 1.0), 50.0), device=logits.device)
        loss = bce_with_logits(logits, mask, pos_weight=pw) + self.l_iou * iou_loss(logits, mask)
        loss = loss + self.l_edge * edge_loss(logits, mask)
        if self.l_ssim > 0:
            loss = loss + self.l_ssim * ssim_loss(logits, mask)
        if self.l_cldice > 0:
            loss = loss + self.l_cldice * cldice_loss(logits, mask)
        if mdl is not None:
            if self.l_xmc > 0:
                cube01 = getattr(mdl, "last_cube01", None)
                if cube01 is not None:
                    from physics_v2 import xmc_weight
                    w = xmc_weight(cube01, mask)
                    loss = loss + self.l_xmc * (F.binary_cross_entropy_with_logits(
                        logits, mask, reduction="none") * w).mean()
                else:
                    self._expect("xmc", False)
            if self.l_opp > 0:
                if hasattr(mdl, "opponency_reg"):
                    loss = loss + self.l_opp * mdl.opponency_reg()
                else:
                    self._expect("opp", False)
            if self.l_phys > 0:
                if hasattr(mdl, "physics_consistency") and getattr(mdl, "last_aux", None) is not None:
                    loss = loss + self.l_phys * mdl.physics_consistency()
                else:
                    self._expect("phys", False)
            if self.l_boundary > 0:
                if getattr(mdl, "last_edge", None) is not None:
                    loss = loss + self.l_boundary * bce_with_logits(mdl.last_edge, sobel_edges(mask))
                else:
                    self._expect("boundary", False)
            if self.l_ds > 0:                                            # deep supervision (strong decoder)
                if getattr(mdl, "last_auxs", None):
                    for ax in mdl.last_auxs:
                        gt_d = F.interpolate(mask, size=ax.shape[-2:], mode="nearest")
                        loss = loss + self.l_ds * bce_with_logits(ax, gt_d)
                else:
                    self._expect("ds", False)
            if self.l_anomaly > 0 and hasattr(mdl, "anomaly_reg") \
                    and getattr(mdl, "saa_branch", None) is not None:
                loss = loss + self.l_anomaly * mdl.anomaly_reg(mask)
            if self.l_recon > 0 and hasattr(mdl, "edm") and mdl.edm is not None:
                rec = mdl.edm.recon_loss() + 0.01 * mdl.edm.endmember_diversity()
                loss = loss + self.l_recon * rec
            if self.l_align > 0 and hasattr(mdl, "align_loss"):
                loss = loss + self.l_align * mdl.align_loss(mask)
        return loss
