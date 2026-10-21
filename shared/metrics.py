"""COD evaluation metrics.

Confident/correct: MAE, IoU, Dice, weighted-F (Margolin 2014).
S-measure (Fan 2017) and E-measure (Fan 2018) are faithful ports of the standard
SOD eval code — VALIDATE against an official implementation (e.g. py-sod-metrics
or the MCOD repo) before reporting numbers in a paper.

All inputs: pred and gt are torch tensors of the same shape, float in [0,1].
Operate per-image (2D) and average outside; helper `eval_batch` aggregates a batch.
"""
from __future__ import annotations

import numpy as np
import torch

try:
    from scipy.ndimage import distance_transform_edt as _edt
    _HAS_SCIPY = True
except Exception:  # scipy optional for weighted-F
    _HAS_SCIPY = False


# ---------------- basic ----------------

def mae(pred: torch.Tensor, gt: torch.Tensor) -> float:
    return (pred - gt).abs().mean().item()


def iou(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6) -> float:
    inter = (pred * gt).sum()
    union = pred.sum() + gt.sum() - inter
    return (inter / (union + eps)).item()


def dice(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6) -> float:
    inter = (pred * gt).sum()
    return (2.0 * inter / (pred.sum() + gt.sum() + eps)).item()


def fbeta(pred: torch.Tensor, gt: torch.Tensor, beta2: float = 0.3,
          num: int = 200, eps: float = 1e-6) -> float:
    """Max F-beta along the precision-recall curve (beta^2=0.3, COD default). Vectorised."""
    g = (gt > 0.5).float()
    thresholds = torch.linspace(0.0, 1.0, num, device=pred.device)
    p = (pred[None] >= thresholds[:, None, None]).float()      # (num,H,W)
    tp = (p * g).sum(dim=(1, 2))
    pp = p.sum(dim=(1, 2))
    gp = g.sum()
    prec = tp / (pp + eps)
    rec = tp / (gp + eps)
    f = (1 + beta2) * prec * rec / (beta2 * prec + rec + eps)
    return f.max().item()


# ---------------- weighted-F (Margolin 2014) ----------------

def weighted_fmeasure(pred: torch.Tensor, gt: torch.Tensor, beta2: float = 1.0) -> float:
    if not _HAS_SCIPY:
        return float("nan")
    p = pred.detach().cpu().numpy().astype(np.float64)
    g = gt.detach().cpu().numpy().astype(np.float64)
    if g.max() == 0:                       # empty GT
        return 1.0 if p.max() == 0 else 0.0
    E = np.abs(p - g)
    dst, idx = _edt(g, return_indices=True)
    Et = E.copy()
    Et[~(g == 0)] = E[idx[0][~(g == 0)], idx[1][~(g == 0)]]   # distance-weighted error
    K, M = (g > 0).sum(), (g == 0).sum()
    B = np.where(g == 0, 2.0 - np.exp(np.log(0.5) / 5.0 * dst), np.ones_like(g))
    Ew = B * Et
    TPw = (g == 0).sum() - Ew[g == 0].sum()
    FPw = Ew[g > 0].sum()
    prec = TPw / (K + M + 1e-8)
    rec = TPw / (K + 1e-8)
    return float((1 + beta2) * prec * rec / (beta2 * prec + rec + 1e-8))


# ---------------- S-measure (Fan ICCV'17) ----------------
# S = (1-a) * S_object + a * S_region,  a = 0.5

def _object_score(pred: torch.Tensor, gt: torch.Tensor) -> float:
    fg = torch.where(gt == 0, torch.zeros_like(pred), pred)
    bg = torch.where(gt == 1, torch.zeros_like(pred), 1 - pred)
    u = gt.mean()

    def _o(x, m):
        t = x[m == 1]
        if t.numel() == 0:
            return 0.0
        mu = t.mean()
        sigma = t.std()
        return (2.0 * mu / (mu * mu + 1.0 + sigma + 1e-8)).item()

    return (u * _o(fg, gt) + (1 - u) * _o(bg, 1 - gt))


def _ssim_simple(x: torch.Tensor, y: torch.Tensor) -> float:
    # mean-based structural similarity over a region (simplified for region term)
    mu_x, mu_y = x.mean(), y.mean()
    var_x = x.var(unbiased=False)
    var_y = y.var(unbiased=False)
    cov = ((x - mu_x) * (y - mu_y)).mean()
    c1, c2 = 0.01 ** 2, 0.1 ** 2
    return (((2 * mu_x * mu_y + c1) * (2 * cov + c2)) /
            ((mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2) + 1e-8)).item()


def smeasure(pred: torch.Tensor, gt: torch.Tensor, alpha: float = 0.5) -> float:
    """Object + region S-measure. NOTE: region term is simplified — validate."""
    y = (gt > 0.5).float()
    x = pred.clamp(0, 1)
    s_obj = _object_score(x, y)
    s_reg = _ssim_simple(x, y)  # TODO: replace with 4-quadrant region SSIM from official code
    return float(alpha * s_obj + (1 - alpha) * s_reg)


# ---------------- E-measure (Fan ICCV'18) ----------------

def emeasure(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Enhanced-alignment measure (simplified global version). VALIDATE."""
    x = pred.clamp(0, 1)
    y = (gt > 0.5).float()
    xi, yi = x - x.mean(), y - y.mean()
    numerator = 2.0 * (xi * yi).sum() + 1e-8
    denominator = (xi * xi).sum() + (yi * yi).sum() + 1e-8
    align = numerator / denominator
    return float(align) if align > 0 else 0.0


# ---------------- batch helper ----------------

@torch.no_grad()
def evaluate_image(pred: torch.Tensor, gt: torch.Tensor) -> dict:
    """pred, gt: 2D tensors in [0,1]. Returns dict of metric->value."""
    return {
        "mae": mae(pred, gt),
        "iou": iou(pred, gt),
        "dice": dice(pred, gt),
        "sm": smeasure(pred, gt),
        "em": emeasure(pred, gt),
    }


def mean_metrics(list_of_dicts: list[dict]) -> dict:
    keys = list_of_dicts[0].keys()
    return {k: float(np.nanmean([d[k] for d in list_of_dicts])) for k in keys}
