"""Stratified evaluation by target size — where does PGSA-Net win/lose?

Bins the 500 test images by target-area fraction (computed from GT), then reports
per-bin S-measure / F-measure for PGSA-Net and each available competitor. Shows
whether PGSA-Net's edge holds across object sizes (esp. tiny targets, MCOD's hard case).
Zero training — uses saved per-image predictions.

Run:  python stratified_eval.py
(uses mmcv env: py_sod_metrics)
"""
import os, sys, json
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from shared.dataset import load_mask

import py_sod_metrics as PSM

EXP = os.path.join(ROOT, "experiments")
GT_DIR = r"G:\MCOD\TestDataset\GT"
# PGSA + competitors that have saved preds (CamoDiffusion/RUN excluded — no preds)
METHODS = {
    "PGSA-Net": "exp34_sds_physaug",
    "MSCAF": "cmp_MSCAF",
    "ZoomNeXt": "cmp_ZoomNeXt",
    "SARNet": "cmp_SARNet",
    "ERRNet": "cmp_ERRNet",
    "HitNet": "cmp_HitNet",
    "FEDER": "cmp_FEDER",
}
NBIN = 3   # terciles by target-area fraction


def per_image(pred01, gt01):
    pred_u = (pred01 * 255).clip(0, 255).astype(np.uint8)
    gt_u = ((gt01 > 0.5).astype(np.uint8) * 255)
    sm = PSM.Smeasure(); sm.step(pred=pred_u, gt=gt_u)
    fm = PSM.Fmeasure(); fm.step(pred=pred_u, gt=gt_u)
    fb = float(np.asarray(fm.get_results()["fm"]["curve"]).mean())
    return float(sm.get_results()["sm"]), fb


def load_pred(d, name):
    return np.asarray(Image.open(os.path.join(EXP, d, "preds", name + ".png")), dtype=np.float32) / 255.0


def main():
    names = sorted(os.path.splitext(f)[0] for f in os.listdir(os.path.join(EXP, "exp34_sds_physaug", "preds")) if f.endswith(".png"))
    print(f"[stratified] {len(names)} images; methods={list(METHODS)}")

    # 1) compute target-area fraction per image + per-image metrics per method
    area = np.zeros(len(names))
    M = {m: {"Sm": np.zeros(len(names)), "Fb": np.zeros(len(names))} for m in METHODS}
    for i, name in enumerate(names):
        gt = load_mask(os.path.join(GT_DIR, name + ".jpg"))
        area[i] = gt.sum() / gt.size
        for m, d in METHODS.items():
            pred = load_pred(d, name)
            if pred.shape != gt.shape:
                pred = np.asarray(Image.fromarray((pred * 255).clip(0, 255).astype(np.uint8)).resize(
                    (gt.shape[1], gt.shape[0]), Image.BILINEAR), dtype=np.float32) / 255.0
            sm, fb = per_image(pred, gt)
            M[m]["Sm"][i] = sm; M[m]["Fb"][i] = fb
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(names)}", flush=True)

    # 2) bin by area terciles
    qs = np.quantile(area, [1 / NBIN, 2 / NBIN]) if NBIN == 3 else np.quantile(area, np.linspace(1 / NBIN, 1 - 1 / NBIN, NBIN - 1))
    edges = [-1] + list(qs) + [1e9]
    labels = []
    for b in range(len(edges) - 1):
        lo, hi = edges[b], edges[b + 1]
        labels.append(f"<{qs[b]:.3%}" if b == 0 else (f">{qs[b-1]:.3%}" if b == len(edges) - 2 else f"{qs[b-1]:.3%}-{qs[b]:.3%}"))
    # simpler labels
    labels = [f"bin{b}({edges[b]:.4f}-{edges[b+1]:.4f})" for b in range(len(edges) - 1)]
    bins = np.digitize(area, qs)   # 0..NBIN-1

    print("\n" + "=" * 80)
    print(f"Target-area fraction tercile edges: {qs.tolist()}  (median area = {np.median(area):.4%})")
    print(f"{'bin (area range)':30s} | n | " + " | ".join(f"{m} Sm/Fb" for m in METHODS))
    print("-" * 80)
    out = {"edges": [float(x) for x in qs], "bins": {}}
    for b in range(NBIN):
        idx = np.where(bins == b)[0]
        if len(idx) == 0:
            continue
        cells = " | ".join(f"{M[m]['Sm'][idx].mean():.3f}/{M[m]['Fb'][idx].mean():.3f}" for m in METHODS)
        rng = f"[{area[idx].min():.4%}, {area[idx].max():.4%}]"
        print(f"{rng:30s} | {len(idx):3d} | {cells}")
        out["bins"][rng] = {"n": int(len(idx)),
                            "metrics": {m: {"Sm": float(M[m]["Sm"][idx].mean()), "Fb": float(M[m]["Fb"][idx].mean())} for m in METHODS}}
    print("=" * 80)
    # also overall (sanity vs official)
    print("overall (sanity): " + " | ".join(f"{m} Sm={M[m]['Sm'].mean():.4f}" for m in ["PGSA-Net", "MSCAF", "ZoomNeXt"]))

    with open(os.path.join(EXP, "stratified_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\n[stratified] saved -> experiments/stratified_results.json")


if __name__ == "__main__":
    main()
