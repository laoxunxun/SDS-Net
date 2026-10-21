"""Paired significance test (Wilcoxon signed-rank) on per-image metrics.

PGSA-Net (exp34_sds_physaug) vs each competitor that has saved per-image predictions.
Computes per-image S-measure and weighted-F (scalar per image), then a paired Wilcoxon
test across the 500 test images. Turns the "comparable to SOTA" claim from an assertion
into a test with p-values, effect sizes, and win-rates.

Run:  python sig_test.py
(uses the mmcv env: torch + py_sod_metrics + scipy)

Note: CamoDiffusion and RUN per-image predictions were not saved, so they are excluded
from the paired test (stated honestly in the paper). The available top-tier deterministic
competitors MSCAF (Sm 0.852) and ZoomNeXt (0.846) ARE included.
"""
import os, sys, json
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from shared.dataset import load_mask

import py_sod_metrics as PSM
from scipy.stats import wilcoxon

EXP = os.path.join(ROOT, "experiments")
GT_DIR = r"G:\MCOD\TestDataset\GT"
OURS = "exp34_sds_physaug"   # PGSA-Net (seed42, the paper's headline single run)
COMP = {
    "MSCAF": "cmp_MSCAF",       # Sm 0.852 (top deterministic)
    "ZoomNeXt": "cmp_ZoomNeXt", # Sm 0.846
    "SARNet": "cmp_SARNet",     # 0.783
    "ERRNet": "cmp_ERRNet",     # 0.769
    "HitNet": "cmp_HitNet",     # 0.746
    "FEDER": "cmp_FEDER",       # 0.735
}
METRICS = ["Sm", "wF"]   # both scalar per image (Smeasure, WeightedFmeasure)


def per_image_metrics(pred01, gt01):
    """pred01, gt01 in [0,1], 2D. Returns per-image {Sm, wF} scalars (same units as official)."""
    pred_u = (pred01 * 255).clip(0, 255).astype(np.uint8)
    gt_u = ((gt01 > 0.5).astype(np.uint8) * 255)
    out = {}
    sm = PSM.Smeasure(); sm.step(pred=pred_u, gt=gt_u); out["Sm"] = float(sm.get_results()["sm"])
    wf = PSM.WeightedFmeasure(); wf.step(pred=pred_u, gt=gt_u); out["wF"] = float(wf.get_results()["wfm"])
    return out


def load_pred(exp_dir, name):
    p = os.path.join(EXP, exp_dir, "preds", name + ".png")
    return np.asarray(Image.open(p), dtype=np.float32) / 255.0


def main():
    names = sorted(os.path.splitext(f)[0] for f in os.listdir(os.path.join(EXP, OURS, "preds")) if f.endswith(".png"))
    print(f"[sig] {len(names)} test images; ours={OURS}; competitors={list(COMP)}")

    # per-image metric matrices: method -> metric -> np.array(n)
    methods = {"PGSA-Net": OURS, **{k: v for k, v in COMP.items()}}
    M = {m: {k: np.zeros(len(names)) for k in METRICS} for m in methods}
    for i, name in enumerate(names):
        gt = load_mask(os.path.join(GT_DIR, name + ".jpg"))
        for m, d in methods.items():
            pred = load_pred(d, name)
            # align pred to gt resolution (preds saved at GT-native res, but be safe)
            if pred.shape != gt.shape:
                pred = np.asarray(Image.fromarray((pred * 255).clip(0, 255).astype(np.uint8)).resize(
                    (gt.shape[1], gt.shape[0]), Image.BILINEAR), dtype=np.float32) / 255.0
            r = per_image_metrics(pred, gt)
            for k in METRICS:
                M[m][k][i] = r[k]
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(names)}", flush=True)

    # report: per-method mean, then paired Wilcoxon PGSA vs each competitor
    print("\n" + "=" * 78)
    print(f"{'Method':12s} | " + " | ".join(f"{k} mean (sem)" for k in METRICS) + " | win% vs PGSA")
    print("-" * 78)
    sem = lambda a: float(a.std(ddof=1) / np.sqrt(len(a)))
    for m in methods:
        cells = " | ".join(f"{M[m][k].mean():.4f} ({sem(M[m][k]):.4f})" for k in METRICS)
        if m == "PGSA-Net":
            print(f"{m:12s} | {cells} |   —")
        else:
            win = (M["PGSA-Net"]["Sm"] > M[m]["Sm"]).mean() * 100
            print(f"{m:12s} | {cells} | {win:5.1f}%")
    print("=" * 78)

    print("\nPaired Wilcoxon signed-rank  (H0: PGSA == competitor, per-image)")
    print(f"{'vs':12s} | " + " | ".join(f"{k}: Δmean  p(win%)" for k in METRICS))
    print("-" * 78)
    rows = {}
    for m, d in COMP.items():
        rows[m] = {}
        cells = []
        for k in METRICS:
            diff = M["PGSA-Net"][k] - M[m][k]
            # drop exact-zero pairs for wilcoxon; one-sided alt "PGSA greater" where appropriate
            try:
                w = wilcoxon(M["PGSA-Net"][k], M[m][k], zero_method="wilcox",
                             alternative="two-sided")
                p = w.pvalue
            except ValueError:
                p = float("nan")
            win = (M["PGSA-Net"][k] > M[m][k]).mean() * 100
            cells.append(f"{k}: {diff.mean():+.4f}  p={p:.2e} ({win:.0f}%)")
            rows[m][k] = {"delta_mean": float(diff.mean()), "p_two_sided": float(p),
                          "win_pct": float(win),
                          "ours_mean": float(M['PGSA-Net'][k].mean()),
                          "comp_mean": float(M[m][k].mean())}
        print(f"{m:12s} | " + " | ".join(cells))
    print("=" * 78)

    summary = {
        "ours": OURS,
        "n_images": len(names),
        "mean_metrics": {m: {k: float(M[m][k].mean()) for k in METRICS} for m in methods},
        "paired_wilcoxon_vs_PGSA": rows,
    }
    with open(os.path.join(EXP, "sig_test_results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[sig] saved -> experiments/sig_test_results.json")


if __name__ == "__main__":
    main()
