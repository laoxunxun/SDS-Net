"""Aggregate exp34 (3 seeds) into mean +/- std for the paper."""
import json, os, numpy as np
EXPS = ["exp34_sds_physaug", "exp34_s123", "exp34_s2024"]
import os
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments")
METRICS = ["MAE", "Sm", "Em", "wF", "Fb"]
rows = {e: None for e in EXPS}
for e in EXPS:
    p = os.path.join(ROOT, e, "metrics.json")
    if os.path.exists(p):
        rows[e] = json.load(open(p))["official"]
available = [e for e in EXPS if rows[e]]
print(f"Seeds available: {len(available)} / 3  ({available})")
if len(available) >= 1:
    print("\n" + "=" * 60)
    print(f"{'Metric':8s} | " + " | ".join(f"{e:>20s}" for e in available) + " |   mean +/- std")
    print("-" * 60)
    for k in METRICS:
        vals = [rows[e][k] for e in available]
        ms = f"{np.mean(vals):.4f} +/- {np.std(vals, ddof=1):.4f}" if len(vals) > 1 else f"{vals[0]:.4f}"
        print(f"{k:8s} | " + " | ".join(f"{rows[e][k]:>20.4f}" for e in available) + f" |   {ms}")
    print("=" * 60)
