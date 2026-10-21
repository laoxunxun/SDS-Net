"""Robustness suites R1-fix + R2 (inference-only; server bug-fixed weights now local).

R1-fix: spectral-shift robustness with the CORRECT random-jitter weights (previous run used
        the pre-bugfix local checkpoint) + shuffled-smooth + single-pathway rows.
R2:     V-band degradation (physically = low-light / V-sensor degradation): does the dual
        V/NV pathway (exp29) degrade more gracefully than a single pathway (abl_arch_single)?
        This is the robustness test of the ORIGINAL architecture narrative.

All perturbations applied on the [0,1] cube BEFORE z-scoring; per-image rng seeded by index
(identical across models). No TTA. Per-image Sm dumped for paired Wilcoxon.

Run: python robustness_eval.py   -> experiments/robustness_r1/results_fix.json (+per_image)
"""
import os, sys, json
import warnings
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "models"))

import numpy as np
import torch
import torch.nn.functional as F
import py_sod_metrics as PSM

from shared.dataset import BAND_MEAN, BAND_STD, load_mask, _smooth_bands

EXP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments")
SRV = os.path.join(EXP, "abl_on_autodl")
V_B = [0, 1, 2, 3, 4]

# name, label, ckpt, build-cfg (None => default SDS)
MODELS = [
    ("exp29_sds_frod384", "no-aug(dual)", f"{EXP}/exp29_sds_frod384/best.pth", None),
    ("exp34_sds_physaug", "PGSA-full",    f"{EXP}/exp34_sds_physaug/best.pth", None),
    ("abl_random_srv",    "random(bugfix)", f"{SRV}/abl_random/best.pth", None),
    ("abl_shuffled_srv",  "shuffled",     f"{SRV}/abl_shuffled/best.pth", None),
    ("abl_single_srv",    "single-path",  f"{SRV}/abl_arch_single/best.pth", {"type": "plain", "backbone_type": "swin", "backbone": "swin_small_patch4_window7_224", "pretrained": False, "decoder": "strong"}),
]
# spectral-shift conditions (A=gain amp, B=shift amp, both wavelength-smoothed)
SPECTRAL = {"clean": (0.0, 0.0), "illum_mild": (0.10, 0.0), "illum_severe": (0.25, 0.0),
            "atmos_mild": (0.0, 0.03), "atmos_severe": (0.0, 0.08),
            "comb_mild": (0.10, 0.03), "comb_severe": (0.25, 0.08)}
# V-band degradation conditions (R2): darkening factors / mean-fill
VDEGR = {"v_dark_mild": 0.7, "v_dark_severe": 0.35, "v_mean_fill": None}


def build(cfg):
    if cfg is None:
        from sds_net import SDSNet
        return SDSNet(v_arch="swin_small_patch4_window7_224", nv_arch="swin_tiny_patch4_window7_224",
                      pretrained=False, img_size=384)
    from plain_net import PlainNet
    return PlainNet(in_bands=8, backbone_arch=cfg["backbone"], pretrained=False,
                    decoder=cfg["decoder"], backbone_type=cfg["backbone_type"], img_size=384)


def perturb(cube01, cond, img_idx):
    if cond in SPECTRAL:
        A, B = SPECTRAL[cond]
        if A == 0 and B == 0:
            return cube01
        # PER-SCENE shift: every image draws its own smoothed gain/shift (physically =
        # scene-to-scene illumination variation; a single global shift is far too easy —
        # measured -0.71 vs -2.73 no-aug degradation — and compresses all group gaps).
        rng = np.random.default_rng(img_idx * 7919 + int.from_bytes(cond.encode(), "little") % 4096)
        g = 1.0 + A * _smooth_bands(torch.from_numpy(rng.uniform(-1, 1, 8).astype(np.float32)))
        s = B * _smooth_bands(torch.from_numpy(rng.uniform(-1, 1, 8).astype(np.float32)))
        return (cube01 * g.view(8, 1, 1) + s.view(8, 1, 1)).clamp(0.0, 1.0)
    if cond in VDEGR:
        c = cube01.clone()
        f = VDEGR[cond]
        if f is None:                     # mean-fill: V bands replaced by global mean reflectance
            c[V_B] = torch.tensor(BAND_MEAN[V_B]).view(5, 1, 1)
        else:
            c[V_B] = (c[V_B] * f).clamp(0.0, 1.0)
        return c
    raise KeyError(cond)


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_dir, gt_dir = "G:/MCOD/cache352/test", "G:/MCOD/TestDataset/GT"
    names = sorted(os.path.splitext(p)[0] for p in os.listdir(cache_dir) if p.endswith(".npy"))
    mean = torch.tensor(BAND_MEAN).view(8, 1, 1); std = torch.tensor(BAND_STD).view(8, 1, 1)
    conds = list(SPECTRAL) + list(VDEGR)
    print(f"[R1fix+R2] {len(names)} imgs x {len(MODELS)} models x {len(conds)} conds", flush=True)
    gts = {n: load_mask(os.path.join(gt_dir, n + ".jpg")) for n in names}

    results, per_image = {}, {}
    for key, label, ckpt, cfg in MODELS:
        model = build(cfg).to(dev).eval()
        ck = torch.load(ckpt, map_location=dev)
        model.load_state_dict(ck["model"] if "model" in ck else ck)
        results[key] = {}; per_image[key] = {}
        for cond in conds:
            sm = PSM.Smeasure(); mae_m = PSM.MAE(); per_image[key][cond] = []
            with torch.no_grad():
                for i, n in enumerate(names):
                    cube = torch.from_numpy(np.load(os.path.join(cache_dir, n + ".npy")).astype(np.float32))
                    if cube.shape[-1] != 384:
                        cube = F.interpolate(cube[None], size=(384, 384), mode="bilinear")[0]
                    c = perturb(cube, cond, i)
                    x = ((c - mean) / std)[None].to(dev)
                    pred = torch.sigmoid(model(x)[0, 0]).cpu()
                    H, W = gts[n].shape
                    pred = F.interpolate(pred[None, None], size=(H, W), mode="bilinear")[0, 0].numpy()
                    gtb = ((gts[n] > 0.5).astype(np.uint8)) * 255
                    pu = (pred * 255).clip(0, 255).astype(np.uint8)
                    sm.step(pred=pu, gt=gtb); mae_m.step(pred=pu, gt=gtb)
                    sm_i = PSM.Smeasure(); sm_i.step(pred=pu, gt=gtb)
                    per_image[key][cond].append(float(sm_i.get_results()["sm"]))
            results[key][cond] = {"Sm": float(sm.get_results()["sm"]), "MAE": float(mae_m.get_results()["mae"])}
            print(f"[{label:16s}] {cond:15s} Sm={results[key][cond]['Sm']:.4f} MAE={results[key][cond]['MAE']:.4f}", flush=True)
        del model; torch.cuda.empty_cache()

    out = os.path.join(EXP, "robustness_r1")
    json.dump(results, open(os.path.join(out, "results_fix.json"), "w"), indent=2)
    json.dump(per_image, open(os.path.join(out, "per_image_sm.json"), "w"))

    print("\n=== Sm degradation vs clean (pp) ===")
    print(f"{'model':18s}" + "".join(f"{c:>16s}" for c in conds if c != 'clean'))
    for key, label, *_ in MODELS:
        base = results[key]["clean"]["Sm"]
        print(f"{label:18s}" + "".join(f"{(results[key][c]['Sm']-base)*100:+16.2f}" for c in conds if c != 'clean'))
    print(f"\n-> {out}/results_fix.json", flush=True)


if __name__ == "__main__":
    main()
