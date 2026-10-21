"""Em calibration via logits temperature — tuned on VAL (legitimate), applied to TEST once.

Motivation: exp37 beats every competitor on Sm/wF/Fb/IoU but trails PRNet on Em
(0.910 vs 0.926) while Em_max is 0.940 — a probability-calibration gap, not an
alignment-capability gap. Scaling logits by 1/T sharpens/flattens the probability
curve and directly reshapes the threshold-integrated Em.

Protocol: sweep T on the 103-image held-out VAL split -> apply argmax-Em T to the test
set exactly once -> report deltas for exp37 (and exp35 as reference).

Run: python em_calibration.py
"""
import os, sys, json
import warnings
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "models")); sys.path.insert(0, ROOT)

import numpy as np
import torch
import torch.nn.functional as F
import py_sod_metrics as PSM

from sds_net import SDSNet
from shared.dataset import MCODDataset, load_mask

EXP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments")
TEMPS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 1.75, 2.0, 2.5]


def build():
    m = SDSNet(v_arch="swin_small_patch4_window7_224", nv_arch="swin_tiny_patch4_window7_224",
               pretrained=False, img_size=384, use_prior=True)
    return m


def build_baseline():
    return SDSNet(v_arch="swin_small_patch4_window7_224", nv_arch="swin_tiny_patch4_window7_224",
                  pretrained=False, img_size=384, use_prior=False)


@torch.no_grad()
def logits_cache(model, ds, dev):
    """One forward pass per image; cache logits + GT (native-res)."""
    L, G = [], []
    model.eval()
    for i in range(len(ds)):
        cube, _, _ = ds[i]
        lg = model(cube.unsqueeze(0).to(dev))[0, 0].cpu()
        name = ds.names[i]
        gt = load_mask(os.path.join("G:/MCOD", ds.split == "test" and "TestDataset" or "TrainDataset", "GT", name + ".jpg"))
        L.append(lg); G.append(gt)
    return L, G


def em_at_temp(L, G, T, use_tta=False):
    em = PSM.Emeasure(); sm = PSM.Smeasure(); mae = PSM.MAE(); wf = PSM.WeightedFmeasure()
    for lg, gt in zip(L, G):
        pred = torch.sigmoid(lg / T).numpy()
        H, W = gt.shape
        pred = F.interpolate(torch.from_numpy(pred)[None, None], size=(H, W), mode="bilinear")[0, 0].numpy()
        gtb = ((gt > 0.5).astype(np.uint8)) * 255
        pu = (pred * 255).clip(0, 255).astype(np.uint8)
        for M in (em, sm, mae, wf):
            M.step(pred=pu, gt=gtb)
    r = em.get_results()["em"]; c = np.asarray(r["curve"]) if isinstance(r, dict) else np.asarray(r)
    return {"Em": float(c.mean()), "Em_max": float(c.max()), "Sm": float(sm.get_results()["sm"]),
            "MAE": float(mae.get_results()["mae"]), "wF": float(wf.get_results()["wfm"])}


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_ds = MCODDataset("G:/MCOD", "val", 384, augment=False, cache_dir="G:/MCOD/cache352")
    test_ds = MCODDataset("G:/MCOD", "test", 384, augment=False, cache_dir="G:/MCOD/cache352")
    out = {}
    for name, builder, ckpt in [("exp37_prior", build, f"{EXP}/exp37_prior/best.pth"),
                                ("exp35_valbaseline", build_baseline, f"{EXP}/exp35_valbaseline/best.pth")]:
        model = builder().to(dev)
        ck = torch.load(ckpt, map_location=dev)
        model.load_state_dict(ck["model"] if "model" in ck else ck)
        Lv, Gv = logits_cache(model, val_ds, dev)
        Lt, Gt = logits_cache(model, test_ds, dev)
        curve = {T: em_at_temp(Lv, Gv, T)["Em"] for T in TEMPS}
        best_T = max(curve, key=curve.get)
        base_test = em_at_temp(Lt, Gt, 1.0)
        cal_test = em_at_temp(Lt, Gt, best_T)
        out[name] = {"val_em_curve": {str(k): v for k, v in curve.items()}, "best_T": best_T,
                     "test_T1.0": base_test, "test_Tbest": cal_test}
        print(f"[{name}] best T (val) = {best_T} | TEST Em: {base_test['Em']:.4f} -> {cal_test['Em']:.4f} "
              f"(max {base_test['Em_max']:.4f} -> {cal_test['Em_max']:.4f}) | Sm {base_test['Sm']:.4f} -> {cal_test['Sm']:.4f}", flush=True)
        del model; torch.cuda.empty_cache()
    p = os.path.join(EXP, "em_calibration.json")
    json.dump(out, open(p, "w"), indent=2)
    print("->", p)


if __name__ == "__main__":
    main()
