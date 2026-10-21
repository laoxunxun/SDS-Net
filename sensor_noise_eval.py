"""Sensor-level degradation: photon shot + read noise, saturation+quantization, blur+shot
on the V bands only (visible-sensor fault scenario). Same protocol as Table IV:
no retraining, deterministic per-image seeds, paired clean baseline.

Models: ours(dual), single-path Swin-B control, CGFNet (dual), 6 monolithic competitors.
Run: E:/anaconda/envs/mcod/python.exe sensor_noise_eval.py   (~40 min, inference only)
"""
import os, sys, json
import warnings
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "models"));
sys.path.insert(0, os.path.dirname(ROOT))   # shared 包在上一级

import numpy as np
import torch
import torch.nn.functional as F
import py_sod_metrics as PSM

from shared.dataset import BAND_MEAN, BAND_STD, load_mask

EXP = os.path.join(ROOT, "experiments")
MODELS = [
    ("SDS-Net(dual)", f"{EXP}/exp34_sds_physaug/best.pth"),
    ("single-SwinB",  f"{EXP}/abl_singleB_clean/best.pth"),
    ("CGFNet",        f"{EXP}/cmp_CGFNet_clean/best.pth"),
    ("MSCAF",         f"{EXP}/cmp_MSCAF_clean/best.pth"),
    ("ZoomNeXt",      f"{EXP}/cmp_ZoomNeXt_clean/best.pth"),
    ("PRNet",         f"{EXP}/cmp_PRNet_clean/best.pth"),
    ("RUN",           f"{EXP}/cmp_RUN_clean/best.pth"),
    ("FIRNet",        f"{EXP}/cmp_FIRNet_clean/best.pth"),
    ("TPRNet",        f"{EXP}/cmp_TPRNet_clean/best.pth"),
]
N_PHOT_SHOT, SIG_READ_SHOT = 500, 0.008      # 条件1: 弱光光子噪声
SAT_CLIP, QUANT_LV = 0.85, 32                # 条件2: 饱和削顶 + 5bit量化
BLUR_SIG, N_PHOT_BLUR, SIG_READ_BLUR = 1.2, 2000, 0.01   # 条件3: 镜头污染模糊+噪声
CONDS = ["clean", "shot_read", "sat_quant", "blur_shot"]

def gblur(x, sigma):
    """可分离高斯模糊, x [C,H,W]"""
    k = int(np.ceil(sigma * 3)) * 2 + 1
    r = torch.arange(k, dtype=x.dtype, device=x.device) - k // 2
    g = torch.exp(-(r ** 2) / (2 * sigma * sigma)); g = g / g.sum()
    x = F.conv2d(x[None], g[None, None, :].expand(x.shape[0], 1, 1, k), padding=(0, k // 2), groups=x.shape[0])[0]
    x = F.conv2d(x[None], g[None, :, None].expand(x.shape[0], 1, k, 1), padding=(k // 2, 0), groups=x.shape[0])[0]
    return x

def sensor_perturb(cube01, cond, idx, dev):
    """cube01: [8,H,W] in [0,1]; 只扰动 V 组(前5波段)"""
    gen = torch.Generator(device=dev).manual_seed(90210 + idx)
    out = cube01.clone()
    V = out[:5]
    if cond == "shot_read":
        lam = (V * N_PHOT_SHOT).clamp(min=1e-6)
        Vn = torch.poisson(lam, generator=gen) / N_PHOT_SHOT
        Vn = Vn + torch.randn(V.shape, generator=gen, device=dev) * SIG_READ_SHOT
        out[:5] = Vn.clamp(0, 1)
    elif cond == "sat_quant":
        Vn = V.clamp(max=SAT_CLIP)
        Vn = torch.round(Vn * (QUANT_LV - 1)) / (QUANT_LV - 1)
        out[:5] = Vn
    elif cond == "blur_shot":
        Vn = gblur(V, BLUR_SIG)
        lam = (Vn * N_PHOT_BLUR).clamp(min=1e-6)
        Vn = torch.poisson(lam, generator=gen) / N_PHOT_BLUR
        Vn = Vn + torch.randn(V.shape, generator=gen, device=dev) * SIG_READ_BLUR
        out[:5] = Vn.clamp(0, 1)
    return out

def build(name):
    if name == "single-SwinB":
        from models.plain_net import PlainNet
        return PlainNet(in_bands=8, backbone_arch="swin_base_patch4_window7_224",
                        pretrained=False, decoder="strong", backbone_type="swin", img_size=384)
    sys.path.insert(0, os.path.join(ROOT, "compare"))
    from compare_adapter import CompareWrapper
    return CompareWrapper(name, pretrained=False, img_size=384)

def _sds():
    from sds_net import SDSNet
    return SDSNet(v_arch="swin_small_patch4_window7_224", nv_arch="swin_tiny_patch4_window7_224",
                  pretrained=False, img_size=384)

def main():
    dev = torch.device("cuda")
    cache_dir, gt_dir = "G:/MCOD/cache352/test", "G:/MCOD/TestDataset/GT"
    names = sorted(os.path.splitext(p)[0] for p in os.listdir(cache_dir) if p.endswith(".npy"))
    mean = torch.tensor(BAND_MEAN).view(8, 1, 1).to(dev); std = torch.tensor(BAND_STD).view(8, 1, 1).to(dev)
    gts = {n: load_mask(os.path.join(gt_dir, n + ".jpg")) for n in names}
    results, per_image = {}, {}
    for key, ckpt in MODELS:
        model = (_sds() if key == "SDS-Net(dual)" else build(key)).to(dev).eval()
        ck = torch.load(ckpt, map_location=dev)
        model.load_state_dict(ck["model"] if "model" in ck else ck)
        results[key], per_image[key] = {}, {}
        for cond in CONDS:
            sm = PSM.Smeasure()
            per_image[key][cond] = []
            with torch.no_grad():
                for i, n in enumerate(names):
                    cube = torch.from_numpy(np.load(os.path.join(cache_dir, n + ".npy")).astype(np.float32))
                    if cube.shape[-1] != 384:
                        cube = F.interpolate(cube[None], size=(384, 384), mode="bilinear")[0]
                    c = sensor_perturb(cube.to(dev), cond, i, dev) if cond != "clean" else cube.to(dev)
                    x = (c - mean) / std
                    pred = torch.sigmoid(model(x[None])[0, 0]).cpu()
                    H, W = gts[n].shape
                    pred = F.interpolate(pred[None, None], size=(H, W), mode="bilinear")[0, 0].numpy()
                    pu = (pred * 255).clip(0, 255).astype(np.uint8)
                    g = ((gts[n] > 0.5).astype(np.uint8)) * 255
                    sm.step(pred=pu, gt=g)
                    sm_i = PSM.Smeasure(); sm_i.step(pred=pu, gt=g)
                    per_image[key][cond].append(float(sm_i.get_results()["sm"]))
            results[key][cond] = float(sm.get_results()["sm"])
            print(f"[{key:14s}] {cond:10s} Sm={results[key][cond]:.4f}", flush=True)
        del model; torch.cuda.empty_cache()
    print("\n=== Sm drop vs clean (pp) ===")
    for key, _ in MODELS:
        b = results[key]["clean"]
        print(f"{key:14s}" + "".join(f"{c}: {(results[key][c]-b)*100:+.2f}  " for c in CONDS if c != "clean"))
    os.makedirs(os.path.join(EXP, "sensor_noise"), exist_ok=True)
    json.dump(results, open(os.path.join(EXP, "sensor_noise", "results.json"), "w"), indent=2)
    json.dump(per_image, open(os.path.join(EXP, "sensor_noise", "per_image_sm.json"), "w"))
    print("saved -> experiments/sensor_noise/results.json")

if __name__ == "__main__":
    main()
