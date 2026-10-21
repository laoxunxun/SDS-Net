"""表8推理时间统一复测：同一环境(RTX 4070 Laptop, batch 1, 384×384, 无TTA, fp32)。

Run: /e/anaconda/envs/mmcv/python.exe bench_timing.py
"""
import os, sys, time, json
import warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "models"))

import torch

EXP = os.path.join(HERE, "experiments")

def build(key):
    if key == "SDS-Net":
        from sds_net import SDSNet
        m = SDSNet(v_arch="swin_small_patch4_window7_224", nv_arch="swin_tiny_patch4_window7_224",
                   pretrained=False, img_size=384)
        ck = os.path.join(EXP, "exp34_sds_physaug", "best.pth")
    elif key == "single-SwinB":
        from plain_net import PlainNet
        m = PlainNet(in_bands=8, backbone_arch="swin_base_patch4_window7_224", pretrained=False,
                     decoder="strong", backbone_type="swin", img_size=384)
        ck = os.path.join(EXP, "abl_singleB_clean", "best.pth")
    else:
        sys.path.insert(0, os.path.join(HERE, "compare"))
        from compare_adapter import CompareWrapper
        m = CompareWrapper(key, pretrained=False, img_size=384)
        ckmap = {"MSCAF": "cmp_MSCAF_clean", "ZoomNeXt": "cmp_ZoomNeXt_clean", "FIRNet": "cmp_FIRNet_clean",
                 "TPRNet": "cmp_TPRNet_clean", "CGFNet": "cmp_CGFNet_clean", "PRNet": "cmp_PRNet_clean"}
        ck = os.path.join(EXP, ckmap[key], "best.pth")
    sd = torch.load(ck, map_location="cpu")
    m.load_state_dict(sd["model"] if isinstance(sd, dict) and "model" in sd else sd)
    return m.cuda().eval()

dev = torch.device("cuda")
x = torch.randn(1, 8, 384, 384, device=dev)
out = {}
print(f"GPU: {torch.cuda.get_device_name(0)}")
for key in ["SDS-Net", "MSCAF", "ZoomNeXt", "FIRNet", "TPRNet", "CGFNet", "PRNet", "single-SwinB"]:
    try:
        m = build(key)
        with torch.no_grad():
            for _ in range(5):
                m(x)
            torch.cuda.synchronize(); t0 = time.time()
            for _ in range(30):
                m(x)
            torch.cuda.synchronize()
        ms = (time.time() - t0) / 30 * 1000
        n_par = sum(p.numel() for p in (m.net if hasattr(m, "net") else m).parameters()) / 1e6
        out[key] = {"ms": round(ms, 1), "params_M": round(n_par, 1)}
        print(f"{key:14s} {ms:6.1f} ms/img  {n_par:5.1f}M", flush=True)
        del m; torch.cuda.empty_cache()
    except Exception as e:
        print(f"{key:14s} FAILED: {type(e).__name__}: {e}", flush=True)
json.dump(out, open(os.path.join(EXP, "bench_timing.json"), "w"), indent=2)
print("saved -> experiments/bench_timing.json")
