"""E2 (M2 修复): 平滑退化剖面重评 + V/N 联合退化敏感性 —— 仅推理, 不训练.

背景: 原 Table 3 的 V-退化是硬阶跃 (V 组 ×0.35, N 组 ×1.0), 与式(3)"光照谱随波长平滑
变化"的物理断言自相矛盾。本脚本用跨红边的平滑 sigmoid 剖面重评全部已有检查点:

  f(lam) = f_min + (f_max - f_min) * sigmoid((lam - lam0) / tau),  lam0=700nm (红边中心)

条件集 (全部在 [0,1] cube 上、z-score 之前施加; 无 TTA; 逐图确定性):
  clean            无扰动 (同时为分层评估导出逐图 Sm)
  v_hard_sev       硬阶跃 ×0.35 (仅 V 组) —— 复现原 Table 3 口径, 供对照
  v_sig_sev        平滑剖面 tau=20nm, f_min=0.35, f_max=1.0
  v_sig_wide       平滑剖面 tau=40nm (更缓), 同 f_min/f_max —— 剖面形状敏感性
  v_sig_n09        平滑剖面 + N 组整体 ×0.90 (N 组轻微衰减的联合退化)
  v_sig_n085       平滑剖面 + N 组整体 ×0.85

每个 (model, cond) 导出 500 张逐图 Sm -> paired Wilcoxon (vs 各自 clean, 以及
dual 组代表 vs 整体式组代表的跨模型检验)。输出:
  experiments/smooth_degrad/results.json      汇总均值 + p 值 + 相对下降
  experiments/smooth_degrad/per_image_sm.json 逐图值 (供本地再分析/分层/作图)

用法 (服务器):  MCOD_ROOT=/root/autodl-tmp/MCOD python smooth_degrad_eval.py [--only name1,name2]
用法 (本地4070): python smooth_degrad_eval.py --only exp34 (等本地有权重的模型)
"""
import os, sys, json, argparse
import warnings
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "models"))
sys.path.insert(0, HERE)

import numpy as np
import torch
import torch.nn.functional as F
import py_sod_metrics as PSM

from shared.dataset import BAND_MEAN, BAND_STD, load_mask

EXP = os.path.join(HERE, "experiments")
SRV = os.path.join(EXP, "abl_on_autodl")   # server-trained ablation checkpoints (also valid locally if copied)

# ---- MCOD 8 波段中心波长 (nm), 见 MCOD README / arXiv:2509.15753 -------------------
CWL = np.array([422.5, 487.5, 550.0, 602.5, 660.0, 725.0, 785.0, 887.5], dtype=np.float64)
V_B = [0, 1, 2, 3, 4]
N_B = [5, 6, 7]
LAM0, F_MIN, F_MAX = 700.0, 0.35, 1.0

def sigmoid_profile(tau):
    return F_MIN + (F_MAX - F_MIN) / (1.0 + np.exp(-(CWL - LAM0) / tau))

PROF20, PROF40 = sigmoid_profile(20.0), sigmoid_profile(40.0)

CONDS = ["clean", "v_hard_sev", "v_sig_sev", "v_sig_wide", "v_sig_n09", "v_sig_n085"]

def perturb(cube01, cond):
    if cond == "clean":
        return cube01
    if cond == "v_hard_sev":                       # 原 Table 3 口径 (复现用)
        c = cube01.clone(); c[V_B] = (c[V_B] * 0.35).clamp(0, 1); return c
    if cond == "v_sig_sev":
        f = torch.tensor(PROF20, dtype=torch.float32).view(8, 1, 1); return (cube01 * f).clamp(0, 1)
    if cond == "v_sig_wide":
        f = torch.tensor(PROF40, dtype=torch.float32).view(8, 1, 1); return (cube01 * f).clamp(0, 1)
    if cond in ("v_sig_n09", "v_sig_n085"):
        g = 0.90 if cond == "v_sig_n09" else 0.85
        f = PROF20.copy(); f[N_B] *= g            # N 组额外轻微衰减
        f = torch.tensor(f, dtype=torch.float32).view(8, 1, 1); return (cube01 * f).clamp(0, 1)
    raise KeyError(cond)

# ---- 模型清单: (key, group, ckpt 候选路径列表, build 配置) --------------------------
# group: dual=模态分离双流 / mono=整体式单流; 候选路径按序探测, 缺失则跳过并警告。
_S = lambda *ps: [os.path.join(EXP, *p) for p in ps]
MODELS = [
    ("SDS-Net(dual)",   "dual",  _S(("exp34_sds_physaug", "best.pth"), ("exp37_prior", "best.pth")), None),
    ("single-SwinS",    "mono",  _S(("abl_on_autodl", "abl_arch_single", "best.pth"), ("abl_arch_single", "best.pth"),
                                   ("abl_single_clean", "best.pth")),
     {"type": "plain", "backbone_type": "swin", "backbone": "swin_small_patch4_window7_224", "pretrained": False, "decoder": "strong"}),
    ("single-SwinB",    "mono",  _S(("abl_singleB_clean", "best.pth")),
     {"type": "plain", "backbone_type": "swin", "backbone": "swin_base_patch4_window7_224", "pretrained": False, "decoder": "strong"}),
    ("CGFNet",          "dual",  _S(("cmp_CGFNet", "best.pth"), ("cmp_CGFNet_clean", "best.pth")), "CGFNet"),
    ("PRNet",           "mono",  _S(("cmp_PRNet_clean", "best.pth")), "PRNet"),
    ("MSCAF",           "mono",  _S(("cmp_MSCAF_clean", "best.pth"), ("cmp_MSCAF", "best.pth")), "MSCAF"),
    ("ZoomNeXt",        "mono",  _S(("cmp_ZoomNeXt_clean", "best.pth"), ("cmp_ZoomNeXt", "best.pth")), "ZoomNeXt"),
    ("RUN",             "mono",  _S(("cmp_RUN", "best.pth"), ("cmp_RUN_clean", "best.pth"), ("cmp_RUN_fixed2", "best.pth")), "RUN"),
    ("FIRNet",          "mono",  _S(("cmp_FIRNet", "best.pth"), ("cmp_FIRNet_clean", "best.pth")), "FIRNet"),
    ("TPRNet",          "mono",  _S(("cmp_TPRNet", "best.pth"), ("cmp_TPRNet_clean", "best.pth")), "TPRNet"),
]


def build(cfg):
    if cfg is None:
        from sds_net import SDSNet
        return SDSNet(v_arch="swin_small_patch4_window7_224", nv_arch="swin_tiny_patch4_window7_224",
                      pretrained=False, img_size=384)
    if isinstance(cfg, dict):
        from plain_net import PlainNet
        return PlainNet(in_bands=8, backbone_arch=cfg["backbone"], pretrained=False,
                        decoder=cfg["decoder"], backbone_type=cfg["backbone_type"], img_size=384)
    sys.path.insert(0, os.path.join(HERE, "compare"))
    from compare_adapter import CompareWrapper
    return CompareWrapper(cfg, pretrained=False, img_size=384)


def find_ckpt(cands):
    for p in cands:
        if os.path.exists(p):
            return p
    return None


def wilcoxon_p(a, b):
    try:
        from scipy.stats import wilcoxon
        d = np.asarray(a) - np.asarray(b)
        if np.allclose(d, 0):
            return 1.0
        return float(wilcoxon(d)[1])
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="逗号分隔的模型 key 子集, 如: exp34,PRNet")
    args = ap.parse_args()
    only = set(args.only.split(",")) if args.only else None

    root = os.environ.get("MCOD_ROOT", "G:/MCOD")
    cache_dir = os.path.join(root, "cache352", "test")
    gt_dir = os.path.join(root, "TestDataset", "GT")
    names = sorted(os.path.splitext(p)[0] for p in os.listdir(cache_dir) if p.endswith(".npy"))
    mean = torch.tensor(BAND_MEAN).view(8, 1, 1); std = torch.tensor(BAND_STD).view(8, 1, 1)
    gts = {n: load_mask(os.path.join(gt_dir, n + ".jpg")) for n in names}
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    todo = []
    for key, grp, cands, cfg in MODELS:
        if only and key not in only:
            continue
        ck = find_ckpt(cands)
        if ck is None:
            print(f"[skip] {key}: no checkpoint found under {cands}", flush=True)
            continue
        todo.append((key, grp, ck, cfg))
    print(f"[smooth-degrad] {len(names)} imgs x {len(todo)} models x {len(CONDS)} conds", flush=True)
    print(f"[smooth-degrad] profile tau=20: {np.round(PROF20, 3).tolist()}", flush=True)
    print(f"[smooth-degrad] profile tau=40: {np.round(PROF40, 3).tolist()}", flush=True)

    results, per_image = {}, {}
    out = os.path.join(EXP, "smooth_degrad")
    os.makedirs(out, exist_ok=True)

    def _save_partial(finished_only=True):
        json.dump(results, open(os.path.join(out, "results.json"), "w"), indent=2)
        json.dump(per_image, open(os.path.join(out, "per_image_sm.json"), "w"))

    for key, grp, ckpt, cfg in todo:
        try:
            model = build(cfg).to(dev).eval()
            ck = torch.load(ckpt, map_location=dev)
            model.load_state_dict(ck["model"] if isinstance(ck, dict) and "model" in ck else ck)
        except Exception as e:
            print(f"[skip] {key}: BUILD FAILED ({type(e).__name__}: {e}) — continuing", flush=True)
            continue
        results[key] = {"group": grp}; per_image[key] = {}
        for cond in CONDS:
            sm = PSM.Smeasure(); mae_m = PSM.MAE(); per_image[key][cond] = []
            with torch.no_grad():
                for i, n in enumerate(names):
                    cube = torch.from_numpy(np.load(os.path.join(cache_dir, n + ".npy")).astype(np.float32))
                    if cube.shape[-1] != 384:
                        cube = F.interpolate(cube[None], size=(384, 384), mode="bilinear")[0]
                    c = perturb(cube, cond)
                    x = ((c - mean) / std)[None].to(dev)
                    pred = torch.sigmoid(model(x)[0, 0]).cpu()
                    H, W = gts[n].shape
                    pred = F.interpolate(pred[None, None], size=(H, W), mode="bilinear")[0, 0].numpy()
                    gtb = ((gts[n] > 0.5).astype(np.uint8)) * 255
                    pu = (pred * 255).clip(0, 255).astype(np.uint8)
                    sm.step(pred=pu, gt=gtb); mae_m.step(pred=pu, gt=gtb)
                    sm_i = PSM.Smeasure(); sm_i.step(pred=pu, gt=gtb)
                    per_image[key][cond].append(float(sm_i.get_results()["sm"]))
            base = results[key].get("clean", {}).get("Sm")
            cur = float(sm.get_results()["sm"])
            d_pp = None if base is None else round((cur - base) * 100, 2)
            d_rel = None if base is None else round((cur - base) / base * 100, 2)
            p = None if base is None else wilcoxon_p(per_image[key][cond], per_image[key]["clean"])
            results[key][cond] = {"Sm": cur, "MAE": float(mae_m.get_results()["mae"]),
                                  "dSm_pp": d_pp, "dSm_rel_pct": d_rel, "wilcoxon_vs_clean_p": p}
            print(f"[{key:16s}|{grp}] {cond:11s} Sm={cur:.4f} "
                  f"({'' if d_pp is None else f'{d_pp:+.2f}pp / {d_rel:+.1f}%'}"
                  f"{'' if p is None else f', p={p:.2e}'})", flush=True)
        del model; torch.cuda.empty_cache()
        _save_partial()   # incremental: a later model crash never loses earlier results

    # 跨组核心检验: dual 组代表 vs mono 组代表 在主条件下的逐图对比
    dual_keys = [k for k, g, *_ in todo if g == "dual" and k in per_image]
    mono_keys = [k for k, g, *_ in todo if g == "mono" and k in per_image]
    if dual_keys and mono_keys:
        d0, m0 = dual_keys[0], mono_keys[0]
        for cond in CONDS[1:]:
            p = wilcoxon_p(per_image[d0][cond], per_image[m0][cond])
            print(f"[cross] {cond}: {d0} vs {m0} paired Wilcoxon p={p}", flush=True)
            results.setdefault("_cross", {})[f"{d0}_vs_{m0}@{cond}"] = p

    out = os.path.join(EXP, "smooth_degrad")
    os.makedirs(out, exist_ok=True)
    json.dump(results, open(os.path.join(out, "results.json"), "w"), indent=2)
    json.dump(per_image, open(os.path.join(out, "per_image_sm.json"), "w"))
    print(f"[done] -> {out}/results.json + per_image_sm.json  (回传这两个文件即可)", flush=True)


if __name__ == "__main__":
    main()
