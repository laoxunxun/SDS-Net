"""Unified experiment driver: build model from config -> train (cosine LR) -> save best
-> full eval (reliable + official pysodmetrics + vis) -> append to experiments/LOG.md.
Run: python run_experiment.py --name expNN_desc --config configs/expNN.yaml"""
import sys, os, json, math, argparse, warnings
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.abspath(__file__))  # portable: .../mhcod_code
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "models"))
import yaml, numpy as np, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import py_sod_metrics as PSM
# Reduce allocator fragmentation on small GPUs. Must precede the first CUDA alloc.
# expandable_segments is a torch>=2.1 feature — 2.0.x and 1.x hard-error on it.
_v = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
if _v >= (2, 1):
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

EXP_ROOT = os.path.join(ROOT, "experiments")


def build_model(cfg):
    cm = cfg["model"]; mt = cm.get("type", "sds")
    arch = cm.get("backbone", "resnet34"); pre = cm.get("pretrained", True)
    if mt == "plain":
        from plain_net import PlainNet
        return PlainNet(in_bands=cm.get("in_bands", 8), backbone_arch=arch, pretrained=pre,
                        decoder=cm.get("decoder", "plain"), rgb_bands=cm.get("rgb_bands"),
                        backbone_type=cm.get("backbone_type", "resnet"),
                        img_size=cfg.get("data", {}).get("size", 352),
                        grad_ckpt=cm.get("grad_ckpt", False))
    if mt == "sds":
        from sds_net import SDSNet
        return SDSNet(v_arch=cm.get("v_arch", "swin_tiny_patch4_window7_224"),
                      nv_arch=cm.get("nv_arch", "swin_tiny_patch4_window7_224"),
                      pretrained=pre, img_size=cfg.get("data", {}).get("size", 352),
                      use_bridge=cm.get("use_bridge", True),
                      use_frob=cm.get("use_frob", True),
                      physics_norm=cm.get("physics_norm", "none"),
                      use_prior=cm.get("use_prior", False),
                      grad_ckpt=cm.get("grad_ckpt", False),
                      v_pretrained=cm.get("v_pretrained", None),
                      nv_pretrained=cm.get("nv_pretrained", None))
    raise ValueError(f"Unknown model type: {mt}")

def evaluate_full(model, ds, gt_dir, preds_dir, dev):
    from shared.dataset import load_mask
    os.makedirs(preds_dir, exist_ok=True)
    n = len(ds); mae = iou = dice = fmax = 0.0
    panels = []; vis_idx = set(int(x) for x in np.linspace(0, n - 1, 8)); ths = np.linspace(0, 1, 256)
    m_m = PSM.MAE(); sm = PSM.Smeasure(); em = PSM.Emeasure(); wf = PSM.WeightedFmeasure(); fm = PSM.Fmeasure()
    model.eval()
    with torch.no_grad():
        for i in range(n):
            cube, _, name = ds[i]
            inp = cube.unsqueeze(0).to(dev)
            preds = []
            for flip_dims in [None, [2], [3], [2, 3]]:  # TTA: 4-way flip averaging
                xi = torch.flip(inp, dims=flip_dims) if flip_dims else inp
                pi = torch.sigmoid(model(xi)[0, 0]).cpu()
                if flip_dims:
                    pi = torch.flip(pi, dims=[d - 2 for d in flip_dims])
                preds.append(pi)
            pred352 = sum(preds) / len(preds)
            gt = load_mask(os.path.join(gt_dir, name + ".jpg")); gtn = gt; Hn, Wn = gt.shape
            pred = F.interpolate(pred352[None, None], size=(Hn, Wn), mode="bilinear", align_corners=False)[0, 0].numpy()
            Image.fromarray((pred * 255).clip(0, 255).astype(np.uint8)).save(os.path.join(preds_dir, name + ".png"))
            mae += float(np.abs(pred - gtn).mean()); inter = float((pred * gtn).sum()); union = pred.sum() + gtn.sum() - inter
            iou += inter / (union + 1e-8); dice += 2 * inter / (pred.sum() + gtn.sum() + 1e-8)
            pp = (pred[None] >= ths[:, None, None]); tp2 = (pp * gtn).sum((1, 2))
            P = tp2 / (pp.sum((1, 2)) + 1e-8); R = tp2 / (gtn.sum() + 1e-8)
            fmax += float(((1.3 * P * R) / (0.3 * P + R + 1e-8)).max())
            pred_u = (pred * 255).clip(0, 255).astype(np.uint8); gt_u = ((gtn > 0.5).astype(np.uint8) * 255)
            for M in (m_m, sm, em, wf, fm): M.step(pred=pred_u, gt=gt_u)
            if i in vis_idx: panels.append((name, cube, gtn, pred))
            if (i + 1) % 100 == 0: print(f"  eval {i+1}/{n}", flush=True)

    def cs(d):
        d = d if isinstance(d, dict) else {"curve": d}
        c = np.asarray(d["curve"]) if "curve" in d else np.asarray(d)
        mean = float(c.mean()) if c.ndim else float(c); mx = float(c.max()) if c.ndim else mean
        return mean, mx
    em_mean, em_max = cs(em.get_results()["em"]); fb_mean, fb_max = cs(fm.get_results()["fm"])
    off = {"MAE": m_m.get_results()["mae"], "Sm": sm.get_results()["sm"], "Em": em_mean, "Em_max": em_max,
           "wF": wf.get_results()["wfm"], "Fb": fb_mean, "Fb_max": fb_max}
    rel = {"MAE": mae / n, "IoU": iou / n, "Dice": dice / n, "maxFb": fmax / n}
    R, G, B = 4, 2, 1
    fig, axes = plt.subplots(len(panels), 4, figsize=(13, 3 * len(panels)))
    for r, (name, cube, gtn, pred) in enumerate(panels):
        c = cube[[R, G, B]].permute(1, 2, 0).numpy(); c = (c - c.min()) / (c.max() - c.min() + 1e-8)
        pd = F.interpolate(torch.from_numpy(pred)[None, None], size=c.shape[:2], mode="bilinear")[0, 0].numpy()
        gd = F.interpolate(torch.from_numpy(gtn)[None, None], size=c.shape[:2], mode="nearest")[0, 0].numpy()
        axes[r, 0].imshow(c); axes[r, 0].axis("off")
        axes[r, 1].imshow(gd, cmap="gray"); axes[r, 1].axis("off")
        axes[r, 2].imshow(pd, cmap="jet", vmin=0, vmax=1); axes[r, 2].axis("off")
        ov = c.copy(); ov[..., 0] = np.maximum(ov[..., 0], pd); axes[r, 3].imshow(ov); axes[r, 3].axis("off")
    plt.tight_layout(); plt.savefig(os.path.join(os.path.dirname(preds_dir), "vis.png"), dpi=90); plt.close()
    return off, rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True); ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f: cfg = yaml.safe_load(f)
    # Portable dataset root: set MCOD_ROOT on Linux/AutoDL instead of editing every config.
    _mcod = os.environ.get("MCOD_ROOT")
    if _mcod:
        cfg["data"]["root"] = _mcod
        if cfg["data"].get("cache"):
            cfg["data"]["cache"] = os.path.join(_mcod, os.path.basename(cfg["data"]["cache"].replace("\\", "/")))
    from shared.dataset import MCODDataset
    from shared.engine import train_one_epoch, evaluate
    from shared.metrics import evaluate_image
    from shared.utils import set_seed, get_device
    from losses import MSOLoss
    dev = get_device(); set_seed(cfg.get("seed", 42))
    out = os.path.join(EXP_ROOT, args.name); os.makedirs(out, exist_ok=True)
    cd = cfg["data"].get("cache")
    train_ds = MCODDataset(cfg["data"]["root"], "train", cfg["data"]["size"], augment=True, cache_dir=cd,
                           multiscale=cfg["data"].get("multiscale", False),
                           physics_aug=cfg["data"].get("physics_aug", False),
                           physics_aug_mode=cfg["data"].get("physics_aug_mode", "full"))
    sub = cfg["data"].get("train_subset")
    if sub:
        from torch.utils.data import Subset
        train_ds = Subset(train_ds, list(range(int(sub))))
        print(f"[subset] training on first {sub} train images", flush=True)
    test_ds = MCODDataset(cfg["data"]["root"], "test", cfg["data"]["size"], augment=False, cache_dir=cd)
    # Model selection is done on a held-out val split (deterministic tail of the train set),
    # NEVER on the test set. The test set is touched exactly once, after training, for the
    # reported metrics — the previous test-set-selection protocol leaked and inflated results.
    val_ds = MCODDataset(cfg["data"]["root"], "val", cfg["data"]["size"], augment=False, cache_dir=cd)
    # Throughput knobs for an 8GB laptop GPU: fill the DEDICATED VRAM (raise batch until
    # just below OOM) but avoid spilling into shared GPU memory — that path goes over PCIe
    # into system RAM and is slower than a smaller batch. Workers feed the GPU asynchronously.
    nw = cfg["train"].get("workers", 4)
    train_dl = DataLoader(train_ds, batch_size=cfg["train"]["batch"], shuffle=True, num_workers=nw,
                          drop_last=True, pin_memory=(dev.type == "cuda"),
                          persistent_workers=nw > 0)
    val_dl = DataLoader(val_ds, batch_size=cfg["train"].get("val_batch", 8), shuffle=False,
                        num_workers=min(nw, 2), pin_memory=(dev.type == "cuda"))
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True      # fixed-size inputs -> autotuner picks fastest convs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    model = build_model(cfg).to(dev)
    loss_fn = MSOLoss(**cfg.get("loss", {})); loss_fn.bind(model)
    epochs = cfg["train"]["epochs"]; base_lr = cfg["train"]["lr"]
    opt = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler() if dev.type == "cuda" else None  # portable: old + new torch
    ei = cfg["train"].get("eval_interval", 5); best = -1.0; best_ep = 0

    def _bn_modules(m):
        return [b for b in m.modules() if isinstance(b, torch.nn.modules.batchnorm._BatchNorm)]
    bn_snapshot = {}   # BN-stat guard: transient huge-but-finite activations can poison BN
    def snapshot_bn(): # running stats (train mode keeps working via batch stats, but eval
        for i, b in enumerate(_bn_modules(model)):   # mode goes NaN forever). Restore wins.
            bn_snapshot[i] = (b.running_mean.detach().clone(), b.running_var.detach().clone())
    def restore_bn_if_poisoned() -> bool:
        bad = any((not torch.isfinite(b.running_mean).all()) or (not torch.isfinite(b.running_var).all())
                  for b in _bn_modules(model))
        if bad and bn_snapshot:
            for i, b in enumerate(_bn_modules(model)):
                if i in bn_snapshot:
                    b.running_mean.copy_(bn_snapshot[i][0]); b.running_var.copy_(bn_snapshot[i][1])
            print(f"[{args.name}] BN running stats were non-finite -> restored from last healthy epoch", flush=True)
        return bad
    snapshot_bn()

    for ep in range(epochs):
        lr = base_lr * 0.5 * (1 + math.cos(math.pi * ep / epochs))
        for g in opt.param_groups: g["lr"] = lr
        tr = train_one_epoch(model, train_dl, opt, loss_fn, dev, scaler, ep, log_every=20,
                             clip=cfg["train"].get("clip", 0.0))
        restore_bn_if_poisoned()
        snapshot_bn()   # refresh restore point every epoch (post-restore stats are finite)
        if (ep + 1) % ei == 0 or ep == epochs - 1:
            m = evaluate(model, val_dl, dev, evaluate_image)
            print(f"[{args.name} ep{ep}] loss={tr:.4f} val-MAE={m['mae']:.4f} val-Sm={m['sm']:.4f}", flush=True)
            if m["sm"] > best:
                best = m["sm"]; best_ep = ep
                torch.save({"model": model.state_dict(), "epoch": ep}, os.path.join(out, "best.pth"))
                snapshot_bn()   # checkpoint (and its BN stats) proven healthy -> new restore point
    print(f"[{args.name}] best VAL-Sm={best:.4f}@ep{best_ep}; final test eval (single pass)...", flush=True)
    ck = torch.load(os.path.join(out, "best.pth"), map_location=dev); model.load_state_dict(ck["model"])
    gt_dir = os.path.join(cfg["data"]["root"], "TestDataset", "GT")  # portable: Win G:/MCOD or Linux /root/.../MCOD
    off, rel = evaluate_full(model, test_ds, gt_dir, os.path.join(out, "preds"), dev)
    json.dump({"name": args.name, "official": off, "reliable": rel, "best_val_Sm": float(best), "best_ep": best_ep,
               "protocol": "val-selected (deterministic 103-img tail of train); test touched once"},
              open(os.path.join(out, "metrics.json"), "w"), indent=2, default=float)
    print(f"[{args.name}] OFFICIAL: " + " ".join(f"{k}={v:.4f}" for k, v in off.items()), flush=True)
    print(f"[{args.name}] RELIABLE: " + " ".join(f"{k}={v:.4f}" for k, v in rel.items()), flush=True)
    cm = cfg["model"]
    line = (f"| {args.name} | {cm.get('type','mso')}/{cm.get('backbone','?')}/dec={cm.get('decoder','-')}/"
            f"stem={cm.get('antagonistic_stem','-')} | {off['MAE']:.4f} | {off['Sm']:.4f} | "
            f"{off['Em']:.4f}/{off['Em_max']:.4f} | {off['wF']:.4f} | {off['Fb']:.4f}/{off['Fb_max']:.4f} | "
            f"{rel['IoU']:.4f} | {rel['Dice']:.4f} | |\n")
    with open(os.path.join(EXP_ROOT, "LOG.md"), "a", encoding="utf-8") as f: f.write(line)
    print(f"[{args.name}] appended LOG.md", flush=True)


if __name__ == "__main__":
    main()
