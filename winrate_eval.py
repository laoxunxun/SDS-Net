# -*- coding: utf-8 -*-
"""Fixed inference: exact eval_2026_local structure + NaN guard. Recomputes per-image Sm
for GBNet/SENet/USCNet/VSCode2, rewrites the 6 new winrate rows correctly."""
import os, sys, json, io
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import warnings; warnings.filterwarnings("ignore")
import py_sod_metrics as PSM
from scipy.stats import wilcoxon

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import os
ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(ROOT)); sys.path.insert(0, os.path.join(ROOT, 'compare'))
from shared.dataset import load_mask, BAND_MEAN, BAND_STD
from compare_adapter import CompareWrapper

EXP = os.path.join(ROOT, 'experiments')
CACHE = 'G:/MCOD/cache352/test'
GT_DIR = 'G:/MCOD/TestDataset/GT'
names = sorted(os.path.splitext(p)[0] for p in os.listdir(CACHE) if p.endswith('.npy'))
gts = {n: load_mask(os.path.join(GT_DIR, n + '.jpg')) for n in names}
dev = torch.device('cuda')
mean = torch.tensor(BAND_MEAN).view(8, 1, 1).to(dev)
std = torch.tensor(BAND_STD).view(8, 1, 1).to(dev)

def run_model(key, size):
    model = CompareWrapper(key, pretrained=False, img_size=size).to(dev).eval()
    ck = torch.load(os.path.join(EXP, 'cmp_%s_clean' % key, 'best.pth'), map_location=dev)
    model.load_state_dict(ck.get('model', ck.get('state_dict', ck)))
    outdir = os.path.join(EXP, 'cmp_%s_clean' % key, 'preds')
    os.makedirs(outdir, exist_ok=True)
    arr = []
    n_nan = 0
    for n in names:
        cube = torch.from_numpy(np.load(os.path.join(CACHE, n + '.npy')).astype(np.float32))
        if cube.shape[-1] != size:
            cube = F.interpolate(cube[None], size=(size, size), mode='bilinear')[0]
        x = ((cube.to(dev) - mean) / std)[None]
        with torch.no_grad():
            out = model(x)
        pred = out[0] if isinstance(out, (tuple, list)) else out
        if pred.dim() == 4:
            pred = pred[0, 0]
        pred = torch.sigmoid(pred).cpu()
        if torch.isnan(pred).any():
            n_nan += 1
        H, W = gts[n].shape
        pred = F.interpolate(pred[None, None], size=(H, W), mode='bilinear')[0, 0].numpy()
        pu = (pred * 255).clip(0, 255).astype(np.uint8)
        pu = np.nan_to_num(pu, nan=0).astype(np.uint8)
        Image.fromarray(pu, mode='L').save(os.path.join(outdir, n + '.png'))
        g = ((gts[n] > 0.5).astype(np.uint8)) * 255
        s = PSM.Smeasure()
        s.step(pred=pu, gt=g)
        arr.append(float(s.get_results()['sm']))
    del model
    torch.cuda.empty_cache()
    print('%s: mean Sm %.4f  (nan images: %d)' % (key, np.mean(arr), n_nan), flush=True)
    return np.array(arr)

sm_new = {}
sm_new['CTFNet'] = run_model('CTFNet', 384)
sm_new['CODIB'] = run_model('CODIB', 384)
sm_new['GBNet'] = run_model('GBNet', 384)
sm_new['SENet'] = run_model('SENet', 384)
sm_new['USCNet'] = run_model('USCNet', 352)
sm_new['VSCode2'] = run_model('VSCode2', 352)

pi = json.load(open(os.path.join(EXP, 'smooth_degrad', 'per_image_sm.json')))
ours = np.array(pi['SDS-Net(dual)']['clean'])
print('ours mean %.4f n=%d' % (ours.mean(), len(ours)))

rows = []
for m in ['CTFNet', 'CODIB', 'GBNet', 'SENet', 'USCNet', 'VSCode2']:
    diff = ours - sm_new[m]
    wr = float((diff > 0).mean() * 100)
    try:
        p = float(wilcoxon(ours, sm_new[m])[1])
    except Exception:
        p = 1.0
    rows.append((m, round(wr, 1), p))
    print('%s: winrate %.1f%%  p=%.2e' % (m, wr, p))

csv_path = os.path.join(ROOT, 'paper', 'matlab_figs', 'fig13_winrate', 'data_winrate.csv')
lines = [l for l in open(csv_path, encoding='utf-8-sig').read().splitlines()
         if l and not any(l.startswith(m + ',') for m in
                          ['CTFNet', 'CODIB', 'GBNet', 'SENet', 'USCNet', 'VSCode2'])]
for m, wr, p in rows:
    lines.append('%s,%.1f,%.1e,comp' % (m, wr, p))
open(csv_path, 'w', encoding='utf-8-sig').write('\r\n'.join(lines) + '\r\n')
print('csv rewritten')
json.dump({m: v.tolist() for m, v in sm_new.items()},
          open(os.path.join(ROOT, 'paper', 'figs_pool', '图6_定性对比素材', '新方法_全量逐图Sm.json'), 'w'))
print('per-image Sm saved')
