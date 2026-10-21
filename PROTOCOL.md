# Unified Training & Evaluation Protocol

This document specifies the protocol under which **SDS-Net and all 12 comparison methods
were trained and evaluated identically**, so that every number in the paper is comparable.

## 1. Data split (leakage-free)

- MCOD 8-band training set split into **924 train / 103 val** scenes.
- **Model selection uses the val split only** — the test set (all 500 scenes) is evaluated
  exactly once per method after training completes. No early stopping, tuning, or
  checkpoint choice ever touches the test set.
- Per-scene degradation seeds are derived from the image index, so every model sees
  byte-identical degraded inputs.

## 2. Training recipe (identical for every method)

| Item | Value |
|---|---|
| Epochs | 300 |
| Optimizer | AdamW, lr 1.0e-4 |
| Batch size | 8 |
| Input size | 384 (with multiscale training for SDS-Net) |
| Backbone init | official pretrained weights (`pretrained: true`) |
| Loss | each method keeps its own official loss; SDS-Net uses `losses.py` |
| Spectral enhancement (SDS-Net only) | `physics_aug: true` — data-level band-perturbation augmentation |

Seeds for SDS-Net: 123 / 2 / 7 (`configs/exp34_s123.yaml`, `configs/exp40_prior_seed2.yaml`,
`configs/exp41_prior_seed7.yaml`). Reported clean numbers are three-seed mean ± std.

## 3. Clean evaluation

- Metrics via [py_sod_metrics](https://github.com/lartpang/py_sod_metrics):
  MAE, S-measure, mean/max E-measure, weighted F_β, F_β.
- Per-image S-measure for win rates is computed image-by-image, with paired
  Wilcoxon signed-rank tests for significance.

## 4. V-band degradation conditions

Applied to the visible bands only, at test time, identically for all models:

| Condition | Definition |
|---|---|
| Hard step ×0.35 | V bands scaled to 35% transmittance (sharp step) |
| Smooth τ=20 nm / τ=40 nm | smooth spectral attenuation profiles of width τ |
| Joint N×0.90 | hard step on V combined with 90% scaling on N bands |
| Sensor noise | Poisson shot noise (full-well 500 photons) + saturation with 5-bit quantization + defocus blur with shot noise |

Degradation is expressed as S-measure drop in percentage points (pp) and relative percent
(`smooth_degrad_eval.py`, `sensor_noise_eval.py`).

## 5. Comparison methods

The 12 comparison methods in the paper (MSCAF, ZoomNeXt, RUN, FIRNet, TPRNet, PRNet,
CTFNet, CODIB, GBNet, SENet, USCNet, VSCode2) were each **retrained from their official
code releases** under exactly the recipe above — identical data pipeline, input size,
training length, and evaluation scripts. Their official code belongs to the respective
authors and is not redistributed in this repository; the per-method wrappers we used only
normalized input format, band count, resolution, and output sigmoid into one interface.

## 6. Reporting rules

- Every table row states its protocol (P-clean / P-leaky).
- Capacity controls (Swin-B / Swin-S single-pathway) are trained with the same recipe and
  reported separately from monolithic 8-band adaptations.
- Cross-group robustness claims are backed by Wilcoxon tests over per-image losses,
  with p-values reported in the paper.
