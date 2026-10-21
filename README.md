# SDS-Net

**Physics-Guided Multispectral Camouflaged Object Detection with Degradation Robustness**

This is the official PyTorch implementation of SDS-Net, a dual-pathway architecture for
multispectral camouflaged object detection (MCOD) that pairs **clean accuracy** with
**degradation robustness** on the same footing.

> SDS-Net separates the visible (V) and non-visible (N) band groups into two Swin encoders
> of asymmetric capacity, fuses them at every scale through a bidirectional, mutually gated
> cross-spectral bridge, and injects physical indices via zero-initialized FiLM gating that
> is the identity at initialization and bounded in contribution.

## News

- **[2026-10]** Code and unified evaluation protocol released. Weights will be published in
  the [Releases](../../releases) section upon paper acceptance.

## Highlights

| Property | Result |
|---|---|
| Clean S-measure (3-seed, leakage-free protocol) | **0.8648 ± 0.0019** — ahead of all 12 retrained 2021–2026 methods |
| V-band degradation loss (V/N-separated dual) | **3.5 – 4.7 pp** |
| V-band degradation loss (12 monolithic 8-band adaptations) | 6.6 – 19.1 pp |
| Group separation | Cross-group Wilcoxon p < 10⁻⁴⁰, zero overlap |
| Capacity-matched single-pathway control | gap narrows to ≈ 2× — separation, not parameters, drives robustness |

### Clean comparison (unified protocol, top rows)

| Method | Venue | Protocol | MAE ↓ | Sm ↑ |
|---|---|---|---|---|
| **SDS-Net (ours)** | – | P-clean | **0.0023** | **0.8648 ± 0.0019** |
| ZoomNeXt | TPAMI'24 | P-clean | 0.0025 | 0.8614 |
| MSCAF | TCSVT'23 | P-clean | 0.0026 | 0.8588 |
| CTFNet | CVIU'25 | P-clean | 0.0025 | 0.8552 |
| CODIB | TMM'25 | P-clean | 0.0025 | 0.8530 |

Full tables (12 retrained methods, Em/F_β/wFTE, per-condition degradation, win rates,
latency) are in the paper. All 12 comparison methods were **retrained from their official
code under the identical protocol** — see [PROTOCOL.md](PROTOCOL.md).

## Installation

```bash
conda create -n sdsnet python=3.10 -y
conda activate sdsnet
pip install -r requirements.txt
```

Swin backbones are created through [timm](https://github.com/huggingface/pytorch-image-models);
pretrained weights download automatically on first run.

## Data

Download the MCOD benchmark (8-band multispectral camouflaged scenes) from the official
release of the benchmark authors: **https://github.com/hu-xh/PRNet** (dataset download
links are in that repository's README). Arrange the data as:

```
<MCOD_ROOT>/
├── TrainDataset/
│   ├── Mat/     # 8-band .mat cubes
│   └── GT/      # binary masks
└── TestDataset/
    ├── Mat/
    └── GT/
```

Set `data.root` in the config files to `<MCOD_ROOT>`, then build the fast cache:

```bash
python build_cache.py --root <MCOD_ROOT> --size 352
```

## Training

SDS-Net uses the leakage-free P-clean protocol (300 epochs, model selection never touches
the test set — details in [PROTOCOL.md](PROTOCOL.md)):

```bash
python run_experiment.py --name sds_s123 --config configs/exp34_s123.yaml
```

The three paper seeds are `configs/exp34_s123.yaml`, `configs/exp40_prior_seed2.yaml`, and
`configs/exp41_prior_seed7.yaml`. Ablation configs (`abl_*.yaml`, `exp*.yaml`) live in
[configs/](configs/).

## Evaluation

```bash
# Degradation robustness: hard-step / smooth / joint V-N conditions
python smooth_degrad_eval.py

# Sensor-level stress test (Poisson shot noise, saturation, defocus)
python sensor_noise_eval.py

# Per-image S-measure, win rates and Wilcoxon significance
python winrate_eval.py

# Architecture robustness table / stratified (area, N-contrast) analysis
python robustness_eval.py
python stratified_eval.py
```

## Model Zoo

| Weights | Config | Sm | Link |
|---|---|---|---|
| SDS-Net (seed 123) | `configs/exp34_s123.yaml` | 0.8670 | [release, upon acceptance] |
| SDS-Net (seed 2) | `configs/exp40_prior_seed2.yaml` | – | [release, upon acceptance] |
| SDS-Net (seed 7) | `configs/exp41_prior_seed7.yaml` | – | [release, upon acceptance] |
| SDS-Net (3-seed set) | – | 0.8648 ± 0.0019 | above, all three |

Place downloaded `best.pth` files under `experiments/<exp_name>/best.pth`
(e.g. `experiments/exp34_s123/best.pth`).

## Repository Layout

```
SDS-Net/
├── run_experiment.py     # training / evaluation entry point
├── losses.py             # multi-term loss (IoU / edge / structure)
├── models/               # SDS-Net, bridge, decoders, backbones, ablation variants
├── shared/               # dataset (MCOD 8-band loader), engine, metrics, utils
├── configs/              # paper configs: main seeds + ablations
├── build_cache.py        # MCOD fast-cache builder
├── smooth_degrad_eval.py # degradation evaluation (hard / smooth / joint)
├── sensor_noise_eval.py  # sensor-level stress test
├── winrate_eval.py       # per-image win rates + Wilcoxon tests
├── robustness_eval.py    # architecture robustness table
├── stratified_eval.py    # stratified analysis by target area / N-contrast
├── sig_test.py           # significance tests
├── em_calibration.py     # logit-temperature E-measure calibration
├── aggregate_seeds.py    # aggregate multi-seed results
└── bench_timing.py       # latency benchmark
```

## Citation

If you find this work useful, please cite:

```bibtex
@article{sdsnet2026,
  title   = {Physics-Guided Multispectral Camouflaged Object Detection with Degradation Robustness},
  author  = {Sun, Haoran and Qu, Jue and Wang, Wei and Liu, Haiping and Zhou, Hengfei and Yang, Bo},
  journal = {IEEE Transactions on Multimedia},
  year    = {2026}
}
```

## License

Released under the [MIT License](LICENSE). The MCOD dataset belongs to its original
authors and must be obtained from their official release.

## Acknowledgements

We thank the MCOD benchmark authors, the [timm](https://github.com/huggingface/pytorch-image-models)
team, the [py_sod_metrics](https://github.com/lartpang/py_sod_metrics) maintainers, and the
authors of all comparison methods whose official code made the unified retraining protocol
possible.
