"""MCOD multispectral camouflaged object detection dataloader (with fast cache).

Verified facts (2026-07, local copy at G:/MCOD):
- 8-band cubes in .mat (HDF5 v7.3, key 'gray', shape (8,1170,860) -> transpose
  (0,2,1) to (8,860,1170)). GT/ = .jpg (lossy -> threshold), Pcolor/ = .png
  (extensions swapped). Train 1027 / Test 500.
- DATA-LOADING BOTTLENECK: opening h5py per sample + resize is slow (~220ms/sample),
  leaving the GPU idle. build_cache() pre-decodes+resizes all cubes once to float16
  .npy; MCODDataset loads the cache when available (~10x faster).

MCOD bands (0-based): 0 Violet422,1 Blue487,2 Green550,3 Orange602,4 Red660,
5 RedEdge725,6 NIR1_785,7 NIR2_887.
"""
from __future__ import annotations

import os

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

# Per-band z-score stats on [0,1], recomputed over the full train set (1027 imgs).
BAND_MEAN = np.array([0.4237, 0.5409, 0.5392, 0.4953, 0.4725, 0.4876, 0.5263, 0.4840], dtype=np.float32)
BAND_STD = np.array([0.2217, 0.2297, 0.2195, 0.2202, 0.2581, 0.2191, 0.2025, 0.2241], dtype=np.float32)

_SUB = {"train": "TrainDataset", "test": "TestDataset"}

# Deterministic val split: last VAL_N images of the SORTED train names.
# Fixed indices, not a random seed, so every experiment uses the identical split.
VAL_N = 103  # 10% of 1027 train images


def _smooth_bands(x: torch.Tensor) -> torch.Tensor:
    """3-tap moving average across the 8 bands (reflect-padded).

    PHYSICS: illumination spectra and atmospheric effects vary SMOOTHLY with
    wavelength — neighboring bands are strongly correlated. Smoothing the random
    per-band gain/shift makes the augmentation physically plausible (vs independent
    per-band noise, which is unphysical)."""
    xp = torch.cat([x[:1], x, x[-1:]])                  # reflect-pad to length 10
    return (xp[:-2] + xp[1:-1] + xp[2:]) / 3.0


def physics_spectral_augment(cube: torch.Tensor, mode: str = "full") -> torch.Tensor:
    """Physics-guided spectral augmentation on an 8-band [0,1] reflectance cube.

    Ablation `mode`:
      full     = illumination(gain) + atmospheric(shift), both wavelength-smoothed  [PGSA, default]
      illum    = illumination gain only (smoothed)
      atmos    = atmospheric shift only (smoothed)
      random   = gain + shift but NOT smoothed (independent per-band = unphysics control)
      shuffled = gain + shift, smoothed but in a SHUFFLED band order (smoothness without
                 spectral structure — isolates whether the *spectral order* of the smoothness
                 is the active ingredient, vs generic smoothing of 8 numbers)
      none     = passthrough (no aug)
    """
    if mode == "none":
        return cube
    smooth = mode != "random"
    gain = (1.0 + (torch.rand(8) * 2.0 - 1.0) * 0.15) if mode in ("full", "illum", "shuffled", "random") else torch.ones(8)
    shift = ((torch.rand(8) * 2.0 - 1.0) * 0.03) if mode in ("full", "atmos", "shuffled", "random") else torch.zeros(8)
    if smooth:
        if mode == "shuffled":
            # smooth along a random permutation of bands, then map back to true band order.
            # The gain/shift applied to each true band is now a smoothed function of bands that
            # are NOT its spectral neighbors -> "smoothness" holds but "spectral structure" is broken.
            perm = torch.randperm(8)
            inv = torch.argsort(perm)
            gain = _smooth_bands(gain[perm])[inv]
            shift = _smooth_bands(shift[perm])[inv]
        else:
            gain = _smooth_bands(gain); shift = _smooth_bands(shift)
    return (cube * gain.view(8, 1, 1) + shift.view(8, 1, 1)).clamp(0.0, 1.0)



def load_cube(mat_path: str) -> np.ndarray:
    """Load 8-band cube as float32 (C,H,W) in [0,1], MATLAB transpose fixed."""
    import h5py  # lazy: lets cache-only workflows run without h5py installed
    with h5py.File(mat_path, "r") as f:
        gray = np.asarray(f["gray"], dtype=np.float32)      # (8,1170,860) == (C,W,H) due to v7.3
    cube = np.transpose(gray, (0, 2, 1))                     # -> (C=8, H=860, W=1170)
    return cube / 255.0


def load_mask(gt_path: str) -> np.ndarray:
    arr = np.asarray(Image.open(gt_path).convert("L"), dtype=np.float32) / 255.0
    return (arr > 0.5).astype(np.float32)


def build_cache(root: str, size: int = 352, cache_dir: str | None = None,
                splits=("train", "test")) -> str:
    """Pre-decode + resize all cubes to `size`, save as float16 .npy (one-time).
    Removes the h5py+resize bottleneck at train time."""
    if cache_dir is None:
        cache_dir = os.path.join(root, f"cache{size}")
    for split in splits:
        mat_dir = os.path.join(root, _SUB[split], "Mat")
        out_dir = os.path.join(cache_dir, split)
        os.makedirs(out_dir, exist_ok=True)
        names = sorted(p for p in os.listdir(mat_dir) if p.endswith(".mat"))
        for i, p in enumerate(names):
            out = os.path.join(out_dir, os.path.splitext(p)[0] + ".npy")
            if os.path.exists(out):
                continue
            cube = load_cube(os.path.join(mat_dir, p))               # (8,H,W) [0,1]
            t = torch.from_numpy(cube).unsqueeze(0)
            t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)[0]
            np.save(out, t.numpy().astype(np.float16))
            if (i + 1) % 200 == 0:
                print(f"  [{split}] {i+1}/{len(names)}")
        print(f"[cache] {split}: {len(names)} cubes -> {out_dir}")
    return cache_dir


def compute_band_stats(root: str, sample_n: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    acc = np.zeros(8, dtype=np.float64); acc2 = np.zeros(8, dtype=np.float64); n = 0
    mat_dir = os.path.join(root, "TrainDataset", "Mat")
    for p in sorted(p for p in os.listdir(mat_dir) if p.endswith(".mat"))[:sample_n]:
        c = load_cube(os.path.join(mat_dir, p))
        acc += c.sum(axis=(1, 2)); acc2 += (c * c).sum(axis=(1, 2)); n += c.shape[1] * c.shape[2]
    mean = acc / n; std = np.sqrt(acc2 / n - mean * mean)
    return mean.astype(np.float32), std.astype(np.float32)


class MCODDataset(Dataset):
    """split: 'train' | 'test'. Each item: cube (8,size,size) z-scored, mask (1,size,size),
    name. Loads pre-decoded cache if `cache_dir` is given and built."""

    def __init__(self, root: str, split: str = "train", size: int = 352,
                 augment: bool = False, stats: tuple | None = None, cache_dir: str | None = None,
                 multiscale: bool = False, physics_aug: bool = False, physics_aug_mode: str = "full"):
        # 'val' = held-out tail of the sorted train names (deterministic, same for every run).
        assert split in ("train", "val", "test")
        src = "train" if split == "val" else split
        self.split = split
        self.mat_dir = os.path.join(root, _SUB[src], "Mat")
        self.gt_dir = os.path.join(root, _SUB[src], "GT")
        self.names = sorted(os.path.splitext(p)[0] for p in os.listdir(self.mat_dir) if p.endswith(".mat"))
        if split == "train":
            self.names = self.names[:-VAL_N]
        elif split == "val":
            self.names = self.names[-VAL_N:]
        self.size = size
        self.augment = augment
        self.multiscale = multiscale
        self.physics_aug = physics_aug
        self.physics_aug_mode = physics_aug_mode
        mean, std = stats if stats else (BAND_MEAN, BAND_STD)
        self.mean = torch.tensor(mean).view(8, 1, 1)
        self.std = torch.tensor(std).view(8, 1, 1)

        self.cache_paths = None
        if cache_dir:
            cp = [os.path.join(cache_dir, src, n + ".npy") for n in self.names]
            if all(os.path.exists(c) for c in cp):
                self.cache_paths = cp

    def __len__(self) -> int:
        return len(self.names)

    def _cube(self, idx: int) -> torch.Tensor:    # -> (8,size,size) in [0,1]
        if self.cache_paths is not None:
            cube = torch.from_numpy(np.load(self.cache_paths[idx]).astype(np.float32))
            if cube.shape[-1] != self.size:    # cache may be at different size → resize
                cube = F.interpolate(cube.unsqueeze(0), size=(self.size, self.size),
                                      mode="bilinear", align_corners=False)[0]
            return cube
        cube = torch.from_numpy(load_cube(os.path.join(self.mat_dir, self.names[idx] + ".mat")))
        return F.interpolate(cube.unsqueeze(0), size=(self.size, self.size),
                             mode="bilinear", align_corners=False)[0]

    def __getitem__(self, idx: int):
        n = self.names[idx]
        cube = self._cube(idx)                                          # (8,size,size) [0,1]
        mask = torch.from_numpy(load_mask(os.path.join(self.gt_dir, n + ".jpg")))
        mask = F.interpolate(mask[None, None], size=(self.size, self.size), mode="nearest")[0, 0]

        if self.augment:
            if torch.rand(1).item() < 0.5:
                cube = torch.flip(cube, dims=[2]); mask = torch.flip(mask, dims=[1])
            if torch.rand(1).item() < 0.5:
                cube = torch.flip(cube, dims=[1]); mask = torch.flip(mask, dims=[0])
            k = int(torch.randint(0, 4, (1,)).item())
            if k:
                cube = torch.rot90(cube, k, dims=[1, 2]); mask = torch.rot90(mask, k, dims=[0, 1])
            if self.multiscale:
                scale = 0.75 + torch.rand(1).item() * 0.75
                sh = max(1, int(round(self.size * scale)))
                sw = max(1, int(round(self.size * scale)))
                cube = F.interpolate(cube.unsqueeze(0), size=(sh, sw), mode="bilinear", align_corners=False)[0]
                mask = F.interpolate(mask[None, None], size=(sh, sw), mode="nearest")[0, 0]
                if sh >= self.size:
                    top = int(torch.randint(0, sh - self.size + 1, (1,)).item())
                    left = int(torch.randint(0, sw - self.size + 1, (1,)).item())
                    cube = cube[:, top:top + self.size, left:left + self.size]
                    mask = mask[top:top + self.size, left:left + self.size]
                else:
                    cube = F.pad(cube, (0, self.size - sw, 0, self.size - sh))
                    mask = F.pad(mask, (0, self.size - sw, 0, self.size - sh))

        if self.augment and self.physics_aug:
            cube = physics_spectral_augment(cube, self.physics_aug_mode)                  # physics-guided spectral aug (on [0,1])

        cube = (cube - self.mean) / self.std
        return cube, mask.unsqueeze(0).clamp(0, 1), n
