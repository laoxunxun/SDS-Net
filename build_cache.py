"""One-time: pre-decode + resize all MCOD cubes to a fast .npy cache.

Run from project root:
    python build_cache.py
Takes ~3-5 min once; cuts every subsequent training run from ~8h to ~2-3h.
"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))   # mhcod_code
sys.path.insert(0, ROOT)

from shared.dataset import build_cache

if __name__ == "__main__":
    root = "G:/MCOD"
    size = 352
    cache_dir = os.path.join(root, f"cache{size}")
    print(f"building cache -> {cache_dir} (size={size}) ...")
    build_cache(root, size=size, cache_dir=cache_dir)
    print("done.")
