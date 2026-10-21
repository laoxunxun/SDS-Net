"""Misc utilities: seeding, averaging meter, checkpoint IO."""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class AvgMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, v: float, n: int = 1):
        self.val = v
        self.sum += v * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count else 0.0


def save_ckpt(model: nn.Module, path: str, **extra):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"model": model.state_dict(), **extra}, path)


def load_ckpt(model: nn.Module, path: str, map_location: str = "cpu") -> dict:
    ck = torch.load(path, map_location=map_location)
    state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    model.load_state_dict(state)
    return ck


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
