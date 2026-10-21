"""Generic train / eval loop shared by both papers.

Assumes:
- model(x) returns logits (B,1,H,W)
- loss_fn(logits, mask) returns a scalar
- masks are float in {0,1}, shape (B,1,H,W)
"""
from __future__ import annotations

import torch

from .utils import AvgMeter


def train_one_epoch(model, loader, opt, loss_fn, device, scaler=None,
                    epoch: int = 0, log_every: int = 50, clip: float = 0.0) -> float:
    model.train()
    meter = AvgMeter()
    use_amp = scaler is not None
    skipped = 0
    for i, (x, y, _) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        if use_amp:
            with torch.cuda.amp.autocast():
                logits = model(x)
                loss = loss_fn(logits, y)
            if not torch.isfinite(loss):
                # one overflowing batch (AMP fp16 spikes in unfolding/div ops) must not
                # poison the weights for the rest of training — skip it entirely
                skipped += 1
                continue
            scaler.scale(loss).backward()
            if clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)
            scaler.step(opt)
            scaler.update()
        else:
            logits = model(x)
            loss = loss_fn(logits, y)
            if not torch.isfinite(loss):
                skipped += 1
                continue
            loss.backward()
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)
            opt.step()
        meter.update(loss.item(), x.size(0))
        if (i + 1) % log_every == 0:
            print(f"[epoch {epoch} | {i+1}/{len(loader)}] loss={meter.avg:.4f}"
                  + (f" (skipped {skipped} nan batches)" if skipped else ""))
    return meter.avg


@torch.no_grad()
def evaluate(model, loader, device, metrics_fn) -> dict:
    """metrics_fn(pred, gt) -> dict of metric->value (2D inputs, [0,1])."""
    from .metrics import mean_metrics
    model.eval()
    all_metrics = []
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = torch.sigmoid(logits)
        pred = torch.nn.functional.interpolate(
            pred, size=y.shape[-2:], mode="bilinear", align_corners=False)
        for b in range(pred.shape[0]):
            all_metrics.append(metrics_fn(pred[b, 0], y[b, 0]))
    return mean_metrics(all_metrics)
