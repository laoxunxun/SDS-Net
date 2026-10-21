"""Shared infrastructure for the physics-guided multispectral COD project.

Used by both paper packages: mso-net (Paper 1) and pcsu-net (Paper 2).
- dataset.py  : MCOD 8-band dataloader (working)
- metrics.py  : COD evaluation metrics (MAE / IoU / Dice / weighted-F / S-measure / E-measure)
- backbone.py : multi-scale encoder with 8-channel input patch
- engine.py   : generic train / eval loop
- utils.py    : seed, AvgMeter, checkpoint IO
"""
