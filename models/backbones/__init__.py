"""Backbone category. MSBackbone (+ DoubleOpponentConv antagonistic stem) is shared
infrastructure used by BOTH papers, so it lives in shared/backbone.py and is
re-exported here for the conventional models/backbones/ layout."""
from shared.backbone import MSBackbone, DoubleOpponentConv  # noqa: F401
