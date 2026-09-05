per_model_or.json uses a v3 fusion mode ('OR' over per-model thresholds) that paas/fusion.py
does not implement -- FusionCfg supports 'mean' and 'weighted' only. Kept for reference; it
will NOT load. Implement the mode in paas/fusion.py before moving it back.
