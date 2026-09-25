"""LoRC -- Low-Rank Collapse in Semantic Residuals (arXiv:2608.20882v1) on MIDS / EVAL_SPACE.

Imports are deliberately lazy: `python -c "import lorc.data"` must work without transformers or a
GPU, so that the manifest/label tooling and the smoke test can run in any environment while only
model.py needs the DINOv3-capable interpreter.
"""
__all__ = ["data", "diagnostics", "engine", "get_label", "losses", "metrics", "model"]
