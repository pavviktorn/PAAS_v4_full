"""Environment + path setup for PAAS_ensemble_v2.

Everything in this project is designed to run on the GLOBAL interpreter
the ONE project venv (transformers>=5 + vLLM): /datasets/work/vLLM/temp/PAAS_qwen3vl/venv/bin/python
Import this module (and call :func:`setup`) before importing any FFAA or 9-class code so that
(a) the vendored ``ffaa/`` and ``ensemble9/`` trees are importable, (b) the CUDA device is
chosen *before* the FFAA modules pin ``CUDA_VISIBLE_DEVICES``, and (c) HF stays offline/quiet.
"""
from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FFAA_DIR = os.path.join(PROJECT_ROOT, "ffaa")
ENSEMBLE9_DIR = os.path.join(PROJECT_ROOT, "ensemble9")

# Bundled assets (self-contained; see README "Layout").
BASE_CLIP = os.path.join(PROJECT_ROOT, "base_models", "clip-vit-large-patch14-336")
BASE_T5 = os.path.join(PROJECT_ROOT, "base_models", "t5-base")
# v4 FFAA = Qwen3.5-4B (merged, vLLM) + FROM-SCRATCH MIDS 4-class head (not axon0 warm-start).
QWEN_DIR = os.path.join(PROJECT_ROOT, "weights", "qwen35_4b_merged")
MIDS_PATH = os.path.join(PROJECT_ROOT, "weights", "ffaa_qwen35_mids", "best.pth")
PROMPT_FILE = os.path.join(PROJECT_ROOT, "ffaa", "playground", "prompts.txt")
ENSEMBLE9_CONFIG = os.path.join(PROJECT_ROOT, "config", "ensemble9.json")

# GSD (Exp 10) and SeLop/LROR (Exp 11) - the two CLIP detectors added in v3. Both reuse the shared
# BASE_CLIP backbone above (the CLIP vision weights are identical across all members).
GSD_CKPT = os.path.join(PROJECT_ROOT, "weights", "gsd", "best.pt")
SELOP_CKPT = os.path.join(PROJECT_ROOT, "weights", "selop", "best.pt")
# gsdA = the SAME GSD mechanism with its inference reference basis built from a bounded TRAINSET
# slice instead of from the eval split. It gets its own slot rather than overwriting GSD_CKPT so both
# arms can be served at once and compared; picking one is then a config edit, not a file swap.
GSDA_CKPT = os.path.join(PROJECT_ROOT, "weights", "gsdA", "best.pt")
# PE-SPC: the trained prototype head, plus the FROZEN encoder it was trained against. The encoder is
# named explicitly because the head's dim is tied to it -- a mismatched pair fails the strict
# state_dict load rather than scoring silently wrong.
PESPC_CKPT = os.path.join(PROJECT_ROOT, "weights", "pespc", "best.pt")
PESPC_ENCODER = os.path.join(PROJECT_ROOT, "base_models", "PE-Core-G14-448", "PE-Core-G14-448.pt")
PESPC_MODEL_NAME = "PE-Core-G14-448"

# DINOv3-SPC: the self-supervised member -- frozen DINOv3 ViT-7B/16 @384 + a 98,308-param
# prototype head. The only member with NO language supervision anywhere in its encoder, which is the
# point: the other six are all language-supervised contrastive, so this one decorrelates rather than
# duplicates.
#
# The checkpoint's own cfg carries dim=8192, size=384 and the encoder basename, and the model
# wrapper checks them at load time, so serving cannot silently run at a different resolution or
# against a different encoder than the head was trained on. DINOSPC_SIZE below is only the DEFAULT
# for feature extraction during TRAINING; inference takes the size from the checkpoint.
DINOSPC_CKPT = os.path.join(PROJECT_ROOT, "weights", "dinospc", "best.pt")
DINOSPC_SIZE = 384
# BUNDLED, with no external fallback. The encoder is 26 GB -- bigger than the rest of this
# project's weights combined -- so the tempting design is to reference a shared copy. That is
# exactly what scripts/verify_standalone.py exists to reject: a shared absolute path resolves on the
# machine it was written on and nowhere else, which makes `du -sh` look self-contained while the
# project silently depends on a directory outside it. This project is standalone for weights, and
# that property is checked, so the encoder lives in base_models/ like every other base model.
#
# Absence is an error raised at the point of use, not a silent fallback. Training or serving WITHOUT
# DINOv3-SPC is legitimate and needs no encoder: drop `dinospc` from fusion.components, or set
# RUN_DINOSPC=0 for the training stage.
DINOSPC_ENCODER = os.path.join(PROJECT_ROOT, "base_models", "dinov3-vit7b16-pretrain-lvd1689m")


# LoRC: frozen DINOv3 ViT-H+/16 @384 + LoRA r=16 on q/k/v/o + a rank-32 low-rank-attention head
# (arXiv 2608.20882v1). 5.4M trainable parameters in total.
#
# WHY IT IS DEPLOYED. On the certified report split it is the second-best single member in the set
# (24 errors, behind only its own 7B sibling at 22 and ahead of a3_9c at 36), and adding it takes
# the fusion from 22 errors to 18 -- the only significant improvement an exhaustive search over all
# 2,047 subsets of 11 members could find. Its 7B sibling reaches the same 18 and is redundant given
# this one, at 7.6x the training cost and 5.3x the inference latency, so only H+/16 is served.
#
# The encoder is the H+/16 backbone, 3.2 GB -- about 8x smaller than DINOSPC's 26 GB, so bundling
# it is affordable where a second 7B encoder would not be. Size, crop mode, LoRA shape and the
# register-token offset all come from the checkpoint and are cross-checked against the encoder
# config at load time.
LORC_CKPT = os.path.join(PROJECT_ROOT, "weights", "lorc", "best.pt")
LORC_ENCODER = os.path.join(PROJECT_ROOT, "base_models", "dinov3-vith16plus-pretrain-lvd1689m")


def require_lorc_encoder() -> str:
    """Return the bundled DINOv3 H+/16 encoder dir, or explain precisely what is missing."""
    if os.path.isdir(LORC_ENCODER):
        return LORC_ENCODER
    raise SystemExit(
        f"[paas] LoRC was requested but its encoder is not bundled:\n"
        f"        expected {LORC_ENCODER}\n"
        f"[paas] Restore that directory, or run without the member -- RUN_LORC=0 for training, or\n"
        f"       drop `lorc` from fusion.components for serving.")


def require_dinospc_encoder() -> str:
    """Return the bundled DINOv3 encoder dir, or explain precisely what is missing."""
    if os.path.isdir(DINOSPC_ENCODER):
        return DINOSPC_ENCODER
    raise SystemExit(
        f"[paas] DINOv3-SPC was requested but its encoder is not bundled:\n"
        f"        expected {DINOSPC_ENCODER}\n"
        f"[paas] Restore that directory (26 GB: 6 safetensors shards + config), or run without the\n"
        f"       member -- RUN_DINOSPC=0 for training, or drop `dinospc` from fusion.components\n"
        f"       for serving.")


def setup(device: str | None = None, quiet: bool = True) -> str:
    """Make the vendored trees importable and pin the CUDA device.

    ``device`` like ``"cuda:0"`` / ``"cuda:2"`` / ``"cpu"``. When a cuda index is given we set
    ``CUDA_VISIBLE_DEVICES`` to that physical index and the process then sees it as ``cuda:0`` --
    this is what lets FFAA's ``models.py`` (which calls ``setdefault('CUDA_VISIBLE_DEVICES','0')``)
    land on the device we want, and is the basis for the multi-GPU file-sharding scripts.

    RETURNS the device string to hand torch AFTER that pinning, which is NOT always the one passed
    in. Masking renumbers the GPUs: with ``device="cuda:2"`` we set ``CUDA_VISIBLE_DEVICES=2`` and
    the selected GPU is then the process's ONLY visible one, i.e. ``cuda:0``; ``cuda:2`` at that
    point is an invalid ordinal. Callers that passed ``cfg.device`` straight to their models made
    every non-zero ``--device`` fail. When the caller had already pinned the mask themselves we did
    not renumber anything, so their ordinal is returned unchanged.
    """
    eff = device or "cuda:0"
    if device and device.startswith("cuda:"):
        idx = device.split(":", 1)[1]
        pre = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", idx)
        if pre is None:
            eff = "cuda:0"                      # we masked -> the chosen GPU is now ordinal 0
        else:
            # The mask was already set (e.g. `CUDA_VISIBLE_DEVICES=1 run_server.sh` with
            # device=cuda:0). Their ordinal indexes into that mask; catch the out-of-range case
            # here with a readable message instead of a CUDA "invalid device ordinal" much later.
            nvis = len([x for x in pre.split(",") if x != ""])
            if idx.isdigit() and nvis and int(idx) >= nvis:
                raise SystemExit(
                    f"[paas] device={device!r} but CUDA_VISIBLE_DEVICES={pre!r} exposes only "
                    f"{nvis} GPU(s) (valid: cuda:0..cuda:{nvis - 1}). Inside a masked process the "
                    f"ordinal indexes the MASK, not the physical GPU. Either unset "
                    f"CUDA_VISIBLE_DEVICES and use --device {device}, or keep the mask and use "
                    f"--device cuda:0.")
    if quiet:
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    for d in (FFAA_DIR, ENSEMBLE9_DIR, PROJECT_ROOT):   # PROJECT_ROOT -> `import gsd` / `import selop`
        if d not in sys.path:
            sys.path.insert(0, d)
    return eff


def visible_device() -> str:
    """The device string to hand torch *after* CUDA_VISIBLE_DEVICES has been pinned (always cuda:0
    when a single physical GPU was selected)."""
    try:
        import torch
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"

# ---- ONE-VENV GUARD ----------------------------------------------------------------
# Every stage (training AND inference) runs on the single project venv:
#   /datasets/work/vLLM/temp/PAAS_qwen3vl/venv/bin/python
# The tf5 CLIP ports (flattened .vision_model, tensor-returning CLIPEncoderLayer) and the in-process
# vLLM both require transformers>=5, so silently running on the legacy transformers==4.37 interpreter
# would either crash deep in a model load or, worse, train something that cannot be served.
def _require_project_venv():
    import sys
    try:
        import transformers
        major = int(transformers.__version__.split(".")[0])
    except Exception as e:                      # transformers missing entirely -> wrong interpreter
        raise SystemExit(f"[venv] cannot import transformers ({e}).\n"
                         f"[venv] run everything with: /datasets/work/vLLM/temp/PAAS_qwen3vl/venv/bin/python")
    if major < 5:
        raise SystemExit(
            f"[venv] transformers {transformers.__version__} at {sys.executable} -- this project needs >=5.\n"
            f"[venv] run everything (train AND inference) with: /datasets/work/vLLM/temp/PAAS_qwen3vl/venv/bin/python")


# ---- ISOLATION GUARD ---------------------------------------------------------------
# The venv must be SELF-CONTAINED: no ~/.local, no /usr/local. ~/.local carries a complete
# legacy stack (transformers 4.37.2, peft 0.7.1, tokenizers 0.15.2, accelerate 0.21.0) that
# is inert only because the venv sorts earlier on sys.path -- so an isolation slip would
# silently serve on tf4. Enforced by pyvenv.cfg include-system-site-packages=false plus
# PYTHONNOUSERSITE=1. Mirrors train/_bootstrap.py so BOTH deploy and train paths check.
_VENV_PREFIX = "/datasets/work/vLLM/temp/PAAS_qwen3vl/venv"


def _require_isolated_site():
    import sys
    stray = [p for p in sys.path
             if ("site-packages" in p or "dist-packages" in p) and not p.startswith(_VENV_PREFIX)]
    if stray:
        raise SystemExit(
            "[venv] NON-VENV package paths on sys.path -- the environment is not isolated:\n"
            + "".join(f"        {p}\n" for p in stray)
            + f"[venv] expected only {_VENV_PREFIX}/lib*/python3.12/site-packages.\n"
              "[venv] fix: PYTHONNOUSERSITE=1 and include-system-site-packages=false in pyvenv.cfg.")


_require_project_venv()
_require_isolated_site()
