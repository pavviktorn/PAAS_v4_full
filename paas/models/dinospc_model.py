"""Wrapper around DINO-SPC (prototype head on a frozen self-supervised DINOv3/DINOv2 encoder).

WHY THIS MEMBER EXISTS. Every other member is language-supervised contrastive: the five
CLIP-ViT-L/14-336 arms (A1/A2/A3 9-class, GSD, SeLop) and PE-Core-G14-448 (PE-SPC). Two results
motivate a self-supervised encoder instead of a seventh language-supervised one:

  * frozen GSD scored bin_auc 0.9562 against 0.9993 for the same head over trainable CLIP-L
    features -- CLIP-L's frozen features do not linearly separate forgery types;
  * PE-SPC scores 0.998864 from a 15,364-parameter head on a FROZEN encoder -- so a frozen encoder
    is sufficient when the encoder itself is strong.

DINO is trained with no language objective at all, so it retains the low-level texture and
frequency statistics that language-contrastive pretraining discards -- which is where presentation
attacks and deepfake artifacts live. It is therefore the member most likely to DECORRELATE from the
existing six rather than duplicate them. The deployed encoder is DINOv3 ViT-7B/16 (LVD-1689M).

THREE THINGS THAT DIFFER FROM PE-SPC and are easy to get wrong:

  * READOUT IS CLS || mean(patch tokens), L2-normalised (8192-d for ViT-7B/16). There is no
    projection head and no text tower, so there is no single "image embedding" to take; the concat
    is the standard strong readout for dense/texture tasks. Getting only the CLS token would
    silently halve the feature and fail the strict head load.
  * REGISTER TOKENS. DINOv3's sequence is [CLS, reg x num_register_tokens, patches] -- 4 registers
    for ViT-7B/16 -- while DINOv2 has none. `tokens[:, 1:]` is correct for DINOv2 and WRONG for
    DINOv3: it folds the registers into the patch mean and still yields plausible features and a
    clean load. The offset is therefore read from the model config, never assumed.
  * PREPROCESSING IS A SQUASH RESIZE with ImageNet mean/std -- NOT PE's transform and NOT a 336
    centre crop. A centre crop would discard the frame borders where presentation-attack evidence
    lives (bezels, moire, print edges). DINOv3 is RoPE-based, so resolution is a free choice and the
    serving size must come from the checkpoint (below).
  * PROTOTYPES WERE INITIALISED FROM PER-CLASS K-MEANS on the trainset features, not from CLIP text
    prompts -- there is no text tower to embed. The prototypes are baked into the checkpoint; the
    init method matters only for reproducing training.

Class order is FIXED PROJECT-WIDE as 0=real, 1=pad, 2=deepfake.
"""
from __future__ import annotations

import os
import sys
from typing import List, Optional

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REAL = 0
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)     # DINOv2 uses ImageNet stats


class DinoSPCModel:
    name = "dinospc"

    def __init__(self, ckpt_path: str, encoder_path: str, device: str = "cuda:0",
                 amp_dtype: str = "bf16", size: int = 0):
        for d in (_ROOT, os.path.join(_ROOT, "pespc")):
            if d not in sys.path:
                sys.path.insert(0, d)
        import torch
        import torchvision.transforms as T
        from transformers import AutoModel
        from head import build_head

        self.torch = torch
        self.device = device
        ck = torch.load(ckpt_path, map_location="cpu")
        cfg = ck["cfg"]
        # dim must match the encoder+readout that produced the training features. A 1280-d PE head
        # loaded here, or a CLS-only 1536-d readout against this 3072-d head, fails on the strict
        # state_dict load rather than silently producing garbage scores.
        self.head = build_head(cfg)
        self.head.load_state_dict(ck["state"])
        self.head = self.head.to(device).eval()

        self.amp = {"bf16": torch.bfloat16, "fp16": torch.float16,
                    "fp32": torch.float32}.get(amp_dtype, torch.bfloat16)
        self.enc = AutoModel.from_pretrained(encoder_path, dtype=self.amp).to(device).eval()
        # patch tokens start after CLS + register tokens; DINOv2 reports 0 registers, DINOv3 4.
        self.n_register = int(getattr(self.enc.config, "num_register_tokens", 0) or 0)
        self.patch0 = 1 + self.n_register
        # Size comes from the CHECKPOINT, not from the caller: serving at a different resolution
        # than the head was trained at is a train/serve skew that produces plausible-looking but
        # wrong scores. An explicit `size` argument overrides only for deliberate experiments.
        self.image_size = int(size or cfg.get("size") or self.enc.config.image_size)
        self.tfm = T.Compose([
            T.Resize((self.image_size, self.image_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(), T.Normalize(MEAN, STD)])
        self.cfg = cfg

    def score_frames(self, rgb_list: List[np.ndarray], batch_size: int = 8) -> List[dict]:
        """rgb_list: HxWx3 uint8 arrays. Returns one dict per frame (input order):
        {"fake": float|None, "error": str|None}."""
        torch = self.torch
        from PIL import Image
        out: List[Optional[dict]] = [None] * len(rgb_list)
        with torch.no_grad():
            for i in range(0, len(rgb_list), batch_size):
                chunk = rgb_list[i:i + batch_size]
                try:
                    x = torch.stack([self.tfm(Image.fromarray(r)) for r in chunk]).to(self.device)
                    o = self.enc(pixel_values=x.to(self.amp)).last_hidden_state
                    f = torch.cat([o[:, 0], o[:, self.patch0:].mean(1)], -1)
                    f = torch.nn.functional.normalize(f.float(), dim=-1)
                    p = torch.softmax(self.head(f), dim=-1)[:, REAL].float().cpu().numpy()
                    for j, v in enumerate(p):
                        out[i + j] = {"fake": float(1.0 - v), "error": None}
                except Exception as exc:                                # noqa: BLE001
                    for j in range(len(chunk)):
                        out[i + j] = {"fake": None, "error": repr(exc)}
        return out
