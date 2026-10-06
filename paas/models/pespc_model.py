"""Wrapper around PE-SPC (Semantic Prototype Calibration on a frozen Perception Encoder).

FROZEN PE-Core-G14-448 + a C*k x 1280 prototype head. Only the prototypes were trained (15,364
parameters for the deployed H3/k=4 recipe); the 1.88B-parameter encoder is untouched, and the text
tower is not loaded at all at inference -- the prototypes were placed once, offline, and are baked
into the checkpoint.

Exposes the uniform ``score_frames`` the PAAS pipeline expects: per frame the fake-score
``1 - softmax(logits)[REAL]``, identical in meaning to every other member's ``fake``.

TWO THINGS THAT DIFFER FROM THE OTHER CLIP MEMBERS and are easy to get wrong:

  * PREPROCESSING IS A SQUASH RESIZE TO 448, NOT A 336 CENTRE CROP. PE's own transform keeps the
    whole frame; a centre crop would discard the frame borders where presentation-attack evidence
    lives (bezels, moire, print edges), and would also be a train/serve skew because the encoder was
    pretrained on the squash. This class builds the transform from the ENCODER's own image_size
    rather than accepting one, so it cannot silently diverge from training.
  * COST. The encoder is 1.88B params at 448px -- about 47.8 img/s on an RTX PRO 6000, versus the
    336px CLIP-L members. Enabling this component roughly doubles the non-MLLM inference cost of the
    ensemble. That is a deployment trade, not a bug.

Class order is FIXED PROJECT-WIDE as 0=real, 1=pad, 2=deepfake.
"""
from __future__ import annotations

import os
import sys
from typing import List, Optional

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REAL = 0


class PESPCModel:
    name = "pespc"

    def __init__(self, ckpt_path: str, encoder_path: str, device: str = "cuda:0",
                 amp_dtype: str = "bf16", model_name: str = "PE-Core-G14-448"):
        for d in (_ROOT, os.path.join(_ROOT, "perception_models"), os.path.join(_ROOT, "pespc")):
            if d not in sys.path:
                sys.path.insert(0, d)
        import torch
        import core.vision_encoder.pe as pe
        import core.vision_encoder.transforms as pt
        from head import build_head

        self.torch = torch
        self.device = device
        ck = torch.load(ckpt_path, map_location="cpu")
        cfg = ck["cfg"]
        # The head's `dim` must match the encoder that produced the training features. A 1024-d head
        # (PE-Core-L14-336) loaded against a 1280-d encoder would fail here rather than silently
        # produce garbage scores -- state_dict load is strict.
        self.head = build_head(cfg)
        self.head.load_state_dict(ck["state"])
        self.head = self.head.to(device).eval()

        enc = pe.CLIP.from_config(model_name, pretrained=True, checkpoint_path=encoder_path)
        self.enc = enc.to(device).eval()
        self.image_size = int(enc.image_size)
        self.tfm = pt.get_image_transform(self.image_size)
        self.amp = {"bf16": torch.bfloat16, "fp16": torch.float16,
                    "fp32": torch.float32}.get(amp_dtype, torch.bfloat16)
        self.enc = self.enc.to(self.amp)
        self.cfg = cfg

    def score_frames(self, rgb_list: List[np.ndarray], batch_size: int = 32) -> List[dict]:
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
                    f = self.enc.encode_image(x.to(self.amp), normalize=True)
                    logits = self.head(f.float())
                    p = torch.softmax(logits, dim=-1)[:, REAL].float().cpu().numpy()
                    for j, v in enumerate(p):
                        out[i + j] = {"fake": float(1.0 - v), "error": None}
                except Exception as exc:                                # noqa: BLE001
                    for j in range(len(chunk)):
                        out[i + j] = {"fake": None, "error": repr(exc)}
        return out
