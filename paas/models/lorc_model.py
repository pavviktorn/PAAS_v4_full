"""Wrapper around LoRC (arXiv 2608.20882v1): frozen DINOv3 ViT-H+/16 + LoRA + a rank-32 head.

WHY THIS MEMBER EXISTS. Measured on the certified report split es_dev_eval_c99 (35,171 rows, tau
transferred from the selector, nothing selected on it), LoRC H+ ALONE makes 24 errors where the
previously deployed six-member fusion makes 22 -- it is the second-best single model in the whole
member set, behind only its own 7B sibling, and ahead of a3_9c (36), ensemble (40) and a2_9c (50).
Added as a seventh member it takes the fusion from 22 errors to 18 (+0.000175 fake_rec@real99,
paired bootstrap CI [+0.000044, +0.000351] -- significant). An exhaustive search over all 2,047
subsets of the 11 available members found no combination better than 18, and LoRC H+ appears in
every one of the top 15.

WHAT IT ACTUALLY DOES. Patch tokens are split along the CLS direction (Eq 1): X_sem = X c_hat
c_hat^T, X_res = X - X_sem. A single-head attention through a rank-32 bottleneck runs over the
RESIDUAL only (Eq 3-5), and a linear head classifies [attn_mean || cls] into real/pad/deepfake.
The backbone is frozen; LoRA r=16 on q/k/v/o is the only thing trained inside it, 5.4M parameters
in total.

THREE THINGS THAT DIFFER FROM dinospc AND ARE EASY TO GET WRONG:

  * THE PAPER'S OWN SUBSPACE SEPARATION LOSS IS NOT USED, deliberately. Measured across five
    full-budget arms, switching L_SS off gives bit-identical accuracy (0.999207 either way, the
    same 18 errors on the selector) while the paper claims +2.4 for it. The deployed checkpoint
    trains with lambda_ss=0.1 because that is the arm that was selected; a lambda_ss=0 checkpoint
    is exactly as good. Nothing at serving time depends on it -- it is a training-only term.
  * PREPROCESSING IS A WHOLE-IMAGE SQUASH RESIZE TO 384, not the paper's 224 crop. That is not a
    style choice: the 224 crop was measured at 63 errors against 18 for the resize, the single
    largest effect in a 14-arm matrix. The size and crop mode come from the CHECKPOINT cfg, so
    serving cannot silently diverge from training.
  * REGISTER TOKENS. DINOv3 H+/16 emits [CLS, reg x 4, patches]; `tokens[:, 1:]` would fold four
    register tokens into the patch block, and Eq 1 is explicitly trying to project away the global
    semantic content those registers hold. The offset is read from the checkpoint and cross-checked
    against the encoder config, never assumed.

Class order is FIXED PROJECT-WIDE as 0=real, 1=pad, 2=deepfake, and the served score is
1 - softmax[:, real] -- the same quantity every other member emits, so no rescaling is needed in
the fusion.
"""
from __future__ import annotations

import os
import sys
from typing import List, Optional

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REAL = 0
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def _build(torch, nn, F):
    """Build the three nn classes lazily, so importing this module costs no torch."""

    def semantic_residual_split(patches, cls, eps=1e-6):
        """Eq 1 -> (X_sem, X_res, c_hat), float32.

        Computed as X - (X c_hat) c_hat^T rather than by materialising the D x D projector, which
        is algebraically identical and avoids a 1280x1280 matrix per sample. float32 throughout:
        c_hat is a direction, and in bf16 the subtraction loses most of its significance exactly
        when the residual is small -- the regime that carries the signal.
        """
        c = F.normalize(cls.float(), dim=-1, eps=eps)
        x = patches.float()
        proj = torch.einsum("bnd,bd->bn", x, c)
        sem = proj.unsqueeze(-1) * c.unsqueeze(1)
        return sem, x - sem, c

    class LowRankAttention(nn.Module):
        """Eq 3-5: single-head self-attention through a rank-r bottleneck, scale = 1/sqrt(r)."""

        def __init__(self, dim, rank, bias=False, dropout=0.0):
            super().__init__()
            self.dim, self.rank = dim, rank
            self.q = nn.Linear(dim, rank, bias=bias)
            self.k = nn.Linear(dim, rank, bias=bias)
            self.v = nn.Linear(dim, rank, bias=bias)
            self.o = nn.Linear(rank, dim, bias=bias)
            self.scale = rank ** -0.5
            self.dropout = dropout

        def forward(self, x):
            q, k, v = self.q(x), self.k(x), self.v(x)
            a = F.scaled_dot_product_attention(
                q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1), scale=self.scale).squeeze(1)
            return self.o(a), a

    class LoRCNet(nn.Module):
        """Serving-only twin of PAAS_LoRC's lorc.model.LoRCModel.

        The training-side class carries the SSL loss plumbing, diagnostics and several head
        variants that were measured and rejected. This keeps the parameterised pieces the
        checkpoint actually contains -- so a state_dict from the training tree loads strictly
        here, and any divergence is a load error rather than a silent score shift.
        """

        PART_DIMS = {"attn": 1, "res": 1, "cls": 1, "eq2": 0}

        def __init__(self, encoder, hidden_size, n_register_tokens, rank=32,
                     head_input="attn+cls", head_norm="none", detach_cls=False):
            super().__init__()
            self.encoder = encoder
            self.hidden_size = hidden_size
            self.patch0 = 1 + int(n_register_tokens)
            self.head_input = head_input
            self.detach_cls = bool(detach_cls)
            self.lra = LowRankAttention(hidden_size, rank)
            parts = head_input.split("+")
            feat_dim = sum(self.PART_DIMS[p] * hidden_size if self.PART_DIMS[p] else 1
                           for p in parts)
            self.norm = nn.LayerNorm(feat_dim) if head_norm == "layernorm" else nn.Identity()
            self.drop = nn.Identity()
            self.head = nn.Linear(feat_dim, 3)

        def forward(self, pixel_values):
            out = self.encoder(pixel_values=pixel_values).last_hidden_state
            cls, patches = out[:, 0], out[:, self.patch0:]
            _, res, _ = semantic_residual_split(patches, cls)
            y, _ = self.lra(res)
            cls_f = cls.float()
            if self.detach_cls:
                cls_f = cls_f.detach()
            parts = {"attn": y.mean(1), "res": res.mean(1), "cls": cls_f,
                     "eq2": res.norm(dim=-1).mean(1, keepdim=True)}
            feat = torch.cat([parts[k] for k in self.head_input.split("+")], dim=-1)
            return self.head(self.drop(self.norm(feat)))

    return LoRCNet


class LoRCModel:
    name = "lorc"

    def __init__(self, ckpt_path: str, encoder_path: str, device: str = "cuda:0",
                 amp_dtype: str = "bf16", size: int = 0):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        import torchvision.transforms as T
        from transformers import AutoModel

        self.torch = torch
        self.device = device
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = ck["cfg"]
        if int(cfg.get("num_classes", 3)) != 3:
            raise ValueError(f"LoRC checkpoint declares num_classes={cfg.get('num_classes')}; "
                             f"only 3 (real/pad/deepfake) is supported project-wide")
        self.amp = {"bf16": torch.bfloat16, "fp16": torch.float16,
                    "fp32": torch.float32}.get(amp_dtype, torch.bfloat16)

        enc = AutoModel.from_pretrained(encoder_path, dtype=self.amp)
        hidden = int(enc.config.hidden_size)
        nreg = int(getattr(enc.config, "num_register_tokens", 0) or 0)
        # The checkpoint records what it was TRAINED against. A mismatch here means the encoder
        # directory is not the one the head was fitted to, which would load cleanly and score
        # nonsense -- the register offset in particular is invisible in the state_dict.
        for key, got in (("hidden_size", hidden), ("n_register_tokens", nreg)):
            want = ck.get(key)
            if want is not None and int(want) != got:
                raise ValueError(
                    f"LoRC encoder mismatch: checkpoint was trained with {key}={want} but "
                    f"{encoder_path} reports {got}. Serving would silently produce wrong scores.")
        for p in enc.parameters():
            p.requires_grad_(False)
        lora = cfg.get("lora") or {}
        if lora.get("enabled", True):
            from peft import LoraConfig, get_peft_model
            enc = get_peft_model(enc, LoraConfig(
                r=int(lora.get("r", 16)), lora_alpha=int(lora.get("alpha", 16)),
                lora_dropout=0.0, bias="none",
                target_modules=list(lora.get("target_modules")
                                    or ["q_proj", "k_proj", "v_proj", "o_proj"])))

        LoRCNet = _build(torch, nn, F)
        self.net = LoRCNet(enc, hidden, nreg, rank=int(cfg.get("rank", 32)),
                           head_input=cfg.get("head_input", "attn+cls"),
                           head_norm=cfg.get("head_norm", "none"),
                           detach_cls=bool(cfg.get("detach_cls", False)))
        missing, unexpected = self.net.load_state_dict(ck["state"], strict=False)
        # The frozen backbone is identified by PATH and not stored in the checkpoint, so its keys
        # are legitimately absent. Anything else -- a dropped LoRA adapter, a head of the wrong
        # width -- must be an error: strict=False would otherwise leave those tensors at their
        # random init and still serve plausible-looking scores.
        bad = [k for k in missing if not k.startswith("encoder.") or "lora_" in k]
        if bad or unexpected:
            raise ValueError(f"LoRC checkpoint/architecture mismatch. "
                             f"unexpected={unexpected} missing(non-frozen)={bad}")
        self.net = self.net.to(device).eval()

        # Size and crop mode come from the CHECKPOINT. Serving a 384-trained head at 224 was
        # measured at 63 errors against 18 -- the largest single effect in the arm matrix -- so
        # this must not be a caller's choice.
        self.image_size = int(size or cfg.get("image_size") or 384)
        crop = cfg.get("crop_mode", "resize")
        if crop != "resize":
            raise ValueError(f"LoRC checkpoint was trained with crop_mode={crop!r}; only 'resize' "
                             f"(whole image, no crop) is served here -- see the module docstring.")
        self.tfm = T.Compose([
            T.Resize((self.image_size, self.image_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(), T.Normalize(MEAN, STD)])
        self.cfg = cfg

    def score_frames(self, rgb_list: List[np.ndarray], batch_size: int = 16) -> List[dict]:
        """rgb_list: HxWx3 uint8 arrays. -> one dict per frame, input order:
        {"fake": float|None, "error": str|None}. fake = 1 - softmax[:, real]."""
        torch = self.torch
        from PIL import Image
        out: List[Optional[dict]] = [None] * len(rgb_list)
        with torch.no_grad():
            for i in range(0, len(rgb_list), batch_size):
                chunk = rgb_list[i:i + batch_size]
                try:
                    x = torch.stack([self.tfm(Image.fromarray(r)) for r in chunk]).to(self.device)
                    logits = self.net(x.to(self.amp))
                    p = logits.float().softmax(-1)
                    fake = (1.0 - p[:, REAL]).double().cpu().numpy()
                    for j in range(len(chunk)):
                        out[i + j] = {"fake": float(fake[j]), "error": None}
                except Exception as e:                                  # noqa: BLE001
                    for j in range(len(chunk)):
                        out[i + j] = {"fake": None, "error": f"lorc: {type(e).__name__}: {e}"}
        return [o or {"fake": None, "error": "lorc: not scored"} for o in out]
