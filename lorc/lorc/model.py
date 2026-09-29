#!/usr/bin/env python3
"""LoRC -- Detecting AI-Generated Images via Low-Rank Collapse in Semantic Residuals.

  Yan, Chen, Zhan, Wang, Xiao, Ding, Zhang, Yao, Zhang.  arXiv:2608.20882v1 [cs.CV]
  Shanghai Jiao Tong University + Tencent Youtu Lab.

The method in three parts, each implemented below with the paper's equation number:

  Eq 1  SEMANTIC-RESIDUAL DECOMPOSITION.  Project every patch token onto the unit [CLS] direction
        and keep the ORTHOGONAL COMPLEMENT:
            X_sem = X (c_hat c_hat^T),      X_res = X (I - c_hat c_hat^T)
        X in R^{N x D} are the patch tokens, c_hat = c / ||c||_2 the normalised [CLS] token.
        The claim is that generated images collapse to a LOW-RANK residual subspace while the
        dominant semantic direction is preserved -- so the discriminative signal lives in X_res,
        not in the semantics the encoder was trained to represent.

  Eq 3-5  LOW-RANK ATTENTION.  One self-attention block whose Q/K/V project D -> r (r << D):
            Q = X_res W^Q,  K = X_res W^K,  V = X_res W^V,   W^{Q,K,V} in R^{D x r}
            A = Softmax(Q K^T / sqrt(r)) V
            Y = A W^O,                                        W^O in R^{r x D}
        The rank-r bottleneck is the inductive bias: if the residuals really are low-rank, a rank-r
        read of them loses nothing, and it cannot fit a full-rank artefact.

  Eq 6-7  SUBSPACE SEPARATION LOSS.  Per minibatch, flatten the patch residuals of each label group
        into R in R^{BN x D}, take the Frobenius-normalised second moment, and push the two apart:
            P = R^T R / ||R^T R||_F,        L_SS = <P_real, P_fake>_F
        Total objective: L = L_BCE + lambda_SS * L_SS.

This module is the model and the loss only. Data, training loop and reporting live in
lorc_data.py / train_lorc.py, matching how extract_dino.py / train_dinospc.py are split.

-------------------------------------------------------------------------------------------------
WHAT THE PAPER DOES NOT SAY, AND WHAT THIS FILE CHOSES INSTEAD

Every one of these is a knob, defaulting to the reading that follows the paper most literally.
papers/lorc_2608.20882_notes.md records the quote each reading is based on.

  1. WHICH LAYERS GET LoRA.  The paper says only "DINOv3 ViT-H+/16 as the backbone, fine-tuned via
     LoRA with rank 16 and scaling factor alpha=16". Default here: the four attention projections
     (q,k,v,o) of every block -- the standard choice. `lora.target_modules` overrides it.
  2. WHAT "POOLED" MEANS IN THE HEAD.  "The pooled residual feature, concatenated with the frozen
     [CLS] token, is fed to a linear classifier." The figure routes the Low-Rank Attention OUTPUT
     into the head, so the default is mean_n(Y) || c. `head_input` also accepts the literal reading
     mean_n(X_res) || c, and both.
  3. IS THE [CLS] IN THE HEAD DETACHED.  "frozen [CLS]" most plausibly means "taken straight from
     the encoder, not passed through the new block" rather than "no gradient" -- detaching it would
     cut the only path by which the classifier can teach the LoRA adapters anything. Default: not
     detached. `detach_cls` flips it.
  4. BINARY vs 3-CLASS.  The paper is a real/fake AIGI detector: a single logit under L_BCE, and
     an L_SS with exactly two groups. THIS IMPLEMENTATION IS 3-CLASS ONLY -- real / pad / deepfake
     -- because that is the label space of this project and of every other member of the fusion.
     Concretely:
       * the head emits 3 logits and L_BCE becomes cross-entropy over the 3 classes;
       * the served score is 1 - softmax[:, REAL], identical in definition to what dinospc, pespc,
         selop and the 9-class arms emit, so LoRC's column needs no rescaling in the search;
       * L_SS still groups real-vs-fake by default (ssl_mode="binary"), which is the paper's own
         grouping and the one its derivation supports -- pad and deepfake are pooled into "fake"
         for the covariance, while the CLASSIFIER keeps them apart. ssl_mode="pairwise" instead
         separates all three subspaces and is this project's extension, not the paper's.
     The 3-class choice is not free: pooling pad with deepfake inside L_SS asks one subspace to
     describe both a generated image and a photograph of a screen. If that hurts, ssl_mode=
     "pairwise" is the ablation that tests it.

  5. DINOv3 IS STOCHASTIC IN train() MODE.  Measured, not assumed: two train()-mode forwards of the
     same images through the same weights differ by up to 2.6e-2 in the hidden states, with
     drop_path_rate=0 and attention_dropout=0. The cause is the position-embedding augmentation
     the config enables (pos_embed_rescale=2.0), which randomly rescales the positional grid during
     training. eval()-mode forwards are bit-identical. This matters here more than usual, because
     the augmentation perturbs the positional structure of exactly the patch tokens whose residual
     SPECTRUM the method reads -- so every scoring path (engine.evaluate, predict.py,
     diagnose_collapse.py) puts the encoder in eval() first, and any claim of reproducibility is
     only valid in eval mode.
-------------------------------------------------------------------------------------------------
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .losses import subspace_separation_loss  # noqa: F401  (re-exported for callers)

REAL, PAD, DEEPFAKE = 0, 1, 2

# DINOv3 was trained with ImageNet statistics; extract_dino.py uses the same pair, so features
# from the two code paths are comparable.
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


# ------------------------------------------------------------------ Eq 1
def semantic_residual_split(patches, cls, eps=1e-6):
    """Eq 1. -> (X_sem, X_res, c_hat).  patches (B,N,D), cls (B,D); returns float32.

    Computed as X - (X c_hat) c_hat^T rather than by building the D x D projector: for
    ViT-7B/16 that projector would be a 4096 x 4096 matrix per sample (67 MB in fp32), and the
    rank-1 form is algebraically identical.

    float32 throughout on purpose. c_hat is a direction, and in bf16 (8 mantissa bits) the
    subtraction X - (X.c_hat) c_hat loses most of its significance exactly when the residual is
    small -- which, if the paper is right, is precisely the regime that carries the signal.
    """
    c = F.normalize(cls.float(), dim=-1, eps=eps)              # (B,D)
    x = patches.float()                                        # (B,N,D)
    proj = torch.einsum("bnd,bd->bn", x, c)                    # (B,N)   = X c_hat
    sem = proj.unsqueeze(-1) * c.unsqueeze(1)                  # (B,N,D) = (X c_hat) c_hat^T
    return sem, x - sem, c


# ------------------------------------------------------------------ Eq 3-5
class LowRankAttention(nn.Module):
    """Eq 3-5: single-head self-attention through a rank-r bottleneck.

    Q/K/V map D -> r and W^O maps r -> D, so the block's parameter count is 4*D*r and its output
    lives in a rank-r subspace of R^D by construction.

    The attention itself is F.scaled_dot_product_attention with scale = 1/sqrt(r), which IS
    Eq 4 -- smoke_lorc.py asserts it agrees with the explicit softmax(QK^T/sqrt(r))V to 1e-5,
    because "the fast path is equivalent" is the kind of claim that should be tested rather than
    asserted in a comment.
    """

    def __init__(self, dim, rank, bias=False, dropout=0.0):
        super().__init__()
        self.dim, self.rank = dim, rank
        self.q = nn.Linear(dim, rank, bias=bias)
        self.k = nn.Linear(dim, rank, bias=bias)
        self.v = nn.Linear(dim, rank, bias=bias)
        self.o = nn.Linear(rank, dim, bias=bias)
        self.scale = rank ** -0.5
        self.dropout = dropout

    def forward(self, x, naive=False):
        """-> (Y in R^{B,N,D} after W^O, A in R^{B,N,r} before it).

        Both are returned because the paper's text and its supplement disagree about which one the
        Subspace Separation Loss operates on, and that is a question to settle by experiment rather
        than by picking a reading -- see `ssl_source` on LoRCModel.
        """
        q, k, v = self.q(x), self.k(x), self.v(x)              # (B,N,r)
        if naive:
            w = torch.softmax(q @ k.transpose(-2, -1) * self.scale, dim=-1)
            a = w @ v
        else:
            a = F.scaled_dot_product_attention(
                q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1), scale=self.scale,
                dropout_p=self.dropout if self.training else 0.0).squeeze(1)
        return self.o(a), a                                    # (B,N,D), (B,N,r)

    def n_trainable(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ------------------------------------------------------------------ encoder
def build_encoder(path, dtype=torch.bfloat16, lora=None, grad_checkpointing=False, log=print):
    """-> (module, hidden_size, n_register_tokens, n_trainable).

    The backbone is frozen and LoRA adapters are the only trainable part of it, which is how the
    paper's two statements ("fine-tuned via LoRA", "we keep the encoder frozen") are consistent.

    hidden_size and num_register_tokens are read off the config BEFORE any peft wrapping, because
    after get_peft_model the config sits behind .base_model.model and the indirection is a place
    to get the register offset wrong. The offset matters: DINOv3's sequence is
    [CLS, register x num_register_tokens, patches] (4 registers for ViT-7B/16) while DINOv2 has
    none, so a hardcoded tokens[:, 1:] folds register tokens into the patch block. Register tokens
    are trained to hold global information deliberately kept OUT of the patches, i.e. exactly the
    semantic content Eq 1 is trying to project away.
    """
    from transformers import AutoModel
    m = AutoModel.from_pretrained(path, dtype=dtype)
    hidden = int(m.config.hidden_size)
    nreg = int(getattr(m.config, "num_register_tokens", 0) or 0)
    for p in m.parameters():
        p.requires_grad_(False)

    if lora and lora.get("enabled", True):
        from peft import LoraConfig, get_peft_model
        tgt = list(lora.get("target_modules") or ["q_proj", "k_proj", "v_proj", "o_proj"])
        cfg = LoraConfig(r=int(lora.get("r", 16)), lora_alpha=int(lora.get("alpha", 16)),
                         lora_dropout=float(lora.get("dropout", 0.0)), bias="none",
                         target_modules=tgt)
        m = get_peft_model(m, cfg)
        log(f"[lorc] LoRA r={cfg.r} alpha={cfg.lora_alpha} on {tgt}")
    else:
        log("[lorc] LoRA DISABLED -- encoder fully frozen (the paper's 'Baseline' ablation row)")

    if grad_checkpointing:
        # Without enable_input_require_grads the checkpointed blocks see inputs that require no
        # grad and silently save nothing to differentiate, so the LoRA adapters get no gradient.
        base = getattr(m, "base_model", m)
        base = getattr(base, "model", base)
        if hasattr(base, "enable_input_require_grads"):
            base.enable_input_require_grads()
        (base if hasattr(base, "gradient_checkpointing_enable") else m).\
            gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        log("[lorc] gradient checkpointing ON (use_reentrant=False)")

    n_tr = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return m, hidden, nreg, n_tr


# ------------------------------------------------------------------ the model
class LoRCModel(nn.Module):
    """DINOv3 (frozen + LoRA) -> Eq 1 -> Eq 3-5 -> linear classifier.

    forward() returns a dict rather than a tuple so that adding a diagnostic output later cannot
    silently shift a positional element.
    """

    # `res` feeds the RAW per-patch residual mean to the classifier, which lets the head route
    # AROUND the rank-r bottleneck and read nuisance artefacts directly -- so it is available but
    # not the default. `eq2` is the paper's Eq-2 statistic done correctly: the SCALAR mean of
    # per-patch residual norms, (1/N) sum_i ||x_res,i||_2, which is NOT res.mean(1) (a D-vector
    # mean of signed components, whose norm is not a mean of norms).
    HEAD_INPUTS = ("attn+cls", "attn+eq2+cls", "res+cls", "attn+res+cls", "attn")
    PART_DIMS = {"attn": 1, "res": 1, "cls": 1, "eq2": 0}     # multiples of hidden_size; eq2 is 1-d
    NUM_CLASSES = 3
    CLASS_NAMES = ("real", "pad", "deepfake")

    SSL_SOURCES = ("residual", "attn_out", "attn_latent")

    def __init__(self, encoder, hidden_size, n_register_tokens, num_classes=3, rank=32,
                 head_input="attn+cls", head_norm="none", detach_cls=False,
                 attn_bias=False, attn_dropout=0.0, head_dropout=0.0,
                 ssl_source="attn_out"):
        super().__init__()
        if head_input not in self.HEAD_INPUTS:
            raise ValueError(f"head_input must be one of {self.HEAD_INPUTS}, got {head_input!r}")
        if int(num_classes) != self.NUM_CLASSES:
            # Refused rather than silently coerced: a binary checkpoint and a 3-class checkpoint
            # produce differently-calibrated `fake_score` columns, and a run that quietly changed
            # label space would be compared against the other members as if it had not.
            raise ValueError(
                f"num_classes must be 3 (real/pad/deepfake) -- got {num_classes}. The paper's own "
                f"head is binary, but every member of this fusion is 3-class and the served score "
                f"is 1 - softmax[:, REAL]; see the module docstring, note 4.")
        self.encoder = encoder
        self.hidden_size = hidden_size
        self.patch0 = 1 + int(n_register_tokens)
        self.n_register_tokens = int(n_register_tokens)
        if ssl_source not in self.SSL_SOURCES:
            raise ValueError(f"ssl_source must be one of {self.SSL_SOURCES}, got {ssl_source!r}")
        self.num_classes = self.NUM_CLASSES
        self.head_input = head_input
        self.ssl_source = ssl_source
        self.detach_cls = bool(detach_cls)

        self.lra = LowRankAttention(hidden_size, rank, bias=attn_bias, dropout=attn_dropout)
        parts = head_input.split("+")
        feat_dim = sum(self.PART_DIMS[p] * hidden_size if self.PART_DIMS[p] else 1
                       for p in parts)
        self.out_dim = self.NUM_CLASSES
        self.norm = nn.LayerNorm(feat_dim) if head_norm == "layernorm" else nn.Identity()
        self.drop = nn.Dropout(head_dropout) if head_dropout > 0 else nn.Identity()
        self.head = nn.Linear(feat_dim, self.out_dim)
        nn.init.zeros_(self.head.bias)

    def encode(self, pixel_values):
        """-> (cls (B,D), patches (B,N,D)) in the encoder's dtype."""
        out = self.encoder(pixel_values=pixel_values).last_hidden_state
        return out[:, 0], out[:, self.patch0:]

    def forward(self, pixel_values, need_residual=True, naive_attn=False):
        cls, patches = self.encode(pixel_values)
        _, res, c_hat = semantic_residual_split(patches, cls)       # Eq 1, float32
        y, a = self.lra(res, naive=naive_attn)                      # Eq 3-5, float32

        cls_f = cls.float()
        if self.detach_cls:
            cls_f = cls_f.detach()
        parts = {"attn": y.mean(1), "res": res.mean(1), "cls": cls_f,
                 # Eq 2, literally: scalar mean over patches of the per-patch residual NORM.
                 "eq2": res.norm(dim=-1).mean(1, keepdim=True)}
        feat = torch.cat([parts[k] for k in self.head_input.split("+")], dim=-1)
        logits = self.head(self.drop(self.norm(feat)))
        return {"logits": logits, "residual": res if need_residual else None,
                "cls": cls_f, "c_hat": c_hat, "attn_out": y, "attn_latent": a}

    # ---------------------------------------------------------------- scoring
    def fake_score(self, logits):
        """-> P(fake) = 1 - softmax[:, REAL], shape (B,), in [0,1].

        Deliberately the same quantity every other member of this fusion emits (dinospc, pespc,
        selop, the 9-class arms), so LoRC's score column is directly comparable and needs no
        rescaling in the combination search. PAD and DEEPFAKE both count as fake here; which KIND
        of fake it is comes from the argmax over the two fake logits, which is what
        metrics.block_at_tau reports as the deployable rule.
        """
        return 1.0 - logits.float().softmax(-1)[:, REAL]

    def ssl_features(self, out):
        """The tensor Eq 6-7 operates on, per `ssl_source`.

        WHY THIS IS A KNOB AND NOT A CONSTANT. The paper's main text writes Eq 6 over
        "flattened patch residuals", i.e. X_res -- BEFORE the low-rank bottleneck. Its supplementary
        placement ablation instead reports the released model applying the loss AFTER the
        bottleneck, and by a wide margin (reported: no SSL 94.4, before 95.0, after 96.8). The two
        statements cannot both describe the same model, so the placement is settled here by
        experiment:

          "attn_out"     Y in R^{N x D}, after W^O -- the default, the supplement's reading.
          "attn_latent"  A in R^{N x r}, before W^O -- the other post-bottleneck reading; P is then
                         only r x r, a far stronger constraint.
          "residual"     X_res -- the main text's reading, and what the paper's own ablation calls
                         the weaker "SSL before low-rank" variant.
        """
        t = out[self.ssl_source]
        if t is None:
            raise RuntimeError(f"forward() was called with need_residual=False, so "
                               f"ssl_source={self.ssl_source!r} is unavailable")
        return t

    def class_probs(self, logits):
        """-> (B,3) softmax over real/pad/deepfake."""
        return logits.float().softmax(-1)

    def classification_loss(self, logits, labels, class_weight=None):
        """The paper's L_BCE, as cross-entropy over real/pad/deepfake.

        For a two-class head these coincide; with three classes the extra term is what lets the
        model say WHICH fake it saw, at no cost to the real-vs-fake score the fusion consumes.
        `class_weight` is a 3-vector indexed [real, pad, deepfake].
        """
        return F.cross_entropy(logits.float(), labels, weight=class_weight)

    def trainable_groups(self):
        """-> (encoder_params, new_params). Kept separate so the LoRA adapters and the newly
        initialised block can take different learning rates; the paper reports a single lr, which
        is what config lr/head_lr default to."""
        enc = [p for p in self.encoder.parameters() if p.requires_grad]
        new = [p for n, p in self.named_parameters()
               if p.requires_grad and not n.startswith("encoder.")]
        return enc, new

    def n_trainable(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
