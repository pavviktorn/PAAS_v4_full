"""PAAS_ensemble_v3 experiment configuration.

A single ``PaasConfig`` describes ONE experiment: which detectors to run and how to fuse their
per-frame fake-scores into a decision. v3 fuses up to FIVE component detectors:

    ffaa   - LLaVA-Mistral-7B MLLM + MIDS       (Exp 9)
    A1_9c  - 9-class SVD ensemble member        (Exp 7-8)   } read from the 9-class ensemble's
    A2_9c  - 9-class SVD+GenD ensemble member    (Exp 7-8)   } per-model output (A3_9c also available)
    gsd    - Geometric Semantic Decoupling       (Exp 10)
    selop  - SeLop / LROR low-rank orthogonal     (Exp 11)

The recommended combination (docs/COMBINATION_FINDINGS_axon1.md, Exp 13) is the PLAIN MEAN of
{ffaa, A1_9c, A2_9c, gsd, selop} -- AUC 0.9998, fake-recall 99.97% at a 90% real-recall floor.
Which detectors load is derived from ``fusion.components``, so changing the combination is a config
change, not a code change.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from typing import List, Optional

from . import env

# every component name the fusion understands (which model produces each is handled in the pipeline)
ENS_MEMBERS = ("A1_9c", "A2_9c", "A3_9c")
ALL_COMPONENTS = ("ffaa",) + ENS_MEMBERS + ("gsd", "gsdA", "selop", "pespc", "dinospc", "lorc")


@dataclass
class FFAACfg:
    # v4 FFAA = Qwen3.5-4B MLLM (IN-PROCESS vLLM, single venv/tf5) + FROM-SCRATCH MIDS 4-class head.
    # Everything runs in ONE process on the same venv: vLLM builds the LLM in-process (no server, no
    # HTTP), and the MIDS/CLIP stack loads in the same interpreter via the tf5 weight remap.
    enabled: bool = True
    qwen_dir: str = env.QWEN_DIR                      # merged Qwen3.5-4B model dir (loaded in-process)
    qwen_gpu_mem: float = 0.45                        # vLLM GPU frac; rest is left for MIDS/A2/GSD/SeLop
    qwen_max_model_len: int = 4096
    qwen_max_pixels: int = 451584                     # mm_processor longest_edge
    mids_path: str = env.MIDS_PATH
    # TRUE for this project. Every MIDS head shipped here (and every head run_finetuning.sh trains)
    # is trained on the LETTERBOXED WHOLE FRAME; only the legacy v4 head used a CLIP centre crop.
    # Serving a whole-frame head with whole_frame=False silently changes the preprocessing and the
    # scores move a lot -- measured max|d| 0.997 against EVAL_SPACE's recorded ffaa scores, i.e. the
    # detector effectively becomes a different model. EVAL_SPACE/score_final.py refuses to score with
    # whole_frame=false for exactly this reason; the default now matches the weights.
    whole_frame: bool = True
    clip_path: str = env.BASE_CLIP
    t5_path: str = env.BASE_T5
    prompt_file: Optional[str] = env.PROMPT_FILE     # first line = FFAA base prompt
    prompt: str = "The image is a human face image. Is it real or fake? Why?"
    max_new_tokens: int = 512
    generate_num: int = 3                            # 3-pass conditional protocol (N=1,M=1)
    cache_path: Optional[str] = None


@dataclass
class Ensemble9Cfg:
    enabled: bool = True
    config_path: str = env.ENSEMBLE9_CONFIG


@dataclass
class GSDCfg:
    enabled: bool = True
    ckpt: str = env.GSD_CKPT
    clip_path: str = env.BASE_CLIP
    amp_dtype: str = "bf16"


@dataclass
class GSDACfg:
    """GSD anchor-swap arm: identical mechanism to GSDCfg, different reference basis.

    Kept as a SEPARATE slot rather than a different value of GSDCfg.ckpt so that 'gsd' and 'gsdA' can
    be served side by side. Measured in EVAL_SPACE the two are indistinguishable inside a 6-member
    mean (0.999546 both), so the reason to prefer gsdA is provenance -- its basis is not fitted on the
    evaluation split -- not accuracy.
    """
    enabled: bool = True
    ckpt: str = env.GSDA_CKPT
    clip_path: str = env.BASE_CLIP
    amp_dtype: str = "bf16"


@dataclass
class DinoSPCCfg:
    """DINOv3-SPC: frozen self-supervised DINOv3 ViT-7B/16 + a trained 98,308-param prototype head.

    The only member with NO language supervision anywhere in its encoder, which is the point: the
    other members are all language-supervised contrastive, so this one decorrelates rather than
    duplicates. Measured on the certified split it catches 5 deepfakes the previously deployed six
    miss while missing none of theirs.

    It is also the most expensive member by far: 6.7B encoder params at 384px, and a 26 GB weight
    directory. `encoder_path` is resolved by env.py as bundled-then-shared; set it explicitly to
    override. Resolution is NOT configurable here -- it comes from the checkpoint cfg, so serving
    cannot silently run at a different size than the head was trained at.
    """
    enabled: bool = True
    ckpt: str = env.DINOSPC_CKPT
    encoder_path: str = env.DINOSPC_ENCODER
    amp_dtype: str = "bf16"


@dataclass
class LoRCCfg:
    """LoRC: frozen DINOv3 ViT-H+/16 + LoRA r=16 (q/k/v/o) + a rank-32 low-rank-attention head.

    5.4M trainable parameters. On the certified report split it is the second-best single member
    in the set (24 errors, against 22 for the deployed six-member fusion as a whole), and adding
    it takes that fusion to 18 -- the only significant gain an exhaustive search over all 2,047
    subsets of 11 members found.

    Resolution, crop mode, LoRA shape and the register-token offset all come from the checkpoint
    and are cross-checked against the encoder at load time, so serving cannot silently diverge
    from training. The 224-crop variant was measured at 63 errors against 18, which is why the
    wrapper refuses any crop_mode but 'resize'.
    """
    enabled: bool = True
    ckpt: str = env.LORC_CKPT
    encoder_path: str = env.LORC_ENCODER
    amp_dtype: str = "bf16"


@dataclass
class PESPCCfg:
    """PE-SPC: frozen Perception Encoder + trained prototype head.

    COST WARNING: the encoder is 1.88B params at 448px (~47.8 img/s), against 336px CLIP-L for every
    other member. Enabling this component roughly doubles non-MLLM inference cost.
    """
    enabled: bool = True
    ckpt: str = env.PESPC_CKPT
    encoder_path: str = env.PESPC_ENCODER
    model_name: str = env.PESPC_MODEL_NAME
    amp_dtype: str = "bf16"


@dataclass
class SeLopCfg:
    enabled: bool = True
    ckpt: str = env.SELOP_CKPT
    clip_path: str = env.BASE_CLIP
    amp_dtype: str = "bf16"


@dataclass
class FusionCfg:
    # method: "mean" | "weighted"  (over the component fake-scores listed below)
    method: str = "mean"
    # DEFAULT for PAAS_v4_full: mean{ffaa, A1_9c, A2_9c, gsdA, selop, PE-SPC}.
    # gsdA rather than gsd -- identical inside a 6-member mean (test AUC 0.999546 both), so the
    # cleaner anchor provenance decides it. PE-SPC improved 255/255 fusions on the EVAL_SPACE test
    # set and is the strongest single member there (0.998864).
    # Kept in sync with config/experiments/paas_v4full_default.json.
    components: List[str] = field(
        default_factory=lambda: ["ffaa", "A1_9c", "A2_9c", "gsdA", "selop", "pespc"])
    # for method=="weighted": one weight per component (same order); normalised internally.
    weights: Optional[List[float]] = None


@dataclass
class DecisionCfg:
    # Default = mean{ffaa, A1_9c, A2_9c, gsdA, selop, pespc} on the weights PROMOTED 2026-08-31
    # (full-trainset retrain, runs/finetune_20260825_104051). Fitted by train/fit_threshold.py on
    # TESTSET_DIR, 30,218 frames (10,126 real / 20,092 fake), AUC 1.0000:
    #   real>=0.90 -> tau 0.0168   real>=0.95 -> 0.0386
    #   real>=0.98 -> tau 0.0960   real>=0.99 -> 0.14085  <- this default
    # The 99% floor is kept from the previous build's policy: this fusion's thresholds transfer
    # poorly off the split they were fitted on, and the gap widens as the floor loosens.
    # Superseded 0.200609 (previous weights, fitted on es_val).
    # ONLY VALID FOR THE WEIGHTS IT WAS FITTED ON -- re-fit after any retrain.
    threshold: float = 0.14085
    real_ambiguous_match_min: float = 0.9  # decision==real & match<this -> "ambiguous"
    treat_likely_fake_as_ambiguous: bool = True


@dataclass
class PaasConfig:
    name: str = "paas4_qwen_mean"
    device: str = "cuda:0"
    ffaa: FFAACfg = field(default_factory=FFAACfg)
    ensemble9: Ensemble9Cfg = field(default_factory=Ensemble9Cfg)
    gsd: GSDCfg = field(default_factory=GSDCfg)
    gsdA: GSDACfg = field(default_factory=GSDACfg)
    selop: SeLopCfg = field(default_factory=SeLopCfg)
    pespc: PESPCCfg = field(default_factory=PESPCCfg)
    dinospc: DinoSPCCfg = field(default_factory=DinoSPCCfg)
    lorc: LoRCCfg = field(default_factory=LoRCCfg)
    fusion: FusionCfg = field(default_factory=FusionCfg)
    decision: DecisionCfg = field(default_factory=DecisionCfg)

    # ---- which models must load, derived from the requested components ----
    def needs(self) -> dict:
        c = set(self.fusion.components)
        return {
            "ffaa": "ffaa" in c,
            "ens": bool(c & set(ENS_MEMBERS)),
            "gsd": "gsd" in c,
            "gsdA": "gsdA" in c,
            "selop": "selop" in c,
            "pespc": "pespc" in c,
            "dinospc": "dinospc" in c,
            "lorc": "lorc" in c,
        }

    # ---- validation ----
    def validate(self) -> "PaasConfig":
        comps = self.fusion.components
        if not comps:
            raise ValueError("fusion.components is empty")
        bad = [c for c in comps if c not in ALL_COMPONENTS]
        if bad:
            raise ValueError(f"unknown fusion components {bad}; allowed: {ALL_COMPONENTS}")
        if self.fusion.method not in ("mean", "weighted"):
            raise ValueError(f"fusion.method must be 'mean' or 'weighted', got {self.fusion.method!r}")
        if self.fusion.method == "weighted":
            w = self.fusion.weights
            if not w or len(w) != len(comps):
                raise ValueError("fusion.method=='weighted' needs weights of len(components)")
            # NaN/inf pass every ordering test below: `NaN < 0` is False, and `sum([...NaN]) <= 0`
            # is False, so [1.0, NaN] validated and then made every fused score NaN -- which the
            # threshold comparison silently turns into "real" for every input.
            bad = [x for x in w if not math.isfinite(float(x))]
            if bad:
                raise ValueError(f"fusion.weights must all be finite, got {self.fusion.weights} "
                                 f"(offending: {bad})")
            # weights are normalised by their SUM, so a zero sum divides by zero and a negative
            # weight inverts a detector's polarity -- [1, -1] validated and produced NaN.
            if any(w < 0 for w in self.fusion.weights):
                raise ValueError(f"fusion.weights must be non-negative, got {self.fusion.weights}; "
                                 f"a negative weight inverts that detector's polarity")
            if sum(self.fusion.weights) <= 0:
                raise ValueError(f"fusion.weights must sum to > 0, got {self.fusion.weights} "
                                 f"(they are normalised by their sum)")
        # The threshold is compared against a fused score that is always in [0, 1]. A value outside
        # that range is not a strict operating point, it is a CONSTANT verdict: tau=2 calls every
        # input real, tau=-1 calls every input fake, and tau=NaN makes every comparison False (so
        # every input is real). All three used to validate and serve.
        t = self.decision.threshold
        if not math.isfinite(float(t)):
            raise ValueError(f"decision.threshold must be finite, got {t!r} "
                             f"(a NaN threshold makes every comparison False -> everything 'real')")
        if not (0.0 <= float(t) <= 1.0):
            raise ValueError(f"decision.threshold must be in [0, 1], got {t} -- fused scores are "
                             f"in [0, 1], so this threshold is a constant verdict, not an "
                             f"operating point")
        m = self.decision.real_ambiguous_match_min
        if not math.isfinite(float(m)) or not (0.0 <= float(m) <= 1.0):
            raise ValueError(f"decision.real_ambiguous_match_min must be finite in [0, 1], got {m}")
        need = self.needs()
        if need["ffaa"] and not self.ffaa.enabled:
            raise ValueError("components include 'ffaa' but ffaa.enabled=False")
        if need["ens"] and not self.ensemble9.enabled:
            raise ValueError("components include a 9-class member but ensemble9.enabled=False")
        if need["gsd"] and not self.gsd.enabled:
            raise ValueError("components include 'gsd' but gsd.enabled=False")
        if need["gsdA"] and not self.gsdA.enabled:
            raise ValueError("components include 'gsdA' but gsdA.enabled=False")
        if need["pespc"] and not self.pespc.enabled:
            raise ValueError("components include 'pespc' but pespc.enabled=False")
        if need["dinospc"] and not self.dinospc.enabled:
            raise ValueError("components include 'dinospc' but dinospc.enabled=False")
        if need["lorc"] and not self.lorc.enabled:
            raise ValueError("components include 'lorc' but lorc.enabled=False")
        if need["selop"] and not self.selop.enabled:
            raise ValueError("components include 'selop' but selop.enabled=False")
        return self

    # ---- (de)serialise ----
    @classmethod
    def from_dict(cls, d: dict) -> "PaasConfig":
        d = dict(d)
        sub = {
            "ffaa": (FFAACfg, d.pop("ffaa", {})),
            "ensemble9": (Ensemble9Cfg, d.pop("ensemble9", {})),
            "gsd": (GSDCfg, d.pop("gsd", {})),
            "gsdA": (GSDACfg, d.pop("gsdA", {})),
            "selop": (SeLopCfg, d.pop("selop", {})),
            "pespc": (PESPCCfg, d.pop("pespc", {})),
            "dinospc": (DinoSPCCfg, d.pop("dinospc", {})),
            "lorc": (LoRCCfg, d.pop("lorc", {})),
            "fusion": (FusionCfg, d.pop("fusion", {})),
            "decision": (DecisionCfg, d.pop("decision", {})),
        }
        kw = {k: klass(**(vals or {})) for k, (klass, vals) in sub.items()}
        # ANY unrecognised detector section is a silent data-loss bug: gsdA/pespc were added to the
        # dataclass but not to `sub`, so their sections were dropped here and replaced by defaults --
        # a config naming a retrained gsdA checkpoint quietly served the default path instead.
        # Refuse on unknown non-annotation keys rather than discarding them.
        top = {"name", "device"}
        unknown = [k for k in d if k not in top and not k.startswith("_")]
        if unknown:
            raise ValueError(f"unknown config section(s) {unknown}; known detector sections are "
                             f"{sorted(sub)} plus {sorted(top)} (keys starting with '_' are notes)")
        d = {k: v for k, v in d.items() if k in top}
        return cls(**d, **kw).validate()

    @classmethod
    def from_file(cls, path: str) -> "PaasConfig":
        with open(path) as fh:
            return cls.from_dict(json.load(fh))

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)
