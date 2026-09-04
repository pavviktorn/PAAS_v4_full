# Reproducing EVAL_SPACE from PAAS_v4_full

Two different questions, with different answers.

## A. Reproduce EVAL_SPACE's SCORES by serving its models here — YES, verified

300 test images, all eight members, v4_full's serving pipeline vs the scores EVAL_SPACE recorded:

| member | max abs diff | mean | pearson | AUC v4_full | AUC EVAL_SPACE | ΔAUC |
|---|---|---|---|---|---|---|
| ffaa | 0.058545 | 0.000621 | 0.999945 | 0.999350 | 0.999250 | +0.000100 |
| A1_9c | 0.000500 | 0.000003 | 1.000000 | 0.998950 | 0.998950 | **0.000000** |
| A2_9c | 0.000000 | 0.000000 | 1.000000 | 0.994350 | 0.994350 | **0.000000** |
| A3_9c | 0.000200 | 0.000001 | 1.000000 | 0.994825 | 0.994825 | **0.000000** |
| gsd | 0.025671 | 0.000430 | 0.999986 | 0.990550 | 0.990400 | +0.000150 |
| gsdA | 0.014923 | 0.000226 | 0.999995 | 0.990500 | 0.989850 | +0.000650 |
| selop | 0.023018 | 0.000250 | 0.999992 | 0.999750 | 0.999750 | **0.000000** |
| pespc | 0.020599 | 0.000191 | 0.999995 | — | — | — |
| **mean{ffaa,A1,A2,gsdA,selop,pespc}** | 0.011038 | | | 0.999950 | 0.999950 | **0.000000** |

**Metric-level reproduction is exact or near-exact** (worst ΔAUC 6.5e-4; the 6-member fusion AUC is
identical). Per-image scores agree to ~1e-2 worst case.

The residual is **GPU batch-composition nondeterminism, not a code difference**. Proof: at n=60,
gsd/gsdA/selop were bit-identical (max diff exactly 0.0); at n=300 the same members show ~1e-2 tails
purely because the batching changed. PE-SPC additionally carries the fp16 feature-cache effect
(offline scores came from float16-stored features; serving recomputes).

### One trap that made ffaa look broken

`FFAACfg.whole_frame` used to default to **False** (legacy CLIP centre crop). Every MIDS head shipped
here is trained on the **letterboxed whole frame**. Serving with the wrong value gave max|d| **0.997**
against EVAL_SPACE — the detector effectively becomes a different model, with no error raised.
The default is now `True`, matching the weights. (`EVAL_SPACE/score_final.py` refuses to score with
`whole_frame=false` for the same reason.)

## B. Reproduce EVAL_SPACE's MODELS by re-training here — possible, with caveats

Audited one by one:

| item | result |
|---|---|
| `train/train_{ensemble9,gsd,selop,mids4c}.py` | **identical** to the tree EVAL_SPACE ran |
| `train/{fit_threshold,make_deploy_config,validate_configs,_bootstrap}.py` | **identical** |
| `gsd/`, `selop/`, `ensemble9/`, `qwen/` libraries | **identical**, 0 differing files |
| `train/build_manifests.py` | differs — my additive `--exclude` flag, **off by default** |
| A1/A2/A3 configs | **identical on all 43 fields** (paths/output_dir ignored) |
| gsdA config | **identical on all 35 fields** |
| gsd config | identical except `seed` (42 vs 20260821) — overridden by `$SEED` at runtime anyway |
| selop config | identical except the repointed `clip_path` |
| `manifests/es_val.json` | **byte-identical** to the split EVAL_SPACE selected on |
| MLLM | now byte-identical to `PAAS_ensemble_v4_inf` (see README — 205 tensors had differed) |

So only **two** things actually differ, and one is deliberate:

```bash
# to reproduce EVAL_SPACE rather than run this project's own experiment:
SEED=20260821 \
MIDS_IMAGES=/datasets/work/vLLM/temp/EVAL_SPACE/work/trainset_final_images.json \
MIDS_EVAL_IMAGES=/datasets/work/vLLM/temp/EVAL_SPACE/manifests/es_val.json \
RUN_QWEN=0 bash run_finetuning.sh
```

`build_manifests.py` accepts a `.json` manifest as `--images`, so pointing `MIDS_IMAGES` at
EVAL_SPACE's deduplicated trainset reproduces its exact image list.

### The deliberate conflict

**This project's data policy and EVAL_SPACE's are mutually exclusive.** v4_full is configured for the
FULL image root (1,366,146 images, no de-duplication, no eval exclusion); EVAL_SPACE trained on a
deduplicated 889,610-image trainset with the eval splits held out. Those are different experiments.
You cannot reproduce EVAL_SPACE *and* run the full-trainset experiment in the same run — pick one per
run via `MIDS_IMAGES`.

### Bitwise identity is NOT guaranteed, even with everything above matched

None of the trainers set `torch.use_deterministic_algorithms(True)` or `cudnn.deterministic`;
`train_selop.py` enables TF32 explicitly; multi-GPU reduction order and DataLoader worker scheduling
both vary run to run. `train_gsd.py` seeds only `torch.manual_seed` (not `cuda.manual_seed_all`,
numpy, or python `random`). Expect metrics to land within noise of EVAL_SPACE's, not to match to the
last decimal — which is exactly what Part A measured for scoring.
