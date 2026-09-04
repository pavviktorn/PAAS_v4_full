# PAAS_v4_full — standalone v4, trained on the FULL MIDS trainset

Standalone clone of `PAAS_ensemble_v4`, rewired to train **eight** detectors on the complete
`no_delete_mids_train` image root and to evaluate against **EVAL_SPACE**.

Nothing has been trained yet. `run_finetuning.sh` has NOT been run.

## Data policy — read this before quoting any number

| | |
|---|---|
| trainset | `/datasets/work/vLLM/data/no_delete_mids_train` — **all 1,366,146 images** |
| de-duplication | **none** |
| eval exclusion | **none** |
| val (checkpoint selection) | `EVAL_SPACE/manifests/es_val.json` (50,384) |
| test (reporting) | `EVAL_SPACE/manifests/es_test.json` (241,667) |

This is a deliberate instruction, and it has a consequence that must travel with every metric:

* **100.0% of es_test** and **96.9% of es_val** live *under* the trainset root, so they are trained on.
* Therefore val/test scores measure **memorisation**, not generalisation, and `best.pt` selection
  (`select_metric` on val) is itself driven by contaminated data.

`run_finetuning.sh` measures the overlap at launch and writes `$OUT_ROOT/train_eval_overlap.json`.
It prints a banner but **does not abort** — the policy is intentional. `train/train_pespc.py` prints
the same measurement for its own splits.

If a clean split is ever wanted, `train/build_manifests.py --exclude <manifest...>
--require-excluded N` removes the eval paths and refuses if the exclusion silently matched nothing.
It is **off by default**.

## The eight branches

| branch | stage flag | config | notes |
|---|---|---|---|
| A1_9c | `RUN_A1` | `train/configs/mids_a1_9c.yaml` | Effort SVD only |
| A2_9c | `RUN_A2` | `train/configs/mids_a2_9c_full.yaml` | SVD + GenD |
| A3_9c | `RUN_A3` | `train/configs/mids_a3_9c.yaml` | SVD + GenD + ForensicsAdapter |
| gsd | `RUN_GSD` | `train/configs/gsd_default.json` | deployed mechanism, anchor = **val split** |
| gsdA | `RUN_GSDA` | `train/configs/gsd_anchorswap.json` | same mechanism, anchor = **5,000-image trainset slice** |
| selop | `RUN_SELOP` | `train/configs/selop_config.json` | per-layer LROR, DDP |
| PE-SPC | `RUN_PESPC` | flags on the stage | frozen PE-Core-G14-448 + ~15k prototype params |
| ffaa | `RUN_QWEN`/`RUN_MIDS` | — | Qwen3.5-4B MLLM + MIDS 4-class head |

The three 9-class configs were copied from the proven EVAL_SPACE runs and differ **only** in their
ablation fields (verify with `diff`): A1 flips `tune_layer_norm`, `alignment_weight`,
`uniformity_weight`; A3 flips `artifact_enabled`. Nothing else varies between them.

**Why both `gsd` and `gsdA`.** They are the same detector differing only in where the inference
reference basis comes from. The deployed recipe builds it from the val split, which makes its val
metric partly self-fitted; `gsdA` builds it from a bounded trainset slice. Standalone in EVAL_SPACE
that swap was worth **+4.0 deepfake points**. Keeping both is what makes the anchor question
answerable instead of assumed.

**PE-SPC recipe** is the DEV-selected one from EVAL_SPACE Experiment 25 (EXPERIMENTS.pdf Z7):
H3 head, `k=4`, `K4` prompts, learnable scale, `lr=0.3` (a plateau from 0.03–0.3, not a sharp
optimum), 2 epochs, 15,364 trainable parameters over a frozen 1.88B encoder.

## Running

```bash
# everything (see cost table first)
bash run_finetuning.sh

# MLLM-free detectors only — the seven cheap branches
RUN_QWEN=0 RUN_GEN=0 RUN_MIDS=0 bash run_finetuning.sh

# one branch
RUN_9C=0 RUN_GSD=0 RUN_GSDA=0 RUN_SELOP=0 RUN_QWEN=0 RUN_GEN=0 RUN_MIDS=0 bash run_finetuning.sh   # PE-SPC only

# end-to-end wiring check, tiny data, minutes
SMOKE=1 SMOKE_GPU=1 bash run_finetuning.sh
```

## Cost, and the one decision left open

The trainset is **1.37M images — about 1.54× the 889,610 EVAL_SPACE run** — so scale that run's
timings up accordingly. PE-SPC's feature-caching phase is storage-bound, not GPU-bound: measured
23–97 img/s per GPU on this filesystem, varying ~2.5×, so estimate from completed shards only.

**`ffaa` is the expensive branch and its scope is a judgment call I have not made for you.**
`RUN_QWEN=1` (the inherited default) retrains the Qwen3.5-4B MLLM from scratch and then regenerates
MIDS answers over all 1.37M images — multi-day. But the MLLM trains on the eFFAA *conversation*
corpus, not on `no_delete_mids_train`; the part of `ffaa` that actually consumes this trainset is the
**MIDS 4-class head** (Step 4). So `RUN_QWEN=0` reuses `weights/qwen35_4b_merged` and retrains only
the head on the full trainset, which is very likely what "train ffaa on the full trainset" means.
Set it deliberately.

## Standalone-ness

**Every weight this project trains with lives inside it.** Verify at any time:

```bash
python scripts/verify_standalone.py     # exits non-zero if anything drifts back out
```

| asset | path | size |
|---|---|---|
| CLIP ViT-L/14-336 | `base_models/clip-vit-large-patch14-336` | 3.2 G |
| T5-base | `base_models/t5-base` | 4.2 G |
| PE-Core-G14-448 | `base_models/PE-Core-G14-448/` | 9.1 G |
| Qwen3.5-4B (base) | `base_models/Qwen3.5-4B` | 8.8 G |
| Qwen3.5-4B (merged) | `weights/qwen35_4b_merged` | 8.5 G |

These were originally symlinks into `PAAS_qwen3vl` — which made `du -sh` report a small,
self-contained-*looking* project while 7 of the 8 branches actually depended on another project. One
of them pointed into `PAAS_qwen3vl/runs/`, a run-output directory that routine cleanup could delete.
`scripts/materialize_weights.sh` copied all five in (staging to `.partial` and renaming on success,
so a half-copied model can never masquerade as a complete one).

The verifier checks three things, because each hides differently: symlinks leaving the tree, absolute
paths to other projects' models in code/config, and declared paths that don't resolve in-tree. It
found 4 leftover PE-checkpoint references in `pespc/` that a path sweep had missed.

Also vendored: `pespc/` (from PAAS_simplicity, paths repointed, axon1/baseline scripts removed) and
`perception_models/` (26 MB). The old 69 GB `runs/` was not copied.

**Still external, deliberately:** the venv (`PAAS_qwen3vl/venv`) is a toolchain, not a weight; the
image corpora and the EVAL_SPACE manifests are shared datasets. The verifier exempts both by design.

## Serving: choosing the combination at inference time

`app_fastapi_json_v5_safe_jsonfmt.py` now understands **eight** components, so the combination is a
config choice rather than a code change:

```
ffaa, A1_9c, A2_9c, A3_9c, gsd, gsdA, selop, pespc
```

`gsdA` and `pespc` were added for this: previously the app rejected both
(`unknown fusion components ['gsdA','pespc']`). Only the detectors a combination actually names are
loaded, so an unused member costs nothing.

```bash
# the target combination
PAAS_COMPONENTS="ffaa,A1_9c,A2_9c,gsdA,selop,pespc" bash run_server.sh

# or persist it in the config
_cfg.fusion.components = ["ffaa","A1_9c","A2_9c","gsdA","selop","pespc"]
```

`GET /health` reports `gsdA_enabled` / `pespc_enabled`, and every response carries `gsdA_fake` and
`pespc_fake` under `Details` alongside the existing per-member scores.

**`gsd` and `gsdA` are separate slots** (`PaasConfig.gsd`, `PaasConfig.gsdA`) rather than one path
that gets swapped, so both arms can be served at once and compared. In EVAL_SPACE they were
indistinguishable inside a 6-member mean (0.999546 both) — prefer `gsdA` for provenance, since its
reference basis is not fitted on the evaluation split, not for accuracy.

**Two things to know before serving this combination.**

1. **The threshold is combination-specific.** The legacy `0.2192` was fitted for
   `mean{ffaa,A2_9c,gsd,selop}`. A different member set changes the fused score distribution, so that
   number no longer corresponds to any known real-recall floor. The app now prints a warning when the
   components differ from what the threshold was fitted for; re-fit with `train/fit_threshold.py` or
   set `PAAS_THRESHOLD`.
2. **PE-SPC roughly doubles non-MLLM inference cost.** Its encoder is 1.88 B params at 448 px
   (~47.8 img/s measured), against 336 px CLIP-L for every other member. Its default batch size is
   smaller (32 vs 64) for the same reason.

**Checkpoints required by each component** — `gsdA` and `pespc` do not exist until the corresponding
branches are trained, and requesting them before that fails loudly with `FileNotFoundError` at
startup rather than scoring silently wrong.

## Current weights: EVAL_SPACE-trained, for validation

`weights/` now holds the EVAL_SPACE-retrained detectors, copied in so the serving path could be
validated before this project trains anything:

| component | file | source |
|---|---|---|
| ffaa (MLLM) | `weights/qwen35_4b_merged` | `PAAS_ensemble_v4_inf/weights/` |
| ffaa (MIDS head) | `weights/ffaa_qwen35_mids/best.pth` | `EVAL_SPACE .../deploy/ffaa_mids.pth` |
| A1/A2/A3_9c | `weights/ensemble9/A{1,2,3}_9c.pt` | `EVAL_SPACE .../deploy/` |
| gsd | `weights/gsd/best.pt` | `EVAL_SPACE .../deploy/gsd.pt` |
| gsdA | `weights/gsdA/best.pt` | `EVAL_SPACE .../gsd_anchorswap/best.pt` |
| selop | `weights/selop/best.pt` | `EVAL_SPACE .../deploy/selop.pt` |
| pespc | `weights/pespc/best.pt` | `EVAL_SPACE .../pespc/devsel_h3_head.pt` |

**These were trained on the EVAL_SPACE deduplicated trainset (889,610 images), not the full root this
project targets.** Replace each with `$OUT_ROOT/<branch>/best.pt` after `run_finetuning.sh`.

### MLLM pairing — a mismatch that would have been silent

The MIDS 4-class head consumes a specific MLLM's answers, so the two must be paired. The merged Qwen
originally linked here came from `PAAS_qwen3vl/runs/`, but the EVAL_SPACE ffaa head was trained
against `PAAS_ensemble_v4_inf/weights/`. Those two are **not the same model**: identical size,
identical `config.json` md5, identical vision tower and embeddings — but **205 of 723 tensors differ**,
precisely the LoRA-target modules (`linear_attn.in_proj_qkv/in_proj_z/out_proj`,
`mlp.{gate,up,down}_proj`) in every language layer, max|Δ| ≈ 0.01–0.03. Two different LoRA merges onto
one base. `weights/qwen35_4b_merged` is now byte-identical to the `_inf` one.

Size and config equality are NOT sufficient to identify a merged model — compare tensor content.

### Validated end to end

```
components: ffaa, A1_9c, A2_9c, gsdA, selop, pespc   -> 18 frames, 0 errors
   real     mean fused 0.0141
   pad      mean fused 0.9998
   deepfake mean fused 0.8372
```

PE-SPC's serving path was also checked against the offline scores it must reproduce: on 256 test
images, 251 saturated scores match to 5.6e-4 and only the 5 mid-range ones move (max 0.052), with
**zero decision flips** at τ=0.2192 or τ=0.5. Running the encoder in fp32 made the gap *larger*
(0.137), which confirms the residual is the fp16-quantised feature cache the offline scores came
from, not the serving code.

## Integration review — 11 findings, all fixed

An external review found 11 integration errors. All were reproduced and fixed; the four criticals
and both mediums marked *(mine)* were introduced when the gsdA / PE-SPC / three-9c-arm branches were
added without updating the assembly, config and preflight paths.

| # | finding | fix |
|---|---|---|
| 1 *(mine)* | `make_deploy_config.py` searched only `ninec/` — a successful run kept the OLD A2 and dropped A1/A3/gsdA/PE-SPC, still reporting success | finds all 8 branches; emits **all** retrained 9c members; **refuses** when a branch that ran produced no checkpoint (`--allow-partial` to override) |
| 2 *(mine)* | `PaasConfig.from_dict` never deserialized `gsdA`/`pespc` — custom ckpt paths and `enabled=false` silently replaced by defaults | both deserialized; **unknown sections now raise** instead of being discarded |
| 3 *(mine)* | PE cache shared across runs, skipped on file existence; SMOKE wrote 64-row shards into the same cache, so a later production run trained on smoke data | cache is **per-run** (`$OUT_ROOT/pe_cache`), smoke uses its own; shards carry a **provenance sidecar** (manifest, encoder, ckpt size, row count) and reuse **refuses** on mismatch |
| 4 *(mine)* | bare `wait` returns 0 even when a child fails — a crashed extractor was reported as success | `wait` on each PID, non-zero aborts the stage |
| 5 | `X-Forwarded-For` pasted into a path — `../` or absolute escaped `APP_ROOT/images`; active by default (`SAVE_INPUT_IMAGES=1`) | header sanitised to IP characters, length-capped, plus a realpath containment check |
| M *(mine)* | preflight validated the unused legacy `NINEC_CFG` while training used A1/A2/A3 | validates every config actually used; `--ninec`/`--gsd` made **repeatable** (they were single-valued, so three flags validated only the last) |
| M *(mine)* | smoke moved only the old GPU vars, so `GPU_GSDA`/`GPU_PESPC` still spanned GPUs 0–3 | both included |
| M | 4 bundled experiment configs unloadable | `ensemble_only`/`ffaa_only`/`weighted_ffaa` rewritten onto the supported schema; `per_model_or` retired to `legacy_v3/` (its OR mode is not implemented in `paas/fusion.py`) |
| M | `inference.py` docs showed multiple/positional images, parser took one `--images` | `nargs="+"` plus a positional alias |
| M | `--device cpu` advertised, every model pinned to `cuda:0` | honours `cfg.device`; **refuses** cpu + ffaa with a clear message (vLLM needs CUDA) |
| M | weighted fusion accepted `[1,-1]` → NaN | rejects negative weights and non-positive sums |

Not re-checked by that review, and still true: the **train/eval overlap is 100% of es_test and 96.9%
of es_val**, by explicit instruction. See the data-policy section above.

## Default serving combination

`run_server.sh` now defaults to `config/experiments/paas_v4full_default.json`:

```
mean{ ffaa, A1_9c, A2_9c, gsdA, selop, pespc }      threshold 0.200609
```

`gsd` is disabled and `gsdA` used in its place — the two are indistinguishable inside a 6-member mean
(test AUC 0.999546 both), so the cleaner anchor provenance decides it. The same list is now the
`FusionCfg.components` dataclass default, so `PaasConfig()` and the config file agree.

Change the set without touching code:

```bash
bash run_server.sh                                            # the 6-member default
PAAS_COMPONENTS="ffaa,A2_9c,gsdA,selop" bash run_server.sh    # drop PE-SPC (halves non-MLLM cost)
PAAS_COMPONENTS="A1_9c,A2_9c,gsdA,selop,pespc" bash run_server.sh   # MLLM-free
PAAS_CONFIG=config/experiments/paas4_qwen.json bash run_server.sh   # the old v4 4-member set
```

### The threshold was re-fitted, and not at the usual floor

The inherited 0.2192 was fitted for `mean{ffaa,A2_9c,gsd,selop}` and means nothing for this set. Fitted
fresh on es_val (50,384 images) with the EVAL_SPACE-trained weights:

| val floor | tau | test real | test fake | test pad | test deepfake |
|---|---|---|---|---|---|
| 0.90 | 0.012901 | **0.7920** | 0.9998 | 0.9999 | 0.9997 |
| 0.95 | 0.037108 | 0.8824 | 0.9995 | 0.9996 | 0.9992 |
| 0.98 | 0.125509 | 0.9476 | 0.9983 | 0.9986 | 0.9976 |
| **0.99** | **0.200609** | **0.9768** | **0.9968** | 0.9974 | 0.9952 |

The default is the **99%** floor, not the project's usual 90%. This fusion's thresholds transfer
poorly from val to test and the gap widens as the floor loosens — shipping the val90 tau would have
rejected ~21% of real faces on test. **These taus are only valid for the weights in `weights/` now;
re-fit with `train/fit_threshold.py` after `run_finetuning.sh`.**

### Cost

Measured with all six loaded on one GPU: **63,645 MiB**. vLLM takes `ffaa.qwen_gpu_mem` (0.45) of the
card and the five CLIP/PE detectors take the rest. On a smaller GPU, lower `qwen_gpu_mem` or drop
`pespc` (448 px on a 1.88 B encoder — roughly half the non-MLLM cost).

### run_server.sh / stop_server.sh

Verified end to end: server up in ~80 s, `/batcher_status` reports all six enabled, a real image
scores 0.0–0.02 across every member and a deepfake 0.98–1.00, and `stop_server.sh` returns the GPU
from 63,645 MiB to 66 MiB.

Fixes made to the stop path:
* `[ "$FORCE" = "1" ] && kill -9 … || kill …` fell through to the graceful kill whenever `kill -9`
  failed (`a && b || c`). Replaced with a real if/else.
* Added a `[V]LLM::Worker` pattern for tensor-parallel deployments.
* Added a GPU-release check — freeing the GPU is the point of the script, so it now verifies rather
  than just printing numbers. Thresholded at 256 MiB (`STOP_WARN_MIB`) so unrelated jobs holding
  ~12 MiB don't train you to ignore the warning.

All six detectors load **inside the uvicorn process**; only vLLM forks. Adding detectors does not add
processes to kill.

## Integration review round 2 — 12 findings, all reproduced and fixed

An external review flagged 12 issues before the full training run. Every one was reproduced
first (none were taken on description alone) and every one was real. Fixes and their evidence:

| # | Sev | Issue | Fix | Verified by |
|---|-----|-------|-----|-------------|
| 1 | High | `LIM=(--limit 64)` spliced into `bash -c`: `--limit` joined the script, `64` became `$0`, `for` loop unterminated → **SMOKE=1 could never pass PE-SPC** | scalar `PE_LIMIT` (`--limit 0` = no limit) | argv dump, then a real `SMOKE=1 RUN_PESPC=1` run: 5/5 stages OK |
| 2 | High | deploy inherited `fusion.components` from `paas4_qwen.json` → a run that trained 8 detectors **deployed 4** | `DEPLOY_BASE` + `--base`, and the tool's own default, now `paas_v4full_default.json` | synthetic run dir → components = the 6-member set |
| 3 | High | missing `TESTSET_DIR` printed "calibrate skipped" and **exited 0** with a stale threshold | records `STATUS[calibrate]=1` | `TESTSET_DIR=/nonexistent` → exit **1** |
| 4 | High | `pespc_protos` / `pespc_feat_*` failures did not block deploy; with a reused `PE_CACHE` the head trains on the **previous run's** `.npz` | added them to the gate **and** made `train_pespc.py` verify the provenance sidecars it was ignoring | `PE_CKPT=/nonexistent` → deploy blocked; stale cache → refused |
| 5 | Med | `GSD_FAITHFUL=1` applied the flag to **every** `--gsd` config, so the default `RUN_GSD=1 + RUN_GSDA=1 + GSD_FAITHFUL=1` could not pass preflight | `--gsd-faithful` names which config gets the flag | 4/4 combinations correct; both true mismatches still caught |
| 6 | Med | `--device cuda:2` masked GPU 2 then asked torch for **ordinal 2 of 1 visible device** | `env.setup()` returns the post-mask device; out-of-range ordinals refused with a readable message | 6-case matrix |
| 7 | Med | `run_config.json` written **before** the smoke overrides → recorded production datasets/GPUs for smoke runs; omitted A1/A2/A3, gsdA, PE-SPC | moved after the smoke block, added the missing sections | smoke run records `smoke=1`, smoke trainset, GPU 0, `smoke9.yaml` |
| 8 | Med | batcher checked `total` **before** dequeuing, then appended the whole request: cap 8 → **15 images dispatched** | `carry` slot; oversized single requests still served whole | harness: was 15, now `[7, 8]`, nothing dropped |
| 9 | Low | `inference.py` CLI defaulted to the legacy 4-member config, unlike the server | same default as `run_server.sh` | all 8 configs load + validate |
| 10 | Low | smoke substituted `NINEC_CFG`, but training reads `A1_CFG`/`A2_CFG`/`A3_CFG` → "smoke" ran **three full-size** 9-class trainings | substitutes all three | 10s per 9c branch |
| 11 | Low | `mean.json` omitted `components`, silently inheriting the new 6-member default with its **old** threshold | components listed explicitly (reconstructed — see the note in the file) | — |
| 12 | Low | `threshold` of -1/2/NaN and NaN/inf weights all validated | finiteness + range checks | 8 bad values refused, boundaries 0.0/1.0 still accepted |

**One issue the review did not name, found while fixing #2:** a fused component with no checkpoint
from this run silently serves the *previously deployed* weights. That is often intended (the
standing `RUN_QWEN=0` leaves `ffaa` inherited), so `make_deploy_config.py` now **warns** and records
`_serving_previous_weights` rather than refusing.

## Integration review round 3 — 8 findings, all reproduced and fixed

| # | Sev | Issue | Fix | Verified by |
|---|-----|-------|-----|-------------|
| 1 | **High** | `check_provenance` required every shard's `n_items` to be equal, but extraction shards round-robin — 1,366,146 over 4 GPUs is `[341537, 341537, 341536, 341536]`. **The default production PE-SPC stage would have failed after paying for the full extraction.** | dropped `n_items` from the equality set; kept the properties round-robin *does* guarantee (shard ids exactly `0..n-1`, sizes within 1) | 4 valid layouts accepted, 6 broken ones refused; then a real 3-shard `[14,13,13]` extraction + train |
| 2 | **High** | `PESPC_BS=128` vs 40 smoke records → `steps_per_ep = 40 // 128 = 0`: smoke **saved the untrained head and reported success**. And smoke9.yaml (`artifact_enabled: false`) was substituted for A3, so its ForensicsAdapter head was never built | `PESPC_BS=8` in smoke; new `smoke9_artifact.yaml` for A3 | `steps=0 scale=1.0` → `steps=10`, loss moving, `scale=0.233` |
| 3 | Med | deploy rebuilt `ensemble9.json` with only this run's members while `fusion.components` still asked for the others — `Ensemble9Model` **raises**, so the build was dead on arrival. `ALLOW_PARTIAL` also never reached `--allow-partial` | refuse by default; under `--allow-partial`, drop the untrained members and say so; wired `ALLOW_PARTIAL` through | A1-skipped/A2-retrained case: refused (exit 1), then dropped cleanly |
| 4 | Med | `DEPLOY_FLOOR=0.90` silently reverted the 0.99 operating policy the maintained config ships | default now `0.99`, with the transfer numbers in the comment | — |
| 5 | Med | `PE_CKPT`/`PE_MODEL` reached the extractor but **not** `text_prototypes.py`; the head stored no encoder identity; deploy always pinned the default. A same-width checkpoint mismatched silently | prototypes honour both env vars and record `model_name`/`path`/`size`; trainer checks them; head carries the fingerprint; deploy pins **from the head** | prototypes regenerated bit-identical + new fields; 3 mismatch cases refused/warned |
| 6 | Med | `aggregate_trimmed` refused only at **zero** valid frames, so 1 of 5 frames gave a confident 5-frame-looking verdict; 3 valid trimmed down to 1 score | `MIN_VALID_FRAMES` (default 3); trim only when it still leaves that many; `n_valid`/`degraded`/`trimmed` in the response | 6-case matrix |
| 7 | Low | the ONNX rotation model was built at import (initialising CUDA **before** vLLM → forced `spawn`) and ran per-frame even when `IGNORE_ROTATION_CHECK=1`, the default | lazy construction; inference skipped entirely when the result cannot matter | server start |
| 8 | Low | README called `0.2192` the default | corrected to "legacy" | — |

**Finding 1 was mine**, introduced by the round-2 provenance check: the smoke test that would have
caught it runs on one GPU, where round-robin sharding is trivially equal. The multi-shard test above
now exists so it cannot recur. **Finding 2 was also visible in round 2** — `steps=0` was printed in
that session's own smoke output and went unread.

### Two corrections to the round-3 review's diagnoses

**The vLLM `spawn` fallback is not the app's doing.** The review attributed it to the ONNX rotation
detector and `torch.cuda.is_available()` initialising CUDA before the engine. Probed with vLLM's own
predicate (`vllm.utils.system_utils.cuda_is_initialized`), neither does: importing torch, importing
`Yolo11ClsONNX`, calling `torch.cuda.is_available()`, setting `allow_tf32`, importing transformers
and importing the ffaa utils all leave it `False`. A script containing *only* `env.setup("cuda:0")`
plus `LLM(...)` — no app, no ONNX — prints `cuda_is_initialized = False` immediately before the call
and still emits the warning. **vLLM initialises CUDA itself during engine construction**; the
ordering `paas/env.py` guarantees is intact and there is nothing here to fix. Do not chase it again.

The rotation-detector change was still made, on its own merits: the ONNX session is now built lazily
and the per-frame inference is skipped entirely under `IGNORE_ROTATION_CHECK=1` (the default), where
its result was computed and then discarded.

**`config/experiments/paas_v4full_default.json` was hand-edited at 09:22:42**, swapping gsdA→gsd
while leaving the `_note` describing gsdA. No code in the project writes that path. Reverted to gsdA
on the user's instruction, and `_stale_after_retrain` restored.

## Promotion — 2026-08-31

The full-trainset run (`runs/finetune_20260825_104051`) completed all 18 stages with **zero
failures** in ~5d21h, and its deploy build was promoted into `weights/` and `config/`.

| branch | time | val metric |
|---|---|---|
| A1_9c / A2_9c / A3_9c | 3h01m / 2h58m / 2h56m | acc 0.9975 / 0.9987 / 0.9987 |
| gsd / gsdA | 5h22m / 5h20m | bin_auc 0.9993 / 0.9993 |
| selop | 1h24m | bin_auc 0.9998 |
| PE-SPC | 2h28m | bin_auc 0.9997 |
| qwen_lora / distill / merge | 48h07m / 5h17m / 20s | eval_loss 0.1610 → 0.1316 |
| mids 4-class | 8h27m | — |

**Deployed threshold.** `train/fit_threshold.py` on `TESTSET_DIR` (30,218 frames: 10,126 real /
20,092 fake), AUC 1.0000:

| real floor | τ | real recall | fake recall |
|---|---|---|---|
| 90% | 0.0168 | 90.04% | 100.00% |
| 95% | 0.0386 | 95.01% | 100.00% |
| 98% | 0.0960 | 98.01% | 100.00% |
| **99%** | **0.14085** | **99.00%** | **100.00%** ← deployed |

**AUC 1.0000 deserves scrutiny, not celebration.** Perfect separation on 30k frames is more often a
property of the data than of the model. Confirmed so far: the testset is NOT under the trainset root
(`/temp/testset` vs `/data/no_delete_mids_train`) and was never used for model selection. NOT yet
confirmed: that the two share no content. Under the project's explicit no-dedup policy the same
source videos could appear in both, which would inflate this number. Verify before quoting it.

**Rollback.** Everything replaced is in `weights/_prev_20260831/` (2.9 GB of detectors + the 8.5 GB
previous MLLM + both configs). Restore by copying back and setting `decision.threshold` to 0.200609.

**Serving scripts needed no path changes.** `run_server.sh` resolves weights through `PAAS_CONFIG` +
`paas/env.py` conventions (`weights/<branch>/best.pt`), so promoting in place is sufficient.
`stop_server.sh` kills by process pattern and is weights-independent.
