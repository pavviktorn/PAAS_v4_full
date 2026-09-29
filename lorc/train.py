#!/usr/bin/env python3
"""Train LoRC (arXiv:2608.20882v1) on the DEV-held-out MIDS trainset.

REPORTING CONTRACT -- the same one train_dinospc.py follows, and the reason this project's numbers
can be compared at all:

  TRAIN on    PAAS_v4_full/runs/manifests_devheld/spc3_train.json   (1,257,404 rows; the trainset
              with the EVAL_SPACE DEV holdout removed by content hash)
  SELECT on   EVAL_SPACE/manifests/es_dev_sel_c99.json              (35,568 rows)
  REPORT on   EVAL_SPACE/manifests/es_dev_eval_c99.json             (35,171 rows; DEV rows with
              every >=0.99-cosine near-duplicate of the trainset or the selector removed)

THE SELECTOR IS THE CERTIFIED ONE, NOT es_dev_sel.json. The raw selector has 17,876 of its 53,444
rows (33.448%) within 0.99 cosine of a trainset image, so a checkpoint chosen on it is chosen
partly on how well it memorised the trainset -- and unevenly, since the cut costs PAD 52% of its
rows against deepfake's 14%. The report split was already cleaned this way; cleaning only that side
is the failure EVAL_SPACE/eval_c99.py warns about in its own docstring. scripts/
build_certified_selector.py builds the split and records the provenance. Pointing sel_manifest back
at the raw file is refused below rather than merely warned about.

The report split is NOT touched here. train.py only ever evaluates on the selector, and predict.py
scores the report split afterwards -- so no checkpoint can be chosen, and no epoch stopped, on a
number from it. If both were available in this loop, "select on sel" would be an honour system.

  usage:  ./run_train.sh                        # 4-GPU DDP, config.json
          ./run_train.sh --epochs 1 --limit-train 20000
          SMOKE=1 ./run_train.sh                # configs/lorc_smoke.json
"""

# Hand off to the project interpreter before any third-party import (see lorc/_interp.py).
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
                 if _os.path.basename(_os.path.dirname(_os.path.abspath(__file__)))
                 in ("scripts", "tools") else
                 _os.path.dirname(_os.path.abspath(__file__)))
from lorc._interp import ensure_interpreter                                # noqa: E402
ensure_interpreter()

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lorc import paths as P
from lorc import metrics as M
from lorc import metrics3 as M3                                            # noqa: E402
from lorc.data import (LoRCDataset, build_transforms, class_counts,       # noqa: E402
                       class_weights, load_manifest)
from lorc.engine import (evaluate, is_dist, log0, make_eval_loader,       # noqa: E402
                         make_train_loader, rank0, train_epochs)
from lorc.model import LoRCModel, build_encoder                          # noqa: E402

AMP = {"bf16": torch.bfloat16, "fp16": torch.float16, "none": None, "fp32": None}


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json")
    # Anything in the config can be overridden on the command line; only the knobs worth reaching
    # for from a shell are listed, and --set handles the rest so the config stays the record.
    # action="extend", NOT the default "store". With plain nargs="*" a second --set REPLACES the
    # first, so `--set epochs=1 epoch_steps=4000 --set out_dir=X` silently drops the budget
    # overrides -- which is exactly how run_train.sh composes the screen tier with an out_dir
    # override. default=None (not []) because _ExtendAction mutates the default list in place,
    # which leaks values between parses in the same process.
    ap.add_argument("--set", action="extend", nargs="*", default=None, metavar="KEY=VALUE")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--lambda-ss", type=float)
    ap.add_argument("--rank", type=int)
    ap.add_argument("--crop-mode", choices=["resize", "paper", "resize_crop"])
    # Capacity knobs get first-class flags so run_sweep.sh can select them by measurement rather
    # than inheriting them as a prior; --set would need a nested JSON blob on the command line.
    ap.add_argument("--lora-r", type=int)
    ap.add_argument("--lora-alpha", type=int)
    ap.add_argument("--lora-targets", help="comma list, e.g. q_proj,k_proj,v_proj,o_proj")
    ap.add_argument("--ssl-source", choices=["attn_out", "attn_latent", "residual"],
                    help="which tensor Eq 6-7 operates on; see LoRCModel.ssl_features")
    ap.add_argument("--ssl-mode", choices=["binary", "pairwise", "real_vs_deepfake"])
    ap.add_argument("--head-input")
    ap.add_argument("--limit-train", type=int, default=0)
    ap.add_argument("--limit-eval", type=int, default=0)
    ap.add_argument("--out-dir")
    ap.add_argument("--warm-start", "--resume", dest="resume",
                    help="load new-module + LoRA weights from a checkpoint. WEIGHTS ONLY -- not "
                         "optimizer/scheduler/RNG/epoch state, so it is not a crash-resume.")
    return ap.parse_args()


def _slug(metric):
    return metric.replace("@", "_at_").replace("/", "_")


def ckpt_name(metric):
    """Filename for a tracked metric's own best checkpoint. One rule, used by train and compare."""
    return "best_" + _slug(metric) + ".pt"


def scores_name(metric):
    """Selector scores belonging to that checkpoint -- what the fusion table must read."""
    return "dev_sel_scores_" + _slug(metric) + ".npy"


def merge_config(a):
    cfg = json.load(open(a.config))
    for k in ("epochs", "lr", "batch_size", "lambda_ss", "rank", "crop_mode", "out_dir",
              "ssl_source", "ssl_mode", "head_input"):
        v = getattr(a, k.replace("-", "_"), None)
        if v is not None:
            cfg[k] = v
    lora = cfg.setdefault("lora", {})
    if a.lora_r is not None:
        lora["r"] = a.lora_r
        # alpha tracks r unless it is set explicitly: LoRA's update is scaled by alpha/r, so
        # changing r alone silently changes the effective step size too, which would confound a
        # capacity comparison with a learning-rate one.
        lora["alpha"] = a.lora_alpha if a.lora_alpha is not None else a.lora_r
    elif a.lora_alpha is not None:
        lora["alpha"] = a.lora_alpha
    if a.lora_targets:
        lora["target_modules"] = [t for t in a.lora_targets.split(",") if t]

    cli = set()
    for k in ("epochs", "lr", "batch_size", "lambda_ss", "rank", "crop_mode", "out_dir",
              "ssl_source", "ssl_mode", "head_input"):
        if getattr(a, k.replace("-", "_"), None) is not None:
            cli.add(k)
    for kv in (a.set or []):
        k, _, v = kv.partition("=")
        cli.add(k)
        if not _:
            raise SystemExit(f"--set expects KEY=VALUE, got {kv!r}")
        try:
            cfg[k] = json.loads(v)
        except json.JSONDecodeError:
            cfg[k] = v
    # WHICH KEYS THE CALLER SET EXPLICITLY. Needed because epoch_images derives epoch_steps, and
    # an explicit --set epoch_steps=N from a caller must beat a default epoch_images from the
    # config file -- otherwise a reduced-budget sweep silently trains at full budget.
    cfg["_cli_keys"] = sorted(cli)
    # RELATIVE PATHS RESOLVE AGAINST THE PROJECT ROOT, not the shell's cwd. The configs ship
    # relative paths so the project can be moved or copied; resolving them here means every
    # consumer downstream still sees an absolute path and nothing has to care.
    for k in ("encoder", "train_manifest", "sel_manifest", "report_manifest", "sel_ab_index",
              "out_dir"):
        if cfg.get(k):
            cfg[k] = P.resolve(cfg[k])
    return cfg


def main():
    a = parse()
    cfg = merge_config(a)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
    dev = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    world = dist.get_world_size() if is_dist() else 1
    torch.manual_seed(int(cfg.get("seed", 0)) + (dist.get_rank() if is_dist() else 0))
    amp_dtype = AMP[str(cfg.get("amp_dtype", "bf16"))]
    t0 = time.time()

    out_dir = cfg["out_dir"]
    if rank0():
        os.makedirs(out_dir, exist_ok=True)
    log0(f"[lorc] === LoRC (arXiv 2608.20882v1) ===")
    log0(f"[lorc] world={world} device={dev} amp={cfg.get('amp_dtype')} out={out_dir}")

    # ------------------------------------------------------------------ data
    tr = load_manifest(cfg["train_manifest"], cfg.get("label_source", "manifest"), log=log0)
    if os.path.basename(cfg["sel_manifest"]) == "es_dev_sel.json":
        raise SystemExit(
            "[lorc] REFUSING sel_manifest=es_dev_sel.json: 33.448% of its rows are within 0.99 "
            "cosine of a trainset image, so it selects partly on memorisation, and unevenly by "
            "class (PAD -52%, deepfake -14%). Use es_dev_sel_c99.json -- build it with "
            "scripts/build_certified_selector.py.")
    sel = load_manifest(cfg["sel_manifest"], cfg.get("label_source", "manifest"), log=log0)

    # A/B/C carve of the certified selector, disjoint by capture group:
    #   A (0) chooses the checkpoint and ranks the configs   <- this file
    #   B (1) chooses which arm joins the fusion             <- compare_configs.py
    #   C (2) fits tau and is the final diagnostic           <- run_predict.sh
    # Loaded whenever it exists so all parts are always recorded; `use_ab_split` decides only which
    # part the CHECKPOINT is chosen on.
    ab = None
    ab_path = cfg.get("sel_ab_index")
    if ab_path and os.path.exists(ab_path):
        ab = np.load(ab_path)
        if len(ab) != len(sel):
            log0(f"[lorc] sel_ab_index has {len(ab):,} rows but the selector has {len(sel):,}; "
                 f"ignoring the A/B carve")
            ab = None
    use_ab = bool(cfg.get("use_ab_split", False))
    if use_ab and ab is None:
        raise SystemExit(f"[lorc] use_ab_split=true but {ab_path} is missing or mismatched.")
    if a.limit_train:
        tr = tr[:: max(1, len(tr) // a.limit_train)][:a.limit_train]
        log0(f"[lorc] --limit-train: {len(tr):,} rows (strided, so every class survives)")
    n_sel_full = len(sel)
    if a.limit_eval:
        sel = sel[:: max(1, len(sel) // a.limit_eval)][:a.limit_eval]
        log0(f"[lorc] --limit-eval: {len(sel):,} rows (strided)")
        if ab is not None:
            ab = ab[:: max(1, n_sel_full // a.limit_eval)][:a.limit_eval]
    htr, hsel = class_counts(tr), class_counts(sel)
    log0(f"[lorc] train real={htr[0]:,} pad={htr[1]:,} deepfake={htr[2]:,}")
    log0(f"[lorc] sel   real={hsel[0]:,} pad={hsel[1]:,} deepfake={hsel[2]:,}")

    size = int(cfg.get("image_size", 224))
    ttf = build_transforms(True, cfg.get("crop_mode", "paper"), size,
                           int(cfg.get("resize_short", 256)), bool(cfg.get("aug_hflip", True)),
                           int(cfg.get("jpeg_qf", 0)), degrade=cfg.get("degrade"))
    etf = build_transforms(False, cfg.get("crop_mode", "paper"), size,
                           int(cfg.get("resize_short", 256)), False, int(cfg.get("jpeg_qf", 0)))
    ds_tr = LoRCDataset(tr, ttf, size)
    ds_sel = LoRCDataset(sel, etf, size)

    if int(cfg.get("num_classes", 3)) != 3:
        raise SystemExit(f"[lorc] num_classes must be 3 (real/pad/deepfake); config says "
                         f"{cfg.get('num_classes')}")
    bs = int(cfg["batch_size"])
    workers = int(cfg.get("num_workers", 8))

    # BUDGET IS IN IMAGES, NOT MICROBATCHES. epoch_steps counts microbatches PER RANK, so it means
    # a different amount of training for every arm that changes batch_size or world size: the 7B
    # arm runs batch_size 8 against the default's 16, and at a shared epoch_steps=20000 it saw
    # 640k images and 2,500 optimizer steps against 1.28M and 5,000 -- HALF the budget, in the one
    # comparison whose entire purpose is "is the bigger backbone worth it". `epoch_images` is
    # invariant to both, so every arm gets the same budget unless it deliberately asks not to.
    cli_keys = set(cfg.get("_cli_keys") or [])
    if cfg.get("epoch_images") and not ("epoch_steps" in cli_keys
                                        and "epoch_images" not in cli_keys):
        # EXPLICIT epoch_steps WINS over a config-file epoch_images. Without this precedence the
        # derivation silently overwrote a caller's reduced budget: run_sweep.sh passed
        # `--set epoch_steps=4000` and every point trained on the config's 1,280,000 images
        # instead -- 5x the advertised budget, ~20 selector reads instead of 4, and a "screen"
        # label on what was actually a full run. run_train.sh escaped only because its screen tier
        # happens to set BOTH keys.
        want = int(cfg["epoch_images"])
        derived = max(want // max(bs * world, 1), 1)
        if cfg.get("epoch_steps") and int(cfg["epoch_steps"]) != derived:
            log0(f"[lorc] epoch_images={want:,} overrides epoch_steps="
                 f"{int(cfg['epoch_steps']):,} -> {derived:,} microbatches/rank")
        cfg["epoch_steps"] = derived
        log0(f"[lorc] budget: {want:,} images/epoch = {derived:,} microbatches/rank x {bs} x "
             f"{world} ranks ({derived // max(int(cfg.get('grad_accum', 1)), 1):,} optimizer steps)")
    elif "epoch_steps" in cli_keys:
        st = int(cfg["epoch_steps"])
        log0(f"[lorc] budget: epoch_steps={st:,} set explicitly (overrides epoch_images="
             f"{cfg.get('epoch_images')}) = {st * bs * world:,} images/epoch, "
             f"{st // max(int(cfg.get('grad_accum', 1)), 1):,} optimizer steps")
    loader_tr, dsampler = make_train_loader(
        ds_tr, tr, bs, workers, balanced=bool(cfg.get("balanced_sampler", True)),
        seed=int(cfg.get("seed", 0)), epoch_steps=cfg.get("epoch_steps"))
    loader_sel, _ = make_eval_loader(ds_sel, int(cfg.get("eval_batch_size", bs)), workers)

    # ------------------------------------------------------------------ model
    enc, hidden, nreg, n_enc_tr = build_encoder(
        cfg["encoder"], dtype=torch.bfloat16 if cfg.get("amp_dtype") == "bf16" else torch.float32,
        lora=cfg.get("lora"), grad_checkpointing=bool(cfg.get("grad_checkpointing", False)),
        log=log0)
    model = LoRCModel(enc, hidden, nreg, rank=int(cfg.get("rank", 32)),
                      head_input=cfg.get("head_input", "attn+cls"),
                      head_norm=cfg.get("head_norm", "none"),
                      detach_cls=bool(cfg.get("detach_cls", False)),
                      attn_dropout=float(cfg.get("attn_dropout", 0.0)),
                      head_dropout=float(cfg.get("head_dropout", 0.0)),
                      ssl_source=cfg.get("ssl_source", "attn_out")).to(dev)
    enc_p, new_p = model.trainable_groups()
    n_new = sum(p.numel() for p in new_p)
    log0(f"[lorc] encoder hidden={hidden} n_register={nreg} patch_tokens_start={model.patch0} "
         f"n_patches={(size // int(cfg.get('patch_size', 16))) ** 2}")
    if cfg.get("degrade") and float(cfg["degrade"].get("p", 0)) > 0:
        log0(f"[lorc] degradation aug (TRAIN ONLY, class-blind): {cfg['degrade']}")
    log0(f"[lorc] L_SS on {model.ssl_source!r}, grouping {cfg.get('ssl_mode', 'binary')!r}, "
         f"lambda_ss={cfg.get('lambda_ss')}, head_input={model.head_input!r} "
         f"(head in={model.head.in_features})")
    log0(f"[lorc] trainable: LoRA {n_enc_tr:,} + new {n_new:,} = {model.n_trainable():,} "
         f"(low-rank attn {model.lra.n_trainable():,} at rank {cfg.get('rank', 32)}, "
         f"head {sum(p.numel() for p in model.head.parameters()):,})")

    if a.resume:
        # NAMED HONESTLY: this is a warm start, not a resume. The checkpoint holds the new modules
        # and the LoRA adapters -- no optimizer moments, no scheduler position, no RNG state, no
        # epoch counter, no best-metric-so-far. Continuing a run with it restarts the cosine
        # schedule from step 0 with fresh Adam moments, which is a DIFFERENT trajectory, not a
        # continuation. Kept because warm-starting is useful; renamed so it cannot be mistaken.
        sd = torch.load(a.resume, map_location="cpu")
        missing, unexpected = model.load_state_dict(sd["state"], strict=False)
        bad = [k for k in missing if not k.startswith("encoder.") or "lora_" in k]
        if bad or unexpected:
            raise SystemExit(f"[lorc] --warm-start checkpoint does not match this architecture: "
                             f"unexpected={unexpected[:4]} missing(non-frozen)={bad[:4]}")
        log0(f"[lorc] WARM START from {a.resume} -- weights only. Optimizer state, LR schedule "
             f"position, RNG and epoch counter are NOT restored; this is not a crash-resume.")

    if is_dist():
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], find_unused_parameters=False)

    cw = (class_weights(tr, dev) if cfg.get("class_weight", False) else None)
    if cw is not None:
        log0(f"[lorc] class weights {cw.tolist()}")

    groups = [{"params": new_p, "lr_mult": 1.0}]
    if enc_p:
        groups.append({"params": enc_p, "lr_mult": float(cfg.get("encoder_lr_mult", 1.0))})
    opt = torch.optim.AdamW(groups, lr=cfg["lr"], weight_decay=float(cfg.get("weight_decay", 0.0)),
                            betas=tuple(cfg.get("betas", (0.9, 0.999))))

    epochs = int(cfg.get("epochs", 1))
    steps_per_epoch = max(len(loader_tr) // max(int(cfg.get("grad_accum", 1)), 1), 1)
    total_steps = steps_per_epoch * epochs
    log0(f"[lorc] {steps_per_epoch:,} optimizer steps/epoch x {epochs} = {total_steps:,} "
         f"(batch {bs}/rank x {world} ranks x accum {cfg.get('grad_accum', 1)} = "
         f"{bs * world * int(cfg.get('grad_accum', 1))} images/step)")

    # ------------------------------------------------------------------ loop
    hist, best, tracked = [], None, {}
    sel_metric = cfg.get("select_metric", "fake_rec@real99")
    M3.check_selectable(sel_metric)
    # Extra metrics that get their OWN best checkpoint. Cheap: a checkpoint here is new modules +
    # LoRA only (~5.4M params, ~22 MB), not the frozen 3.2 GB backbone.
    track = [m for m in cfg.get("track_metrics", []) if m != sel_metric]
    for m in track:
        M3.check_selectable(m)
    if track:
        log0(f"[lorc] track_metrics: {track} -> separate best checkpoints "
             f"({', '.join(ckpt_name(m) for m in track)})")
    # SELECT_METRIC IS BINARY BY DEFAULT, ON PURPOSE -- and that is a decision worth restating.
    # The deployed objective is a BINARY fusion thresholded at tau=0.189821 fitted at a 99%
    # real-recall floor, so fake_rec@real99 is what ships. But this model is required to be
    # 3-class, and that metric cannot see whether a detected fake was called PAD or deepfake: a
    # checkpoint that flags everything correctly while being unable to tell the two apart scores
    # identically to one that separates them perfectly (measured: 1.000000 vs 1.000000 on
    # fake_rec@real99, 0.50 vs 0.90 on pad_df_sep@real99).
    #   * deploying into the binary fusion  -> keep fake_rec@real99 (the default)
    #   * deploying a genuine 3-class model -> select_metric = bal_acc3@real99
    # bal_acc3@realT is the balanced accuracy of the DEPLOYABLE RULE at tau (below -> real, above
    # -> argmax(pad, deepfake)). Do NOT use pad_df_sep or any product of it: it is conditioned on
    # the fakes the model happened to detect, so detecting fewer can raise it. metrics3 refuses it.
    #
    # Both families are recorded for every run, and `track_metrics` additionally saves a SEPARATE
    # best checkpoint per listed metric (best_<metric>.pt), so the alternative is recoverable
    # rather than merely inspectable. Recording a metric is NOT enough to recover its winner --
    # only a saved checkpoint is.

    def run_eval(step):
        nonlocal best
        s, y, p, unread = evaluate(model, loader_sel, amp_dtype, len(sel), want_probs=True)
        blk = M3.merged_block(s, y, p)
        # Both halves are always measured, so the cost of the carve is visible rather than argued
        # about; only `use_ab_split` decides which half the checkpoint is chosen on.
        blk_a = blk_b = blk_c = None
        if ab is not None:
            ia, ib, ic = ab == 0, ab == 1, ab == 2
            blk_a = M3.merged_block(s[ia], y[ia], p[ia] if p is not None else None)
            blk_b = M3.merged_block(s[ib], y[ib], p[ib] if p is not None else None)
            if ic.any():
                blk_c = M3.merged_block(s[ic], y[ic], p[ic] if p is not None else None)
        sel_blk = blk_a if (use_ab and blk_a is not None) else blk
        if sel_metric not in sel_blk:
            raise SystemExit(
                f"[lorc] select_metric={sel_metric!r} is not a metric this block produces. "
                f"Selecting on a missing key would compare nan to nan and keep the FIRST "
                f"checkpoint forever. Available: "
                f"{sorted(k for k, v in sel_blk.items() if isinstance(v, float))}")
        val = sel_blk.get(sel_metric, float("nan"))
        hist.append({"step": step, "unreadable": unread,
                     **{k: v for k, v in blk.items() if not isinstance(v, list)},
                     **({"A_" + sel_metric: blk_a.get(sel_metric),
                         "B_" + sel_metric: blk_b.get(sel_metric)} if blk_a else {})})
        log0(f"[lorc] SEL step {step:>6} bin_auc={blk['bin_auc']:.6f} "
             f"fake_rec@98={blk['fake_rec@real98']:.6f} fake_rec@99={blk['fake_rec@real99']:.6f} "
             f"eer={blk['eer']:.6f} [{'A:' if use_ab else ''}{sel_metric}={val:.6f}]"
             + (f" A={blk_a[sel_metric]:.6f} B={blk_b[sel_metric]:.6f}" if blk_a else "")
             + (f" UNREADABLE={unread}" if unread else ""))
        if unread:
            raise SystemExit(f"[lorc] REFUSING: {unread} selector images could not be read. The "
                             f"metric block would be computed on grey placeholders.")
        def save_ckpt(fname, metric, value):
            core = model.module if hasattr(model, "module") else model
            torch.save({"cfg": cfg, "select": {"metric": metric, "value": float(value),
                                               "step": step},
                        "encoder": os.path.abspath(cfg["encoder"]),
                        "hidden_size": hidden, "n_register_tokens": nreg,
                        # Only the new modules and the LoRA adapters are saved: the frozen
                        # backbone is 26 GB and is identified by path, not copied.
                        "state": {k: v for k, v in core.state_dict().items()
                                  if (not k.startswith("encoder.")) or ("lora_" in k)}},
                       os.path.join(out_dir, fname))

        # A NON-FINITE VALUE CANNOT BE A BEST. `best is None or val > best["value"]` accepted a NaN
        # first evaluation as the incumbent, and every later comparison `val > nan` is False -- so
        # the FIRST checkpoint would be shipped no matter what followed, silently. A metric that
        # cannot be computed means the eval set is wrong, so say so rather than limp on.
        if not math.isfinite(val):
            log0(f"[lorc] WARNING: {sel_metric}={val} (non-finite) at step {step}; not eligible to "
                 f"be best. Usually a class is absent from the evaluation rows.")
        elif best is None or val > best["value"]:
            best = {"value": float(val), "step": step, "metric": sel_metric, "block": blk,
                    "block_A": blk_a, "block_B": blk_b, "block_C": blk_c,
                    "scores": s, "labels": y, "probs": p}
            if rank0():
                save_ckpt("best.pt", sel_metric, val)
                log0(f"[lorc]   new best {sel_metric}={val:.6f} -> {out_dir}/best.pt")

        # SEPARATE CHECKPOINTS FOR THE TRACKED METRICS. Recording a metric in lorc.json says how it
        # BEHAVED; it does not let you recover the checkpoint that metric would have chosen, since
        # by then the weights are gone. Only a saved file does that, and one costs ~22 MB here
        # (new modules + LoRA, not the frozen backbone).
        for m in track:
            v = sel_blk.get(m, float("nan"))
            if not math.isfinite(v):            # e.g. probs absent, or a class missing
                continue
            cur = tracked.get(m)
            if cur is None or v > cur["value"]:
                # THE WHOLE BLOCK AND THE SCORES, not just the value. A tracked checkpoint is a
                # DIFFERENT model from best.pt, taken at a different step; recording only its
                # headline number left every other column -- and the fusion, which reads the saved
                # scores -- describing best.pt while the row claimed to be about this one.
                tracked[m] = {"value": float(v), "step": step,
                              "checkpoint": ckpt_name(m),
                              "scores": scores_name(m),
                              "block": blk, "block_A": blk_a, "block_B": blk_b, "block_C": blk_c}
                if rank0():
                    save_ckpt(ckpt_name(m), m, v)
                    np.save(os.path.join(out_dir, scores_name(m)), s)
                    log0(f"[lorc]   new best {m}={v:.6f} -> {out_dir}/{ckpt_name(m)}")
        return True

    step = 0
    for ep in range(epochs):
        if dsampler is not None:
            dsampler.set_epoch(ep)
        log0(f"[lorc] --- epoch {ep + 1}/{epochs} ---")
        step, acc = train_epochs(model, loader_tr, opt, cfg, total_steps, amp_dtype, cw,
                                 on_eval=run_eval, start_step=step,
                                 log_every=int(cfg.get("log_every_steps", 50)))
        n = max(acc["n"], 1)
        n_all = max(acc.get("n_all_ranks", n), 1)
        log0(f"[lorc] epoch {ep + 1} done: loss={acc['loss']/n:.4f} cls={acc['cls']/n:.4f} "
             f"ss={acc['ss']/n:.4f} degenerate_ss_steps={acc['degenerate']}/{n_all} "
             f"(all ranks) unreadable={acc['unreadable']} (all ranks)")
        if acc["degenerate"] > 0.02 * n_all:
            log0(f"[lorc] WARNING: {acc['degenerate']}/{n_all} steps had only one label group "
                 f"present, "
                 f"so L_SS was 0 for them by absence rather than by separation. Raise batch_size "
                 f"or keep balanced_sampler on.")
        run_eval(step)

    # ------------------------------------------------------------------ record
    if best is None:
        raise SystemExit(
            f"[lorc] no checkpoint was ever selected: {sel_metric} was non-finite at every "
            f"evaluation. The selector rows cannot answer this metric -- check that all three "
            f"classes are present in {os.path.basename(cfg['sel_manifest'])}"
            + (" (and that the A part of the carve has them, since selection reads A)"
               if use_ab else "") + ".")
    if rank0() and best is not None:
        res = {"what": "LoRC (arXiv 2608.20882v1) on MIDS; selected on es_dev_sel, "
                       "es_dev_eval_c99 scored separately by predict.py",
               "paper": "LoRC: Detecting AI-Generated Images via Low-Rank Collapse in Semantic "
                        "Residuals, arXiv:2608.20882v1",
               "config": cfg,
               "encoder": os.path.abspath(cfg["encoder"]),
               "lora": cfg.get("lora"),
               "n_trainable": int(sum(p.numel() for p in
                                      (model.module if hasattr(model, "module") else model)
                                      .parameters() if p.requires_grad)),
               "num_classes": 3, "classes": ["real", "pad", "deepfake"],
               "ssl_source": cfg.get("ssl_source", "attn_out"),
               "ssl_mode": cfg.get("ssl_mode", "binary"),
               "head_input": cfg.get("head_input", "attn+cls"),
               "n_train": len(tr), "n_sel": len(sel),
               "train_class_counts": {"real": htr[0], "pad": htr[1], "deepfake": htr[2]},
               "tracked_metrics": tracked,
               "selected": {"metric": sel_metric, "value": best["value"], "step": best["step"],
                            "on": ("certified_sel_A" if use_ab else "certified_sel_all"),
                            "sel_manifest": os.path.abspath(cfg["sel_manifest"]),
                            "use_ab_split": use_ab},
               "dev_sel": best["block"],
               "dev_sel_A": best["block_A"], "dev_sel_B": best["block_B"],
               "dev_sel_C": best["block_C"],
               "ab_reading": ("A selected the checkpoint; B is reserved for fusion-member "
                              "selection; C fitted tau and is untouched by every decision."
                              if use_ab else
                              "The checkpoint was selected on the whole certified selector, which "
                              "is what every other member in this project does -- so tau on these "
                              "rows is mildly optimistic. dev_sel_B is the untouched-by-nothing "
                              "control: it was included in selection too, so it is NOT a holdout "
                              "here. Set use_ab_split=true to make it one."),
               "history": hist,
               "seconds": round(time.time() - t0, 1)}
        # tau is fitted on C when the carve is in use. Not A (which chose the checkpoint) and not
        # B (which chooses the fusion member in compare_configs.py) -- a split that has already
        # decided something has been spent, and a threshold fitted on it is fitted on rows the
        # decision was made to fit.
        # THE LAST PART, whatever the carve has. With --parts 3 that is C; with --parts 2 it is B
        # (which then does double duty, as the 2-way design always did). Hard-coding 2 made the
        # 2-part option unusable end to end -- it would have fallen back to the whole selector
        # without saying so.
        tau_part = int(ab.max()) if (use_ab and ab is not None) else None
        tau_mask = (ab == tau_part) if tau_part is not None \
            else np.ones(len(best["labels"]), bool)
        res["tau_fitted_on"] = (f"certified_sel_{'ABCDEFGH'[tau_part]}"
                                if tau_part is not None else "certified_sel_all")
        res["tau_part_index"] = tau_part
        ts, tl = best["scores"][tau_mask], best["labels"][tau_mask]
        tp = best["probs"][tau_mask] if best["probs"] is not None else None
        for t in (0.95, 0.98, 0.99, 0.999):
            tau, ach = M.tau_at_real_recall(ts[tl == M.REAL], t)
            res.setdefault("dev_sel_at_tau", {})[f"real{t}"] = {
                "tau": float(tau), "achieved_real_rec": float(ach),
                "deployable_rule": M.block_at_tau(ts, tl, tp, tau)}
        np.save(os.path.join(out_dir, "dev_sel_scores.npy"), best["scores"])
        np.save(os.path.join(out_dir, "dev_sel_labels.npy"), best["labels"])
        if ab is not None:
            np.save(os.path.join(out_dir, "dev_sel_ab.npy"), ab)
        with open(os.path.join(out_dir, "lorc.json"), "w") as fh:
            json.dump(res, fh, indent=1)
        log0(f"[lorc] wrote {out_dir}/lorc.json and dev_sel_scores.npy ({res['seconds']}s)")
        # run_predict.sh reads the checkpoint from $CKPT and forwards positional args to
        # predict.py, so "--ckpt ..." on its command line would be a duplicate flag AND would
        # leave the output directory pointing somewhere else. Print the form that works.
        log0(f"[lorc] NEXT: score the selector and then the report split, once:\n"
             f"  CKPT={out_dir}/best.pt ./run_predict.sh")

    if is_dist():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
