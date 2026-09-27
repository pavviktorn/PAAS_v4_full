#!/usr/bin/env python3
"""Train / evaluate loops for LoRC, with DDP.

THE SAMPLER IS PART OF THE METHOD, NOT A DETAIL
-----------------------------------------------
L_SS (Eq 7) is defined on P_real and P_fake computed from the CURRENT minibatch. A batch with no
real images has no pair to separate, so the correct loss is 0.0 -- and a 0.0 from an absent group
is indistinguishable, in the loss curve, from a 0.0 earned by perfect separation. With the project
trainset at roughly 25% real (spc3_train: real 417k / pad 529k / deepfake 311k) and the paper's
per-device batch, an unbalanced sampler leaves a meaningful fraction of steps with no real group
on some rank.

So: a class-balanced sampler is the default, AND `ssl_degenerate_steps` is counted and reported.
If that count is not ~0, the number is in runs/lorc.json rather than hidden behind an average.

WHAT IS EVALUATED, AND WHEN
---------------------------
Selection happens on es_dev_sel only. es_dev_eval_c99 is scored ONCE, at the end, by predict.py --
never inside the training loop, so it cannot influence a checkpoint choice through early stopping.
This mirrors train_dinospc.py's contract.
"""
import math
import os
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, WeightedRandomSampler

from .losses import subspace_separation_loss


def is_dist():
    return dist.is_available() and dist.is_initialized()


def rank0():
    return (not is_dist()) or dist.get_rank() == 0


def log0(*a, **k):
    if rank0():
        print(*a, **k, flush=True)


def balanced_sample_weights(samples):
    """Per-sample weights making real, pad and deepfake equally likely in the stream.

    Balancing on the 3-class label rather than on real-vs-fake is deliberate. The trainset is
    roughly real 417k / pad 529k / deepfake 311k, so a real-vs-fake balance would still feed
    deepfake -- the hard class, and the one the rest of this stack already struggles with -- at
    about 37% of the fake half. Equalising all three puts a third of every batch on it.

    IT DOES NOT GUARANTEE EVERY GROUP IN EVERY MICROBATCH, and an earlier version of this docstring
    claimed it did. At p=1/3 and batch_size 16 PER RANK, P(a rank draws no real at all) = (2/3)^16
    = 0.15%, so across 4 ranks a group is missing from some rank about once every 153 microbatches
    in `binary` mode -- 26 times in a single 4,000-microbatch screen -- and about once every 83 in
    `real_vs_deepfake`, which excludes PAD and so has two thinner groups. Eq 7 handles that
    correctly (the covariance is pooled across ranks, so a locally-absent group is still present
    globally), but it is the reason lorc/losses.py must build an absent group's covariance from a
    zero-ROW index rather than torch.zeros: see the DDP note there.
    """
    h = [0, 0, 0]
    for _, lab in samples:
        h[lab] += 1
    w = [1.0 / max(h[lab], 1) for _, lab in samples]
    return torch.as_tensor(w, dtype=torch.double), h


def make_train_loader(dataset, samples, batch_size, workers, balanced=True,
                      seed=0, epoch_steps=None):
    """DistributedSampler for the plain case; a per-rank WeightedRandomSampler for the balanced one.

    DistributedSampler and WeightedRandomSampler do not compose, so the balanced path shards the
    index space by rank itself: each rank draws from its OWN disjoint slice, which keeps the two
    properties that matter -- every rank sees a class-balanced stream, and no image is used by two
    ranks in the same step.
    """
    if not balanced:
        sampler = (DistributedSampler(dataset, shuffle=True, seed=seed) if is_dist()
                   else None)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                          shuffle=(sampler is None), num_workers=workers, pin_memory=True,
                          drop_last=True, persistent_workers=workers > 0), sampler

    world = dist.get_world_size() if is_dist() else 1
    rank = dist.get_rank() if is_dist() else 0
    w, _ = balanced_sample_weights(samples)
    idx = torch.arange(rank, len(samples), world)                     # this rank's disjoint slice
    n_draw = epoch_steps * batch_size if epoch_steps else len(idx)
    sub = WeightedRandomSampler(w[idx].tolist(), num_samples=n_draw, replacement=True,
                                generator=torch.Generator().manual_seed(seed + rank))
    # WeightedRandomSampler yields positions into its own weight list; map back to dataset indices.
    mapped = _Remap(sub, idx.tolist())
    return DataLoader(dataset, batch_size=batch_size, sampler=mapped, num_workers=workers,
                      pin_memory=True, drop_last=True,
                      persistent_workers=workers > 0), None


class _Remap(torch.utils.data.Sampler):
    def __init__(self, inner, mapping):
        self.inner, self.mapping = inner, mapping

    def __iter__(self):
        for i in self.inner:
            yield self.mapping[i]

    def __len__(self):
        return len(self.inner)


def make_eval_loader(dataset, batch_size, workers):
    sampler = (DistributedSampler(dataset, shuffle=False, drop_last=False) if is_dist() else None)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=False,
                      num_workers=workers, pin_memory=True), sampler


def reduce_int(v, device):
    """Sum a python int across ranks. Per-rank counters printed by rank 0 alone describe one
    quarter of a 4-GPU run, which is how a problem on rank 3 stays invisible."""
    if not is_dist():
        return int(v)
    t = torch.tensor([float(v)], device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return int(t.item())


def cosine_lr(step, total, base, warmup, min_ratio=0.05):
    if step < warmup:
        return base * (step + 1) / max(warmup, 1)
    t = (step - warmup) / max(total - warmup, 1)
    return base * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(t, 1.0))))


@torch.no_grad()
def evaluate(model, loader, amp_dtype, n_total, want_probs=True):
    """-> (fake_scores, labels, probs) as numpy, in ORIGINAL MANIFEST ROW ORDER.

    Gathering by row index rather than by concatenation is what makes the arrays alignable to the
    manifest: with DDP the per-rank order is strided, and DistributedSampler PADS the last batch by
    repeating early samples. Scattering into a preallocated array by row index makes both harmless
    -- a duplicate simply writes the same value twice -- whereas concatenating ranks would silently
    produce a permuted array of the wrong length.
    """
    core = model.module if hasattr(model, "module") else model
    # eval() is load-bearing, not hygiene: DINOv3 enables position-embedding augmentation
    # (pos_embed_rescale=2.0) in train mode, so train-mode forwards of the same image differ by up
    # to 2.6e-2 in the hidden states even with every dropout at 0. Scores taken in train mode
    # would not be reproducible, and the residual spectrum this method reads is exactly what that
    # augmentation perturbs.
    core.eval()
    dev = next(core.parameters()).device
    scores = torch.full((n_total,), float("nan"), dtype=torch.float64, device=dev)
    labels = torch.full((n_total,), -1, dtype=torch.long, device=dev)
    probs = (torch.full((n_total, core.NUM_CLASSES), float("nan"), dtype=torch.float32,
                        device=dev) if want_probs else None)
    n_unreadable = torch.zeros((), dtype=torch.long, device=dev)

    for x, lab, row, ok in loader:
        x = x.to(dev, non_blocking=True)
        row = row.to(dev, non_blocking=True)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            o = core(x, need_residual=False)
        s = core.fake_score(o["logits"]).double()
        scores[row] = s
        labels[row] = lab.to(dev, non_blocking=True)
        if want_probs:
            probs[row] = core.class_probs(o["logits"])
        n_unreadable += (ok == 0).sum().to(dev)

    if is_dist():
        # NaN marks "not scored by this rank"; max-reduce lets the owning rank's value win.
        scores = torch.nan_to_num(scores, nan=-1.0)
        dist.all_reduce(scores, op=dist.ReduceOp.MAX)
        dist.all_reduce(labels, op=dist.ReduceOp.MAX)
        if want_probs:
            probs = torch.nan_to_num(probs, nan=-1.0)
            dist.all_reduce(probs, op=dist.ReduceOp.MAX)
        dist.all_reduce(n_unreadable, op=dist.ReduceOp.SUM)
        missing = int((scores < 0).sum())
    else:
        missing = int(torch.isnan(scores).sum())
    if missing:
        raise RuntimeError(f"[eval] {missing:,} of {n_total:,} rows were never scored -- the score "
                           f"array is not aligned to the manifest and must not be written.")
    core.train()
    return (scores.cpu().numpy(), labels.cpu().numpy(),
            probs.cpu().numpy() if want_probs else None, int(n_unreadable))


def train_epochs(model, loader, opt, cfg, total_steps, amp_dtype, class_weight,
                 on_eval=None, start_step=0, log_every=50):
    """One pass over `loader`; returns (step, stats). `on_eval(step)` is called every
    eval_every_steps and may return False to stop early."""
    core = model.module if hasattr(model, "module") else model
    dev = next(core.parameters()).device
    step = start_step
    t0 = time.time()
    acc = {"loss": 0.0, "cls": 0.0, "ss": 0.0, "n": 0, "degenerate": 0, "unreadable": 0}
    accum = max(int(cfg.get("grad_accum", 1)), 1)
    lam = float(cfg.get("lambda_ss", 0.1))
    ss_mode = cfg.get("ssl_mode", "binary")
    ss_sub = int(cfg.get("ssl_patch_subsample", 0))
    need_res = lam > 0
    strict_unreadable = bool(cfg.get("refuse_unreadable_train", True))

    opt.zero_grad(set_to_none=True)
    for i, (x, lab, row, ok) in enumerate(loader):
        lr_now = cosine_lr(step, total_steps, cfg["lr"], int(cfg.get("warmup_steps", 0)),
                           float(cfg.get("min_lr_ratio", 0.05))) \
            if cfg.get("lr_schedule", "cosine") == "cosine" else cfg["lr"]
        for g in opt.param_groups:
            g["lr"] = lr_now * g.get("lr_mult", 1.0)

        x = x.to(dev, non_blocking=True)
        lab = lab.to(dev, non_blocking=True)
        nbad = int((ok == 0).sum())
        acc["unreadable"] += nbad
        if nbad and strict_unreadable:
            # Previously these were counted and trained on. A grey placeholder carries no forgery
            # artefact and its patch residuals enter Eq 6's covariance as a real measurement, so
            # "counted" is not the same as "handled". Set refuse_unreadable_train=false to go back
            # to counting only.
            raise RuntimeError(
                f"[lorc] {nbad} unreadable image(s) in a training batch. They would be trained on "
                f"as grey frames and their residuals would enter the Eq-6 covariance. Fix the "
                f"manifest, or set refuse_unreadable_train=false to train through them.")

        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            o = model(x, need_residual=need_res)
        cls_loss = core.classification_loss(o["logits"], lab, class_weight)
        if need_res:
            # core.ssl_features picks X_res / Y / A per cfg ssl_source -- the paper's main text and
            # its supplement disagree about which, so it is a configured experiment, not a guess.
            # The loss pools its covariance across ranks internally (see lorc/losses.py), so the
            # grouping below is over the GLOBAL batch, not this rank's 16 images.
            ss_loss, n_pairs, _ = subspace_separation_loss(
                core.ssl_features(o), lab, mode=ss_mode, subsample=ss_sub)
            if n_pairs == 0:
                acc["degenerate"] += 1
        else:
            ss_loss, n_pairs = cls_loss.new_zeros(()), 0
        loss = cls_loss + lam * ss_loss
        (loss / accum).backward()

        if (i + 1) % accum == 0:
            if cfg.get("grad_clip", 0):
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], cfg["grad_clip"])
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1

        # .detach() before float(): these three are pure logging, and float() on a tensor that
        # still requires grad warns once per microbatch per rank -- four lines of torch warning
        # for every batch of a multi-day sweep. The value is identical either way.
        acc["loss"] += float(loss.detach()); acc["cls"] += float(cls_loss.detach())
        acc["ss"] += float(ss_loss.detach()); acc["n"] += 1

        if acc["n"] % log_every == 0:
            n = acc["n"]
            acc["degenerate_all"] = reduce_int(acc["degenerate"], dev)
            acc["unreadable_all"] = reduce_int(acc["unreadable"], dev)
            log0(f"[lorc] step {step:>6}/{total_steps} lr={lr_now:.2e} "
                 f"loss={acc['loss']/n:.4f} (cls={acc['cls']/n:.4f} ss={acc['ss']/n:.4f}) "
                 f"degenerate_ss={acc['degenerate_all']} "
                 f"{n*x.shape[0]*(dist.get_world_size() if is_dist() else 1)/(time.time()-t0):.1f} "
                 f"img/s")

        if on_eval is not None and cfg.get("eval_every_steps") and \
                step > 0 and step % int(cfg["eval_every_steps"]) == 0 and (i + 1) % accum == 0:
            if on_eval(step) is False:
                break
    # Reduce before returning, so the epoch summary and the degenerate-step warning in train.py
    # describe every rank rather than rank 0 alone.
    acc["degenerate"] = reduce_int(acc["degenerate"], dev)
    acc["unreadable"] = reduce_int(acc["unreadable"], dev)
    acc["n_all_ranks"] = reduce_int(acc["n"], dev)
    return step, acc
