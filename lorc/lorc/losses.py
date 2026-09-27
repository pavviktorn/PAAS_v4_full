#!/usr/bin/env python3
"""LoRC objectives (arXiv:2608.20882v1, Eq 6-7).

The Subspace Separation Loss is the paper's own ablation winner: Table 3 credits it with
+2.4 average points (94.4 -> 96.8) and with almost all of the in-the-wild gain (90.2 -> 95.2),
which is a larger effect than either architectural part. It is the piece most worth getting right.

The classification term L_BCE lives on LoRCModel.classification_loss, because whether it is BCE or
cross-entropy follows from the head's output dimension.

TWO THINGS THIS FILE GETS RIGHT THAT THE OBVIOUS IMPLEMENTATION DOES NOT
------------------------------------------------------------------------
1. THE COVARIANCE IS POOLED ACROSS RANKS, EXACTLY.  Eq 6 normalises: P = R'R / ||R'R||_F. That
   makes P a NONLINEAR function of the batch, so computing L_SS on each rank's local microbatch and
   letting DDP average the gradients is NOT the same quantity as computing it on the combined
   batch -- it is a mean of four normalised 16-image estimates rather than one normalised 64-image
   estimate, with four times the variance in the covariance it rests on. Gradient accumulation does
   not fix it either, because the loss is renormalised for every microbatch.

   The fix is exact rather than approximate: R'R is a SUM over rows, so all-reducing the
   UNNORMALISED second moment and normalising afterwards reproduces the pooled result bit-for-bit.
   That is what accumulate_second_moments + subspace_separation_loss_from_moments do, and
   test_pooling_equivalence() in smoke_test.py asserts the identity on constructed inputs.

   Cost: one all-reduce of a D x D matrix per group per step -- 6.5 MB at D=1280, against 47 MB
   per rank to all-gather the residuals themselves.

2. DDP's GRADIENT AVERAGING NEEDS NO CORRECTION HERE -- AND APPLYING ONE IS A BUG.
   An earlier version of this file multiplied L_SS by world_size, reasoning that each rank's
   autograd contributes only its own share of dL_SS/dtheta while DDP divides the sum by W. That
   reasoning skipped a step. `torch.distributed.nn.functional.all_reduce` is DIFFERENTIABLE, and
   its backward all-reduces the incoming gradient:

       def backward(ctx, grad_output):
           return (None, None) + (_AllReduce.apply(ctx.op, ctx.group, grad_output),)

   Every rank holds the same global C and therefore the same dL/dC, so that backward hands each
   rank W * dL/dC where the true local value is dL/dC. The differentiable collective ALREADY
   pre-multiplies by W, and DDP's 1/W then cancels it exactly. Multiplying by W again makes the
   SSL gradient W times too large -- at 4 GPUs a configured lambda_ss=0.1 would act like 0.4, and
   a sweep over {0, 0.1, 0.3} would silently be a sweep over {0, 0.4, 1.2}.

   Measured, 2 ranks on gloo, gradient norm of a shared parameter:
       single-process pooled reference   0.684594512
       distributed, no extra scaling     0.684594512   <- exact
       distributed, scaled by world_size 1.369189024   <- exactly 2x
   test_ddp_matches_pooled_gradient() in smoke_test.py spawns two processes and asserts the first
   identity, so this cannot regress silently again.
"""
import torch
import torch.distributed as dist

REAL, PAD, DEEPFAKE = 0, 1, 2

# Which label groups the covariances are formed over. The paper is binary, and Eq 7 is written for
# exactly two groups; the other two modes exist because this project's label space is not.
SSL_MODES = ("binary", "pairwise", "real_vs_deepfake")


def _groups(labels, mode):
    """-> list[(name, boolean mask)] for the requested grouping."""
    if mode == "binary":
        # The paper's grouping: pad and deepfake pooled into one "fake" covariance.
        return [("real", labels == REAL), ("fake", labels != REAL)]
    if mode == "pairwise":
        # All three subspaces separated. Not the paper's; this project's extension.
        return [("real", labels == REAL), ("pad", labels == PAD),
                ("deepfake", labels == DEEPFAKE)]
    if mode == "real_vs_deepfake":
        # PAD EXCLUDED FROM L_SS ENTIRELY, while still being classified by the cross-entropy term.
        # The paper's premise is about GENERATED pixels; a PAD sample is a camera capture of a
        # print, screen or mask, so pooling it with deepfake asks one covariance to describe two
        # unrelated physical processes. This mode tests whether that pooling costs anything.
        return [("real", labels == REAL), ("deepfake", labels == DEEPFAKE)]
    raise ValueError(f"ssl_mode must be one of {SSL_MODES}, got {mode!r}")


def normalised_second_moment(C, eps=1e-8):
    """Eq 6: P = C / ||C||_F, for C = R'R already accumulated."""
    return C / (C.norm(p="fro") + eps)


def accumulate_second_moments(feats, labels, mode="binary", subsample=0, generator=None):
    """-> (dict name -> UNNORMALISED C = R'R, dict name -> row count).

    `feats` (B, N, D) are the per-patch features the loss operates on -- X_res for the paper's
    main-text reading, or the low-rank attention output for the supplementary one. Which is passed
    is the caller's decision; see lorc/model.py ssl_source.

    Returned unnormalised ON PURPOSE: normalising here would make cross-rank pooling impossible,
    which is finding (1) in this file's docstring.
    """
    B, N, D = feats.shape
    flat = feats.reshape(B * N, D)
    C, counts = {}, {}
    for name, m in _groups(labels, mode):
        n = int(m.sum())
        counts[name] = n * N
        # AN ABSENT GROUP STILL PRODUCES AN ENTRY, and it is produced the same way as a present
        # one. `flat[all_false]` is (0, D), so R'R is the correct D x D of zeros AND stays attached
        # to this rank's feature graph. Skipping the group instead (`if n == 0: continue`) left
        # reduce_second_moments to substitute a bare torch.zeros, which has requires_grad=False --
        # so autograd recorded NO node for it, that rank issued no all-reduce in backward, and the
        # ranks that DID have the group blocked forever waiting for it. See this file's finding (3).
        R = flat[m.repeat_interleave(N)].float()
        if subsample and R.shape[0] > subsample:
            idx = torch.randperm(R.shape[0], device=R.device, generator=generator)[:subsample]
            R = R[idx]
            counts[name] = int(R.shape[0])
        C[name] = R.t() @ R
    return C, counts


def reduce_second_moments(C, counts, names, dim, device, dtype=torch.float32, anchor=None):
    """Sum each group's C and row count over all ranks, differentiably.

    EVERY RANK MUST ALL-REDUCE EVERY GROUP, AND EVERY ONE OF THOSE TENSORS MUST CARRY A GRAD_FN.
    Matching the forward collectives is not sufficient. `torch.distributed.nn.functional.all_reduce`
    is an autograd Function, so it records a backward node only if its input requires grad; handed
    a bare `torch.zeros` it silently becomes a no-op in backward. The ranks that did have the group
    then wait in `all_reduce` for a peer that will never call it -- a hang, not an error.

    accumulate_second_moments now always supplies a graph-connected zero, so `anchor` is a
    defensive fallback for callers that build C some other way.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return C, counts
    from torch.distributed.nn.functional import all_reduce as all_reduce_autograd
    out_C, out_n = {}, {}
    for name in names:
        local = C.get(name)
        if local is None:
            local = torch.zeros(dim, dim, dtype=dtype, device=device)
            if anchor is not None and anchor.requires_grad:
                local = local + 0.0 * anchor.float().sum()
        if not local.requires_grad and torch.is_grad_enabled() and anchor is not None \
                and anchor.requires_grad:
            raise RuntimeError(
                f"[lorc] group {name!r} covariance is detached from the graph. Under DDP this "
                f"hangs: this rank would record no backward node for its all-reduce while other "
                f"ranks wait on it. Build absent groups with a zero-ROW index, not torch.zeros.")
        out_C[name] = all_reduce_autograd(local, op=dist.ReduceOp.SUM)
        n = torch.tensor([float(counts.get(name, 0))], device=device)
        dist.all_reduce(n, op=dist.ReduceOp.SUM)
        out_n[name] = int(n.item())
    return out_C, out_n


def subspace_separation_loss_from_moments(C, counts, eps=1e-8):
    """Eq 7 from accumulated moments. -> (loss, n_pairs, counts).

    n_pairs IS RETURNED SO A DEGENERATE BATCH CANNOT BE SILENT. With one group absent there is no
    pair to separate and the correct value is 0.0 -- indistinguishable, from the loss curve alone,
    from a perfectly separated batch. train.py counts these steps and warns if they are common,
    which is what the class-balanced sampler is meant to prevent.
    """
    present = [n for n in C if counts.get(n, 0) > 0]
    if len(present) < 2:
        z = next(iter(C.values())).sum() * 0.0 if C else torch.zeros(())
        return z, 0, counts
    P = {n: normalised_second_moment(C[n], eps) for n in present}
    terms = [(P[present[i]] * P[present[j]]).sum()
             for i in range(len(present)) for j in range(i + 1, len(present))]
    return torch.stack(terms).mean(), len(terms), counts


def subspace_separation_loss(feats, labels, mode="binary", subsample=0, generator=None,
                             eps=1e-8, distributed=True, scale_for_ddp=False):
    """Eq 6-7 end to end, pooled across ranks. -> (loss, n_pairs, counts).

    `scale_for_ddp` MUST STAY FALSE. It is retained only to reproduce a fixed bug; see item (2).

    `subsample` caps the rows of R per group per rank. R'R costs O(M D^2) and R is retained for the
    backward pass, so M is the only knob that touches it; 0 = no cap (paper-faithful). At D=1280
    the whole term is ~0.02 TFLOP per step, so the cap is only needed for the 7B encoder.
    """
    if mode not in SSL_MODES:
        raise ValueError(f"ssl_mode must be one of {SSL_MODES}, got {mode!r}")
    D = feats.shape[-1]
    C, counts = accumulate_second_moments(feats, labels, mode, subsample, generator)
    names = [n for n, _ in _groups(labels, mode)]
    world = 1
    if distributed and dist.is_available() and dist.is_initialized():
        C, counts = reduce_second_moments(C, counts, names, D, feats.device, anchor=feats)
        world = dist.get_world_size()
    loss, n_pairs, counts = subspace_separation_loss_from_moments(C, counts, eps)
    if scale_for_ddp and world > 1:
        # WRONG, and off by exactly world_size -- kept only so the bug is reproducible from a
        # config. The differentiable all-reduce's backward already pre-multiplies by W; see
        # docstring item (2). Never enable this.
        loss = loss * world
    return loss, n_pairs, counts
