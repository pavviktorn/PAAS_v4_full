#!/usr/bin/env python3
"""Three-class discrimination metrics, ADDED BESIDE lorc/metrics.py rather than inside it.

WHY A SEPARATE FILE. `lorc/metrics.py` is vendored VERBATIM from PAAS_simplicity/spc/metrics.py so
that LoRC's numbers come out of the same code as dinospc.json and pespc's -- that is what makes the
members comparable at all. Editing it to add keys would quietly end that guarantee. Everything here
is additive: the shared keys keep coming from the shared file, and these extra keys are merged on
top.

WHAT IT MEASURES, AND WHY IT IS NOT ALREADY COVERED. `fake_rec@real99` is the deployed objective --
the fusion's threshold is fitted at a 99% real-recall floor -- but it is BINARY. A checkpoint that
flags every PAD and every deepfake as "fake" while being unable to tell them apart scores exactly
as well as one that separates them perfectly. Since this project requires a real/pad/deepfake model
(not a binary one), that is a real blind spot in checkpoint selection.

The metrics here are all CONDITIONED ON THE OPERATING POINT, not on argmax over the whole split:
a 3-class accuracy that includes images the deployed rule would never have called fake is not
measuring anything the deployment does.

  pad_df_sep@realT   among rows CORRECTLY flagged fake at tau(T), the balanced accuracy of the
                     pad-vs-deepfake call. 0.5 = coin flip, 1.0 = perfect.
  bal_acc3@realT     3-class balanced accuracy of the deployable rule at tau(T): below tau -> real,
                     above -> argmax(pad, deepfake).
DO NOT RANK ON pad_df_sep, AND DO NOT RANK ON A PRODUCT OF IT. An earlier version of this file
offered `fake_rec_x_sep@realT = fake_rec@realT * pad_df_sep@realT` as the 3-class selection metric.
It is unsafe and has been REMOVED. The reason is a Jensen-style mismatch: the product multiplies
two AVERAGES -- mean detection over {pad, deepfake} times mean separation over {pad, deepfake} --
whereas real 3-class accuracy averages the PRODUCTS per class. mean(a)*mean(b) != mean(a*b) unless
a and b are uncorrelated, so whenever detection and separation are ANTI-correlated across the two
fake classes (a model good at catching PAD but bad at naming it, and vice versa) the composite
misranks. Measured on this code, 40k rows per class:

    model   fake_rec@99  pad_df_sep@99   COMPOSITE   bal_acc3@99   per-class 3-class recall
    A            0.6008         0.7009      0.4211        0.5311   pad 1.000 / df 0.402
    B            0.5240         0.5234      0.2743        0.6641   pad 0.047 / df 1.000

The composite prefers A by 0.147 while A is 0.133 WORSE in 3-class balanced accuracy.

USE bal_acc3@realT INSTEAD. It is the balanced accuracy of the DEPLOYABLE RULE at tau -- below tau
-> real, above -> argmax(pad, deepfake) -- so it already averages per-class products and cannot be
gamed this way. `pad_df_sep` stays as a DIAGNOSTIC (it answers "given that it caught the fake, did
it name it?") and train.py refuses to select on it.
"""
import numpy as np

from . import metrics as M

TARGETS = (0.95, 0.98, 0.99, 0.999)


def three_class_block(fake_score, labels, probs=None, targets=TARGETS):
    """-> dict of the keys above. Empty if `probs` is absent (they need a pad/deepfake call)."""
    out = {}
    if probs is None:
        return out
    probs = np.asarray(probs)
    if probs.ndim != 2 or probs.shape[1] < 3:
        return out
    fake_score = np.asarray(fake_score)
    labels = np.asarray(labels)
    # THE SAME EXPRESSION metrics.block_at_tau USES, deliberately copied rather than re-derived:
    #     pred = np.where(fake_score < tau, REAL, 1 + probs[:, 1:].argmax(1))
    # This file previously wrote `probs[DEEPFAKE] >= probs[PAD] -> DEEPFAKE`, which breaks ties the
    # OPPOSITE way from argmax (argmax returns the first maximum, i.e. PAD). Selection and
    # prediction then disagreed on every tied row -- on constructed all-tied probabilities that is
    # the difference between 66.33% and 99.67% balanced accuracy from identical inputs. A metric
    # that scores a checkpoint must apply the rule that checkpoint will be deployed under.
    df_call = M.PAD + np.asarray(probs)[:, M.PAD:].argmax(1)
    real = fake_score[labels == M.REAL]
    for t in targets:
        tag = f"real{int(t * 100) if t < 0.999 else 999}"
        if len(real) == 0:
            continue
        tau, _ = M.tau_at_real_recall(real, t)
        flagged = fake_score >= tau
        recs = []
        for c in (M.PAD, M.DEEPFAKE):
            m = (labels == c) & flagged
            recs.append(float((df_call[m] == c).mean()) if m.any() else float("nan"))
        sep = float(np.mean(recs)) if not any(np.isnan(recs)) else float("nan")
        out[f"pad_df_sep@{tag}"] = sep
        out[f"pad_sep@{tag}"], out[f"deepfake_sep@{tag}"] = recs

        pred = np.where(flagged, df_call, M.REAL)
        r3 = []
        for c in (M.REAL, M.PAD, M.DEEPFAKE):
            m = labels == c
            r3.append(float((pred[m] == c).mean()) if m.any() else float("nan"))
        out[f"bal_acc3@{tag}"] = float(np.mean(r3)) if not any(np.isnan(r3)) else float("nan")

        # No composite is emitted: see this module's docstring for why the obvious one misranks.
    return out


# Keys that describe but must never RANK. pad_df_sep is conditioned on detection, so a model can
# raise it by detecting less; the per-class variants inherit that.
UNSAFE_FOR_SELECTION = {
    "pad_df_sep": "it is conditioned on the fakes the model happened to detect, so detecting "
                  "FEWER fakes can raise it. Use bal_acc3@realT.",
    "pad_sep": "conditioned on detection -- see pad_df_sep. Use bal_acc3@realT.",
    "deepfake_sep": "conditioned on detection -- see pad_df_sep. Use bal_acc3@realT.",
    "fake_rec_x_sep": "REMOVED. It multiplies mean-detection by mean-separation, while true "
                      "3-class accuracy averages the per-class products; when detection and "
                      "separation are anti-correlated across pad/deepfake it picks the worse "
                      "model (measured: 0.133 worse bal_acc3). Use bal_acc3@realT.",
}


def check_selectable(metric):
    """Raise if `metric` is one this module records but that must not be ranked on."""
    base = metric.split("@", 1)[0]
    if base in UNSAFE_FOR_SELECTION:
        raise SystemExit(
            f"[lorc] select_metric={metric!r} must not be used to choose a checkpoint: "
            f"{UNSAFE_FOR_SELECTION[base]}")


def merged_block(fake_score, labels, probs=None):
    """metrics.block(...) plus the three-class keys. Shared keys still come from metrics.py."""
    b = M.block(fake_score, labels, probs)
    b.update(three_class_block(fake_score, labels, probs))
    return b
