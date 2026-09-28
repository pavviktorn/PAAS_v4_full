"""Metric block for LoRC.

VENDORED VERBATIM from PAAS_simplicity/spc/metrics.py (the PE-SPC / DINOv3-SPC metric block), on
purpose and without edits to the maths. Every number in runs/lorc.json is then produced by the
exact code that produced runs/dinospc.json and runs/pespc_final.json, so LoRC's row is comparable
to the existing members by construction rather than by inspection of two implementations.

One caveat for the num_classes=2 configuration: `block_at_tau` indexes probs[:, 1:] to choose
between pad and deepfake, which needs the 3-class head. train.py only calls it when
num_classes == 3; the binary head reports `block` and the applied-tau recalls, which are
class-agnostic.
"""

from __future__ import annotations
import numpy as np

REAL, PAD, DEEPFAKE = 0, 1, 2
CLS = ("real", "pad", "deepfake")


def _midranks(x: np.ndarray) -> np.ndarray:
    """Tie-averaged ranks, fully vectorised. The obvious Python tie-loop is O(n) *interpreted* and
    the obvious EER loop is O(n^2); axon1 has 614,029 rows, so both have to be array ops or the
    metric block costs more than the training run."""
    order = np.argsort(x, kind="mergesort")
    sx = x[order]
    start = np.empty(len(sx), dtype=bool)
    start[0] = True
    np.not_equal(sx[1:], sx[:-1], out=start[1:])
    grp = np.cumsum(start) - 1
    first = np.flatnonzero(start)
    last = np.r_[first[1:], len(sx)] - 1
    mid = 0.5 * (first + last) + 1.0
    ranks = np.empty(len(sx), dtype=np.float64)
    ranks[order] = mid[grp]
    return ranks


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann-Whitney U with proper mid-rank tie handling (ties are common: many identical scores)."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = _midranks(np.concatenate([pos, neg]))
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def _ap(scores: np.ndarray, y: np.ndarray) -> float:
    """Average precision, TIE-AWARE. positives = y == 1.

    The previous implementation walked the sorted rows individually, so tied scores were ranked by
    array position and AP depended on INPUT ORDER. Measured on all-tied inputs: 3 positives and 7
    negatives gave AP 1.000000 with the positives first and 0.215741 with them last, where the only
    defensible answer is the prevalence, 0.300000. That is not academic here -- the eval manifest is
    block-ordered by class, and a saturated detector produces long runs of identical scores, so the
    metric would have been reading the manifest's layout as if it were model skill.

    Fix: collapse each set of equal scores into ONE operating point (the same thing
    precision_recall_curve does), and integrate the step function over those points:
        AP = sum_i (R_i - R_{i-1}) * P_i
    with P_i, R_i evaluated at the END of tie group i. All-tied input then yields exactly the
    prevalence, and the result is independent of the input order.
    """
    n_pos = int((y == 1).sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    s, yy = scores[order], y[order]
    # end index of every run of equal scores
    last = np.r_[np.flatnonzero(s[1:] != s[:-1]), len(s) - 1]
    tp = np.cumsum(yy)[last]
    n_pred = last + 1                              # predicted-positive count at each threshold
    prec = tp / n_pred
    rec = tp / n_pos
    d_rec = np.diff(np.r_[0.0, rec])
    return float((d_rec * prec).sum())


def _eer(pos: np.ndarray, neg: np.ndarray) -> float:
    """Vectorised: FAR/FRR at every candidate threshold via searchsorted, not a per-threshold scan."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    thr = np.unique(np.concatenate([pos, neg]))
    ps, ns = np.sort(pos), np.sort(neg)
    far = 1.0 - np.searchsorted(ns, thr, side="left") / len(ns)     # real accepted as fake
    frr = np.searchsorted(ps, thr, side="left") / len(ps)           # fake rejected as real
    i = int(np.argmin(np.abs(far - frr)))
    return float((far[i] + frr[i]) / 2.0)


def tau_at_real_recall(real_scores: np.ndarray, target: float) -> tuple[float, float]:
    """Smallest tau achieving real recall >= target. Returns (tau, achieved_real_recall).

    A real sample AT tau is predicted fake (decision is `>= tau`), so tau is nudged strictly above
    the k-th smallest real score with nextafter -- off-by-one here silently costs one real per
    thousand at the exact operating point the product lives at.
    """
    n = len(real_scores)
    if n == 0:
        return float("nan"), float("nan")
    s = np.sort(real_scores)
    k = int(np.ceil(target * n))
    k = min(max(k, 1), n)
    tau = float(np.nextafter(s[k - 1], np.inf))
    return tau, float((real_scores < tau).mean())


def block(fake_score: np.ndarray, labels: np.ndarray, probs: np.ndarray | None = None,
          targets=(0.95, 0.98, 0.99)) -> dict:
    """The full metric block. `labels` are 3-class (0 real / 1 pad / 2 deepfake)."""
    labels = np.asarray(labels).astype(np.int64)
    fake_score = np.asarray(fake_score, dtype=np.float64)
    is_fake = (labels != REAL).astype(np.int64)
    pos, neg = fake_score[is_fake == 1], fake_score[is_fake == 0]
    out = {
        "n": int(len(labels)),
        "n_real": int((labels == REAL).sum()),
        "n_pad": int((labels == PAD).sum()),
        "n_deepfake": int((labels == DEEPFAKE).sum()),
        "bin_auc": _auc(pos, neg),
        "ap": _ap(fake_score, is_fake),
        "eer": _eer(pos, neg),
    }
    for t in targets:
        tau, ach = tau_at_real_recall(neg, t)
        out[f"tau@real{int(t*100)}"] = tau
        out[f"real_rec@real{int(t*100)}"] = ach                     # ACHIEVED, next to the target
        out[f"fake_rec@real{int(t*100)}"] = float((pos >= tau).mean()) if len(pos) else float("nan")
        for c, nm in ((PAD, "pad"), (DEEPFAKE, "deepfake")):
            m = labels == c
            out[f"{nm}_rec@real{int(t*100)}"] = float((fake_score[m] >= tau).mean()) if m.any() else float("nan")
    if probs is not None:
        pred = np.asarray(probs).argmax(1)
        out["acc3_NOT-A-DECISION-METRIC"] = float((pred == labels).mean())
        recs = []
        for c in range(3):
            m = labels == c
            r = float((pred[m] == c).mean()) if m.any() else float("nan")
            out[f"rec3_{CLS[c]}"] = r
            if m.any():
                recs.append(r)
        out["bal_acc3"] = float(np.mean(recs))
        out["confusion"] = [[int(((labels == a) & (pred == b)).sum()) for b in range(3)] for a in range(3)]
    return out


def block_at_tau(fake_score, labels, probs, tau) -> dict:
    """3-class metrics under the DEPLOYABLE decision rule, which is not argmax:

        real      if fake_score <  tau
        pad/deepfake  otherwise, by argmax over the two fake classes

    Raw argmax is misleading for an SPC head -- with unit features, unit prototypes and bias
    {0,1,1} the logits span [-1,2], so p_real rarely wins outright even when the RANKING is
    perfect (measured: rec3_real 0.305 at bin_auc 1.000000). A served system thresholds the fake
    score and only then asks which kind of fake it is, so that is what gets reported as accuracy.
    """
    labels = np.asarray(labels).astype(np.int64)
    fake_score = np.asarray(fake_score, dtype=np.float64)
    probs = np.asarray(probs)
    pred = np.where(fake_score < tau, REAL, 1 + probs[:, 1:].argmax(1))
    out = {"tau": float(tau), "acc3@tau": float((pred == labels).mean())}
    recs = []
    for c in range(3):
        m = labels == c
        if m.any():
            r = float((pred[m] == c).mean()); out[f"rec3@tau_{CLS[c]}"] = r; recs.append(r)
    out["bal_acc3@tau"] = float(np.mean(recs))
    out["confusion@tau"] = [[int(((labels == a) & (pred == b)).sum()) for b in range(3)]
                            for a in range(3)]
    return out


def fmt(m: dict, keys=None) -> str:
    keys = keys or ["bin_auc", "ap", "eer", "fake_rec@real95", "fake_rec@real98", "fake_rec@real99"]
    return " ".join(f"{k}={m[k]:.6f}" for k in keys if k in m and isinstance(m[k], float))
