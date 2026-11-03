"""Classification metrics, calibration, threshold selection and bootstrap intervals."""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

import config as C

THRESHOLD_METRICS = ("accuracy", "sensitivity", "specificity", "precision", "f1",
                     "balanced_accuracy", "mcc", "youden_j")
SCORE_METRICS = ("auroc", "auprc", "brier", "ece")


def confusion(y, pred):
    y = np.asarray(y).astype(bool)
    pred = np.asarray(pred).astype(bool)
    tp = int(np.sum(y & pred))
    tn = int(np.sum(~y & ~pred))
    fp = int(np.sum(~y & pred))
    fn = int(np.sum(y & ~pred))
    return tp, tn, fp, fn


def _safe(a, b):
    return float(a) / float(b) if b else float("nan")


def threshold_metrics(tp, tn, fp, fn):
    sens = _safe(tp, tp + fn)
    spec = _safe(tn, tn + fp)
    prec = _safe(tp, tp + fp)
    denom = np.sqrt(float(tp + fp) * float(tp + fn) * float(tn + fp) * float(tn + fn))
    return dict(
        accuracy=_safe(tp + tn, tp + tn + fp + fn),
        sensitivity=sens,
        specificity=spec,
        precision=prec,
        f1=_safe(2 * tp, 2 * tp + fp + fn),
        balanced_accuracy=(sens + spec) / 2,
        mcc=_safe(float(tp) * tn - float(fp) * fn, denom),
        youden_j=sens + spec - 1,
    )


def expected_calibration_error(y, p, n_bins=15):
    y, p = np.asarray(y, float), np.asarray(p, float)
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def score_metrics(y, p):
    y, p = np.asarray(y), np.asarray(p, float)
    both = len(np.unique(y)) == 2
    return dict(
        auroc=float(roc_auc_score(y, p)) if both else float("nan"),
        auprc=float(average_precision_score(y, p)) if both else float("nan"),
        brier=float(np.mean((p - y) ** 2)),
        ece=expected_calibration_error(y, p),
    )


def all_metrics(y, p, threshold=0.5):
    y, p = np.asarray(y), np.asarray(p, float)
    tp, tn, fp, fn = confusion(y, p >= threshold)
    out = dict(n=len(y), n_pos=int(np.sum(y)), threshold=float(threshold),
               tp=tp, tn=tn, fp=fp, fn=fn, positive_rate=float(np.mean(p >= threshold)))
    out.update(threshold_metrics(tp, tn, fp, fn))
    out.update(score_metrics(y, p))
    return out


def youden_threshold(y, p):
    """Operating point maximising sensitivity + specificity - 1 (chosen on *source* data)."""
    fpr, tpr, thr = roc_curve(y, p)
    j = tpr - fpr
    t = thr[int(np.argmax(j))]
    return float(min(max(t, 0.0), 1.0))


def bootstrap_ci(y, p, threshold=0.5, groups=None, n_boot=None, seed=C.SEED, alpha=0.05):
    """Percentile bootstrap intervals for every metric.

    groups=None  -> patches are resampled independently (what the earlier study of Paun et al. (2026) did).
    groups=array -> whole groups (slides) are resampled, which respects the correlation
                    between patches cut from the same slide and gives honest, wider intervals.
    """
    n_boot = n_boot or C.N_BOOTSTRAP
    y, p = np.asarray(y), np.asarray(p, float)
    rng = np.random.default_rng(seed)
    names = THRESHOLD_METRICS + SCORE_METRICS
    samples = {k: [] for k in names}

    if groups is not None:
        groups = np.asarray(groups)
        uniq = np.unique(groups)
        members = [np.flatnonzero(groups == g) for g in uniq]

    for _ in range(n_boot):
        if groups is None:
            idx = rng.integers(0, len(y), len(y))
        else:
            pick = rng.integers(0, len(members), len(members))
            idx = np.concatenate([members[i] for i in pick])
        yb, pb = y[idx], p[idx]
        m = threshold_metrics(*confusion(yb, pb >= threshold))
        m.update(score_metrics(yb, pb))
        for k in names:
            samples[k].append(m[k])

    out = {}
    for k in names:
        v = np.asarray(samples[k], float)
        v = v[~np.isnan(v)]
        if len(v) == 0:
            out[f"{k}_lo"], out[f"{k}_hi"], out[f"{k}_sd"] = (float("nan"),) * 3
        else:
            out[f"{k}_lo"] = float(np.quantile(v, alpha / 2))
            out[f"{k}_hi"] = float(np.quantile(v, 1 - alpha / 2))
            out[f"{k}_sd"] = float(np.std(v, ddof=1))
    return out
