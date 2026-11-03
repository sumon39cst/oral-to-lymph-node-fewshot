"""Step 4 (Experiment B) - how little target information closes the cross-domain gap?

For every backbone the source probe (native split, oral images only) is compared with
increasing amounts of lymph node information, all drawn from the *adaptation* pool, whose
slides are disjoint from the slides of the evaluation pool:

    direct              no target data at all (the earlier study's (Paun et al., 2026) setting)
    target_standardise  unlabelled target patches: features re-standardised with target
                        statistics instead of source statistics
    coral               unlabelled: second-order alignment of target to source features
    self_training       unlabelled: confident pseudo-labels on the adaptation pool
    fewshot_k           k labelled target patches per class (k = 1 ... 100), repeated draws
                          - prototype     nearest class mean on target-standardised features
                          - source+shots  source data plus the up-weighted shots
    oracle              all labelled adaptation patches (target-supervised upper bound)

Outputs  tables/adaptation_metrics.csv       one row per backbone / method / k / repeat
         tables/adaptation_summary.csv       mean and SD over repeats
         predictions/adaptation/<backbone>__<method>.csv
"""
import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import config as C
from linear_probe import available_backbones, fit_probe
from metrics import all_metrics, youden_threshold

warnings.filterwarnings("ignore", category=ConvergenceWarning)
KEEP = ("accuracy", "sensitivity", "specificity", "precision", "f1", "balanced_accuracy",
        "mcc", "auroc", "auprc", "ece", "brier", "positive_rate")


def coral(xs, xt_fit, xt_apply, eps=1.0):
    """Align target second-order statistics to the source (Sun et al., 2016)."""
    d = xs.shape[1]
    cs = np.cov(xs, rowvar=False) + eps * np.eye(d)
    ct = np.cov(xt_fit, rowvar=False) + eps * np.eye(d)

    def mat_pow(m, power):
        w, v = np.linalg.eigh(m)
        return (v * np.maximum(w, 1e-8) ** power) @ v.T

    transform = mat_pow(ct, -0.5) @ mat_pow(cs, 0.5)
    mu_t = xt_fit.mean(0)
    return (xt_apply - mu_t) @ transform + xs.mean(0)


def self_training(clf, x_adapt, x_test, c, rounds=3, frac=0.2):
    """Refit on the most confident pseudo-labelled target patches, a few rounds."""
    cur = clf
    for _ in range(rounds):
        p = cur.predict_proba(x_adapt)[:, 1]
        n = int(len(p) * frac)
        order = np.argsort(p)
        sel = np.concatenate([order[:n], order[-n:]])
        pseudo = np.concatenate([np.zeros(n, int), np.ones(n, int)])
        cur = LogisticRegression(C=c, max_iter=500).fit(x_adapt[sel], pseudo)
    return cur.predict_proba(x_test)[:, 1]


def prototype_scores(x_shots, y_shots, x_test):
    """Nearest-class-mean score in [0, 1] from cosine similarities."""
    def unit(a):
        return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)
    mu0 = unit(x_shots[y_shots == 0].mean(0))
    mu1 = unit(x_shots[y_shots == 1].mean(0))
    xt = unit(x_test)
    return 1.0 / (1.0 + np.exp(-10.0 * (xt @ mu1 - xt @ mu0)))


def run_backbone(key, index, adapt, test):
    f_dev = np.load(C.FEATURE_DIR / key / "dev.npy")
    f_adapt = np.load(C.FEATURE_DIR / key / "pcam_adapt.npy")
    f_test = np.load(C.FEATURE_DIR / key / "pcam_test.npy")
    y_dev, y_adapt, y_test = index.label.to_numpy(), adapt.label.to_numpy(), test.label.to_numpy()
    ok = index[index.decoded]
    tr = ok[ok.native_split == "train"].idx.to_numpy()
    va = ok[ok.native_split == "val"].idx.to_numpy()

    scaler, clf, c_best, _ = fit_probe(f_dev[tr], y_dev[tr], f_dev[va], y_dev[va])
    xs = scaler.transform(f_dev[tr])
    thr_src = youden_threshold(y_dev[va], clf.predict_proba(scaler.transform(f_dev[va]))[:, 1])

    tgt_scaler = StandardScaler().fit(f_adapt)             # unlabelled target statistics
    xa_src, xt_src = scaler.transform(f_adapt), scaler.transform(f_test)
    xa_tgt, xt_tgt = tgt_scaler.transform(f_adapt), tgt_scaler.transform(f_test)

    rows, preds = [], {}

    def record(method, p, k=0, rep=0, labels="none", thr=0.5):
        row = dict(backbone=key, family=C.BACKBONES[key]["family"], method=method, k=k,
                   repeat=rep, target_labels=labels, threshold=thr)
        m = all_metrics(y_test, p, thr)
        row.update({name: m[name] for name in KEEP})
        rows.append(row)
        if rep == 0:
            preds[method if k == 0 else f"{method}_k{k}"] = p

    # ---- no target data
    p_direct = clf.predict_proba(xt_src)[:, 1]
    record("direct", p_direct)
    record("direct_source_youden", p_direct, thr=thr_src)

    # ---- unlabelled target data
    record("target_standardise", clf.predict_proba(xt_tgt)[:, 1], labels="unlabelled")
    record("coral", clf.predict_proba(coral(xs, xa_src, xt_src))[:, 1], labels="unlabelled")
    record("self_training", self_training(clf, xa_tgt, xt_tgt, c_best), labels="unlabelled")

    # ---- k labelled target patches per class
    pos, neg = np.flatnonzero(y_adapt == 1), np.flatnonzero(y_adapt == 0)
    for k in C.FEWSHOT_K:
        for rep in range(C.FEWSHOT_REPEATS):
            rng = np.random.default_rng(C.SEED + 1000 * k + rep)
            sel = np.concatenate([rng.choice(neg, k, replace=False),
                                  rng.choice(pos, k, replace=False)])
            ys = y_adapt[sel]
            record("fewshot_prototype", prototype_scores(xa_tgt[sel], ys, xt_tgt), k, rep,
                   "few-shot")
            if rep >= C.FEWSHOT_REPEATS_MIX:      # refitting on the full source set is slow
                continue
            # source + shots: the 2k shots carry as much total weight as the whole source set
            x_mix = np.vstack([xs, xa_src[sel]])
            y_mix = np.concatenate([y_dev[tr], ys])
            w = np.concatenate([np.ones(len(tr)), np.full(2 * k, len(tr) / (2 * k))])
            mix = LogisticRegression(C=c_best, max_iter=300).fit(x_mix, y_mix, sample_weight=w)
            record("fewshot_source_plus_shots", mix.predict_proba(xt_src)[:, 1], k, rep,
                   "few-shot")

    # ---- target-supervised upper bound (also verifies image/label alignment of PCam)
    oracle = LogisticRegression(C=c_best, max_iter=500).fit(xa_tgt, y_adapt)
    record("oracle_target_supervised", oracle.predict_proba(xt_tgt)[:, 1], labels="all")

    out_dir = C.PRED_DIR / "adaptation"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, p in preds.items():
        pd.DataFrame(dict(pcam_index=test.pcam_index, wsi=test.wsi, label=y_test,
                          prob=p)).to_csv(out_dir / f"{key}__{name}.csv", index=False)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None)
    a = ap.parse_args()
    index = pd.read_csv(C.DEV_INDEX_CSV)
    adapt = pd.read_csv(C.CACHE_DIR / "pcam_adapt_index.csv")
    test = pd.read_csv(C.CACHE_DIR / "pcam_test_index.csv")
    keys = [k for k in (a.models or available_backbones())
            if (C.FEATURE_DIR / k / "pcam_adapt.npy").exists()]

    out_csv = C.TABLE_DIR / "adaptation_metrics.csv"
    table = pd.read_csv(out_csv) if out_csv.exists() else pd.DataFrame()
    for key in keys:
        rows = pd.DataFrame(run_backbone(key, index, adapt, test))
        if len(table):
            table = table[table.backbone != key]
        table = pd.concat([table, rows], ignore_index=True)
        table.to_csv(out_csv, index=False)
        s = rows.groupby(["method", "k"])[["auroc", "balanced_accuracy", "sensitivity",
                                           "specificity"]].mean().round(3)
        print(f"[{key}]\n{s}\n")

    summary = table.groupby(["backbone", "family", "method", "k", "target_labels"])[list(KEEP)] \
        .agg(["mean", "std"]).round(4)
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary.reset_index().to_csv(C.TABLE_DIR / "adaptation_summary.csv", index=False)
    print("saved", out_csv)


if __name__ == "__main__":
    main()
