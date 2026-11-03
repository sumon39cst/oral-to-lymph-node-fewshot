"""Step 3 (Experiment A) - frozen-backbone linear probe, internal vs. external performance.

Same protocol as the earlier study of Paun et al. (2026) (frozen encoder + trained head), applied to every
backbone, under three ways of splitting the development data:

    random   stratified 60/20/20 over the pooled images, repeated over seeds.
             This is the earlier study's (Paun et al., 2026) split. It is leakage-prone (augmented copies and
             same-patient images can land on both sides) and is kept for comparability only.
    native   each dataset's own train / val / test folders. Leakage-safe by construction.
    lodo_dsK leave-one-dataset-out: train on two oral datasets, test on the third.
             Measures cross-dataset generalisation *inside* the oral domain.

Every trained probe is then applied unchanged to the PatchCamelyon evaluation pool
(external, out-of-distribution). Two operating points are reported, both fixed using
source data only: 0.5, and the Youden-optimal threshold on the source validation fold.

Outputs  tables/linear_probe_metrics.csv
         predictions/linear_probe/<backbone>__<protocol>[__seedS]__<eval_set>.csv
"""
import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import config as C
from metrics import all_metrics, bootstrap_ci, youden_threshold

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def available_backbones():
    return [k for k in C.BACKBONES if (C.FEATURE_DIR / k / "dev.npy").exists()
            and (C.FEATURE_DIR / k / "pcam_test.npy").exists()]


def make_splits(index: pd.DataFrame):
    """Yield (protocol, seed, train_idx, val_idx, test_idx)."""
    ok = index[index.decoded].copy()
    idx = ok.idx.to_numpy()
    strat = (ok.dataset + "_" + ok.label.astype(str)).to_numpy()
    pos = np.arange(len(ok))
    for seed in C.SEEDS:
        tr, rest = train_test_split(pos, test_size=0.4, random_state=seed, stratify=strat)
        va, te = train_test_split(rest, test_size=0.5, random_state=seed, stratify=strat[rest])
        yield "random", seed, idx[tr], idx[va], idx[te]

    yield ("native", -1, ok[ok.native_split == "train"].idx.to_numpy(),
           ok[ok.native_split == "val"].idx.to_numpy(),
           ok[ok.native_split == "test"].idx.to_numpy())

    for held in ("ds1", "ds2", "ds3"):
        src = ok[ok.dataset != held]
        yield (f"lodo_{held}", -1, src[src.native_split != "val"].idx.to_numpy(),
               src[src.native_split == "val"].idx.to_numpy(),
               ok[ok.dataset == held].idx.to_numpy())


def fit_probe(x_tr, y_tr, x_va, y_va):
    """Standardise, then L2 logistic regression with C chosen on validation AUROC."""
    scaler = StandardScaler().fit(x_tr)
    xt, xv = scaler.transform(x_tr), scaler.transform(x_va)
    best = None
    for c in C.LOGREG_C_GRID:
        clf = LogisticRegression(C=c, max_iter=500, class_weight="balanced").fit(xt, y_tr)
        auc = roc_auc_score(y_va, clf.predict_proba(xv)[:, 1])
        if best is None or auc > best[0]:
            best = (auc, c, clf)
    return scaler, best[2], best[1], best[0]


def run_backbone(key, index, pcam, n_boot):
    f_dev = np.load(C.FEATURE_DIR / key / "dev.npy")
    f_ext = np.load(C.FEATURE_DIR / key / "pcam_test.npy")
    y_dev = index.label.to_numpy()
    y_ext, wsi_ext = pcam.label.to_numpy(), pcam.wsi.to_numpy()
    slide_groups = wsi_ext if len(np.unique(wsi_ext)) > 1 else None
    rows = []
    out_dir = C.PRED_DIR / "linear_probe"
    out_dir.mkdir(parents=True, exist_ok=True)

    for protocol, seed, tr, va, te in make_splits(index):
        scaler, clf, c_best, val_auc = fit_probe(f_dev[tr], y_dev[tr], f_dev[va], y_dev[va])
        p_val = clf.predict_proba(scaler.transform(f_dev[va]))[:, 1]
        p_int = clf.predict_proba(scaler.transform(f_dev[te]))[:, 1]
        p_ext = clf.predict_proba(scaler.transform(f_ext))[:, 1]
        thresholds = {"fixed_0.5": 0.5, "source_youden": youden_threshold(y_dev[va], p_val)}

        tag = f"{key}__{protocol}" + (f"__seed{seed}" if seed >= 0 else "")
        pd.DataFrame(dict(idx=te, dataset=index.dataset.to_numpy()[te], label=y_dev[te],
                          prob=p_int)).to_csv(out_dir / f"{tag}__internal.csv", index=False)
        pd.DataFrame(dict(pcam_index=pcam.pcam_index, wsi=wsi_ext, label=y_ext,
                          prob=p_ext)).to_csv(out_dir / f"{tag}__external.csv", index=False)

        want_ci = protocol == "native" and n_boot > 0       # headline protocol only
        for eval_set, y, p, groups in (("internal", y_dev[te], p_int, None),
                                       ("external", y_ext, p_ext, slide_groups)):
            for rule, thr in thresholds.items():
                row = dict(backbone=key, family=C.BACKBONES[key]["family"], protocol=protocol,
                           seed=seed, eval_set=eval_set, threshold_rule=rule, C=c_best,
                           val_auroc=val_auc, n_train=len(tr))
                row.update(all_metrics(y, p, thr))
                if want_ci:
                    row.update(bootstrap_ci(y, p, thr, groups=groups, n_boot=n_boot))
                    row["ci_method"] = "slide-cluster bootstrap" if groups is not None \
                        else "patch bootstrap"
                rows.append(row)
        ext = [r for r in rows if r["protocol"] == protocol and r["seed"] == seed
               and r["eval_set"] == "external" and r["threshold_rule"] == "fixed_0.5"][-1]
        inte = [r for r in rows if r["protocol"] == protocol and r["seed"] == seed
                and r["eval_set"] == "internal" and r["threshold_rule"] == "fixed_0.5"][-1]
        print(f"  {tag:<44} internal acc {inte['accuracy']:.3f} auc {inte['auroc']:.3f} | "
              f"external acc {ext['accuracy']:.3f} sens {ext['sensitivity']:.3f} "
              f"spec {ext['specificity']:.3f} auc {ext['auroc']:.3f}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--n_boot", type=int, default=C.N_BOOTSTRAP)
    a = ap.parse_args()

    index = pd.read_csv(C.DEV_INDEX_CSV)
    pcam = pd.read_csv(C.CACHE_DIR / "pcam_test_index.csv")
    keys = a.models or available_backbones()
    out_csv = C.TABLE_DIR / "linear_probe_metrics.csv"
    old = pd.read_csv(out_csv) if out_csv.exists() else pd.DataFrame()

    for key in keys:
        print(f"[{key}]")
        rows = pd.DataFrame(run_backbone(key, index, pcam, a.n_boot))
        if len(old):
            old = old[old.backbone != key]
        old = pd.concat([old, rows], ignore_index=True)
        old.to_csv(out_csv, index=False)
    print("saved", out_csv)


if __name__ == "__main__":
    main()
