"""Second review round (cancers-4665402): experiments requested by Reviewers 1 and 2, on cached embeddings.

    python rev3_experiments.py reference     R1.1  reconcile the target-supervised reference (0.968 vs 0.976)
    python rev3_experiments.py pool          R1.2  normal-only Phikon, unlabelled pool with 10% vs 50% tumour
    python rev3_experiments.py smallk        R1.3  prototypes vs logistic regression at k = 1, 2 (+ paired tests, Holm)
    python rev3_experiments.py centre        R1.4  cross-centre test with centre-specific standardisation
    python rev3_experiments.py table9        R2.3  F1, MCC, AUPRC and ECE for every support draw
    python rev3_experiments.py noise         R2.5  label noise: why 40% flipped labels hurts so much
    python rev3_experiments.py dedup         R2.7  direct transfer after removing all near-duplicates
    python rev3_experiments.py operating     R1 minor: operating points with thresholds from separate calibration data
    python rev3_experiments.py all

Support draws, standardisation, heads (cosine prototypes with tau = 10; logistic regression with C = 0.01) and
slide-cluster bootstrap are those of the original analyses (code/revision_cpu.py, code/v2_analyses.py).
Results go to Majore Revision/analysis/.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
import config as C  # noqa: E402
from linear_probe import fit_probe  # noqa: E402
from metrics import all_metrics  # noqa: E402
from revision_cpu import D, auc, draw_shots, proto_score, unit  # noqa: E402
from v2_analyses import centre_of, lr_score, slide_boot_index  # noqa: E402

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parents[2] / "results" / "revision_analysis"
OUT.mkdir(parents=True, exist_ok=True)
TAU = 10.0
C_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0)


def sigm(s):
    return 1.0 / (1.0 + np.exp(-TAU * s))


def paired(y, boots, sa_list, sb_list):
    """Mean AUROC difference a - b over draws, slide-bootstrap 95% interval and two-sided p (as in the paper)."""
    point = np.mean([auc(y, a) - auc(y, b) for a, b in zip(sa_list, sb_list)])
    diffs = np.array([np.mean([auc(y[i], a[i]) - auc(y[i], b[i]) for a, b in zip(sa_list, sb_list)]) for i in boots])
    p = max(2 * min((diffs <= 0).mean(), (diffs >= 0).mean()), 1 / len(boots))
    return point, np.quantile(diffs, .025), np.quantile(diffs, .975), p


def holm(pvals):
    p = np.asarray(pvals, float)
    order, adj, run = np.argsort(p), np.empty(len(p)), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(p) - r) * p[i]))
        adj[i] = run
    return adj


# ===================================================================================== R1.1 reference
def reference():
    rows = []
    for key in ("phikon", "phikon_v2") + (("midnight",) if (C.FEATURE_DIR / "midnight" / "pcam_adapt.npy").exists() else ()):
        d = D(key)
        _, _, c_src = d.source_probe()                     # C chosen on the ORAL validation fold (old reference)
        groups = d.adapt.wsi.to_numpy()
        cv = {}
        for c in C_GRID:                                   # target-only choice: 5-fold CV over adaptation slides
            sc = []
            for tr, va in GroupKFold(5).split(d.za, d.y_adapt, groups):
                m = LogisticRegression(C=c, max_iter=1000).fit(d.za[tr], d.y_adapt[tr])
                sc.append(auc(d.y_adapt[va], m.decision_function(d.za[va])))
            cv[c] = np.mean(sc)
        c_cv = max(cv, key=cv.get)
        for name, c in (("C from oral validation fold (first submission, Table 6)", c_src),
                        ("C = 0.01 (few-shot setting, Figure 9c)", 0.01),
                        ("C by 5-fold slide-grouped CV on the adaptation pool (revised reference)", c_cv)):
            s = LogisticRegression(C=c, max_iter=1000).fit(d.za, d.y_adapt).decision_function(d.zt)
            rows.append(dict(encoder=key, reference=name, C=c, auroc=auc(d.y_test, s),
                             **{f"cv_auroc_C={g}": v for g, v in cv.items()}))
        print(pd.DataFrame(rows[-3:])[["encoder", "reference", "C", "auroc"]].round(4).to_string(index=False), flush=True)
    r = pd.DataFrame(rows)
    r.round(4).to_csv(OUT / "R1_1_reference_reconciliation.csv", index=False)
    # fraction of achievable improvement (Eq. 9) with the revised reference
    ad = pd.read_csv(C.TABLE_DIR / "adaptation_metrics.csv")
    g = []
    for key in r.encoder.unique():
        sup = r[(r.encoder == key) & r.reference.str.startswith("C by")].auroc.iloc[0]
        a = ad[(ad.backbone == key) & (ad.method == "fewshot_prototype")]
        for k in (10, 50):
            if len(a[a.k == k]):
                ak = a[a.k == k].auroc.mean()
                g.append(dict(encoder=key, k=k, auroc_k=ak, reference=sup, G=(ak - .5) / (sup - .5)))
    pd.DataFrame(g).round(3).to_csv(OUT / "R1_1_gap_closed_revised_reference.csv", index=False)
    print(pd.DataFrame(g).round(3).to_string(index=False))


# ===================================================================================== R1.2 pool composition
def pool():
    d = D("phikon")
    y, ya = d.y_test, d.y_adapt
    neg, pos = np.flatnonzero(ya == 0), np.flatnonzero(ya == 1)
    N = 9_000                                             # same total size in every condition (<= 9,981 normals)
    rows = []
    for frac in (0.0, 0.01, 0.10, 0.25, 0.50):
        for rep in range(20):
            rng_s = np.random.default_rng(C.SEED + 7 * 50 + rep)     # identical to the original normal-only draws
            sup0 = rng_s.choice(neg, 50, replace=False)
            sup2 = draw_shots(ya, 50, rep)                            # two-class support of the main analysis
            rng = np.random.default_rng(C.SEED + 9_000 + int(frac * 1000) * 31 + rep)
            n_pos = int(round(frac * N))
            pool_idx = np.concatenate([rng.choice(neg, N - n_pos, replace=False), rng.choice(pos, n_pos, replace=False)])
            sc = StandardScaler().fit(d.f_adapt[pool_idx])
            za, zt = sc.transform(d.f_adapt), sc.transform(d.f_test)
            c0 = unit(za[sup0].mean(0))
            rows.append(dict(tumour_fraction=frac, pool_size=N, rep=rep, head="normal-only centroid (-s0), k = 50 normals",
                             auroc=auc(y, -(unit(zt) @ c0))))
            rows.append(dict(tumour_fraction=frac, pool_size=N, rep=rep, head="two-class prototypes, k = 50 per class",
                             auroc=auc(y, proto_score(za[sup2], ya[sup2], zt))))
    for rep in range(20):                                             # original condition: all 20,000 (50.1% tumour)
        rng_s = np.random.default_rng(C.SEED + 7 * 50 + rep)
        sup0 = rng_s.choice(neg, 50, replace=False)
        rows.append(dict(tumour_fraction=float(ya.mean()), pool_size=len(ya), rep=rep,
                         head="normal-only centroid (-s0), k = 50 normals", auroc=auc(y, -(unit(d.zt) @ unit(d.za[sup0].mean(0))))))
    # no pool at all: statistics of the 50 labelled normals only (one-class)
    for rep in range(20):
        rng_s = np.random.default_rng(C.SEED + 7 * 50 + rep)
        sup0 = rng_s.choice(neg, 50, replace=False)
        sc = StandardScaler().fit(d.f_adapt[sup0])
        zt = sc.transform(d.f_test)
        rows.append(dict(tumour_fraction=np.nan, pool_size=0, rep=rep, head="one-class, statistics of the 50 normals only",
                         auroc=auc(y, np.linalg.norm(zt, axis=1))))
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "R1_2_pool_composition_all.csv", index=False)
    s = r.groupby(["head", "pool_size", "tumour_fraction"], dropna=False).auroc.agg(["mean", "std", "min", "max"]).round(3)
    s.to_csv(OUT / "R1_2_pool_composition_summary.csv")
    print(s.to_string())
    # paired: 10% vs 50% at the same pool size, per draw
    p = r[(r.pool_size == N) & r["head"].str.startswith("normal-only")].pivot(index="rep", columns="tumour_fraction", values="auroc")
    dlt = p[0.10] - p[0.50]
    print(f"normal-only, 10% - 50% tumour in pool: mean {dlt.mean():+.4f}, SD {dlt.std():.4f}, range {dlt.min():+.4f}..{dlt.max():+.4f}")
    pd.DataFrame(dict(rep=dlt.index, diff_10_minus_50=dlt.values)).to_csv(OUT / "R1_2_pool_paired_diff.csv", index=False)


# ===================================================================================== R1.3 small k
def smallk(n_boot=500, n_draws=10):
    d = D("phikon")
    y, wsi = d.y_test, d.test.wsi.to_numpy()
    rows, tests = [], []
    for k in (1, 2, 5, 10, 20, 50, 100):
        pa, pb = [], []
        for rep in range(20):
            sel = draw_shots(d.y_adapt, k, rep)
            ys = d.y_adapt[sel]
            s_p = proto_score(d.za[sel], ys, d.zt)
            s_l = lr_score(d.za[sel], ys, d.zt)
            rows.append(dict(k=k, rep=rep, prototypes=auc(y, s_p), logistic_regression=auc(y, s_l)))
            if rep < n_draws:
                pa.append(s_l)
                pb.append(s_p)
        if k in (1, 2):                                    # new paired tests (k >= 5 are in the existing family)
            boots = slide_boot_index(wsi, n_boot)
            pt, lo, hi, p = paired(y, boots, pa, pb)
            tests.append(dict(comparison="head", k=k, auroc_logreg_minus_centroid=pt, ci_lo=lo, ci_hi=hi, p_value=p))
            print(tests[-1], flush=True)
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "R1_3_small_k_all_draws.csv", index=False)
    s = r.groupby("k")[["prototypes", "logistic_regression"]].agg(["mean", "std"]).round(3)
    s.to_csv(OUT / "R1_3_small_k_summary.csv")
    print(s.to_string())
    # win counts per draw
    w = r.assign(lr_better=r.logistic_regression > r.prototypes).groupby("k").lr_better.agg(["sum", "size"])
    w.to_csv(OUT / "R1_3_small_k_win_counts.csv")
    print(w.to_string())
    # Holm over the whole family of paired comparisons (21 of the first submission + 2 new)
    old = pd.read_csv(C.ROOT / "paper" / "version2" / "analysis" / "C6_holm_adjusted.csv")
    fam = pd.concat([old[["comparison", "k", "p_value"]], pd.DataFrame(tests)[["comparison", "k", "p_value"]]],
                    ignore_index=True)
    fam["p_holm"] = holm(fam.p_value)
    fam.round(4).to_csv(OUT / "R1_3_holm_family_23.csv", index=False)
    pd.DataFrame(tests).round(4).to_csv(OUT / "R1_3_paired_k1_k2.csv", index=False)
    print(fam.round(4).to_string(index=False))


# ===================================================================================== R1.4 cross-centre
def centre():
    rows = []
    for key in ("phikon", "phikon_v2"):
        d = D(key)
        ca = d.adapt.wsi.map(centre_of).to_numpy()
        ct = d.test.wsi.map(centre_of).to_numpy()
        for src in ("Radboud", "Utrecht"):
            pool_idx = np.flatnonzero(ca == src)
            for std_name, sc in (("pool of both centres (first submission)", d.tsc),
                                 ("pool of the support centre only", StandardScaler().fit(d.f_adapt[pool_idx]))):
                za, zt = sc.transform(d.f_adapt), sc.transform(d.f_test)
                for tgt in ("Radboud", "Utrecht"):
                    ev = np.flatnonzero(ct == tgt)
                    for k in (10, 50):
                        res = {"prototypes": [], "logistic regression": []}
                        for rep in range(20):
                            rng = np.random.default_rng(C.SEED + 100 * k + rep)        # same draws as before
                            sel = np.concatenate([rng.choice(pool_idx[d.y_adapt[pool_idx] == c], k, replace=False)
                                                  for c in (0, 1)])
                            ys = d.y_adapt[sel]
                            res["prototypes"].append(auc(d.y_test[ev], proto_score(za[sel], ys, zt[ev])))
                            res["logistic regression"].append(auc(d.y_test[ev], lr_score(za[sel], ys, zt[ev])))
                        for head, v in res.items():
                            rows.append(dict(encoder=key, standardisation=std_name, support_centre=src,
                                             evaluation_centre=tgt, k=k, head=head, auroc_mean=np.mean(v),
                                             auroc_sd=np.std(v, ddof=1), n_pool=len(pool_idx) if "only" in std_name else len(ca)))
    r = pd.DataFrame(rows).round(3)
    r.to_csv(OUT / "R1_4_cross_centre_standardisation.csv", index=False)
    pc = pd.Series(D("phikon").adapt.wsi.map(centre_of)).value_counts()
    pc.to_csv(OUT / "R1_4_adaptation_pool_centres.csv")
    print(pc.to_string())
    print(r[(r.encoder == "phikon") & (r.k == 50)].to_string(index=False))


# ===================================================================================== R2.3 Table 9 spreads
def table9():
    d = D("phikon")
    rows = []
    for k in (1, 2, 5, 10, 20, 50, 100):
        for rep in range(20):
            sel = draw_shots(d.y_adapt, k, rep)
            p = sigm(proto_score(d.za[sel], d.y_adapt[sel], d.zt))
            m = all_metrics(d.y_test, p, 0.5)
            rows.append(dict(k=k, rep=rep, **{n: m[n] for n in ("accuracy", "sensitivity", "specificity", "f1", "mcc",
                                                               "auroc", "auprc", "ece", "balanced_accuracy")}))
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "R2_3_table9_all_draws.csv", index=False)
    s = r.groupby("k").agg(["mean", "std"]).drop(columns="rep").round(3)
    s.to_csv(OUT / "R2_3_table9_mean_sd.csv")
    print(s.to_string())


# ===================================================================================== R2.5 label noise
def noise():
    d = D("phikon")
    y = d.y_test
    rows = []
    # clean class-mean difference from all 20,000 labelled adaptation patches (direction the support tries to estimate)
    full = d.za[d.y_adapt == 1].mean(0) - d.za[d.y_adapt == 0].mean(0)
    for rho in (0.0, 0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5):
        for rep in range(20):
            sel = draw_shots(d.y_adapt, 50, rep)
            ys = d.y_adapt[sel].copy()
            rng = np.random.default_rng(C.SEED + rep)                  # same flipping as the first submission
            flip = rng.choice(len(ys), int(round(rho * len(ys))), replace=False)
            ys[flip] = 1 - ys[flip]
            yt = d.y_adapt[sel]
            # composition of each noisy class: share of truly metastatic patches
            share_pos_in_1 = yt[ys == 1].mean() if (ys == 1).any() else np.nan
            share_pos_in_0 = yt[ys == 0].mean() if (ys == 0).any() else np.nan
            row = dict(rho=rho, rep=rep, n_labelled_1=int((ys == 1).sum()), n_labelled_0=int((ys == 0).sum()),
                       contrast=share_pos_in_1 - share_pos_in_0)          # expected 1 - 2*rho
            if 0 < ys.sum() < len(ys):
                diff = d.za[sel][ys == 1].mean(0) - d.za[sel][ys == 0].mean(0)
                row["cos_to_full_direction"] = float(diff @ full / (np.linalg.norm(diff) * np.linalg.norm(full)))
                row["prototypes"] = auc(y, proto_score(d.za[sel], ys, d.zt))
                row["logistic_regression"] = auc(y, lr_score(d.za[sel], ys, d.zt))
            rows.append(row)
    # clean supports of k_eff = k (1 - 2 rho)^2 for comparison (the effective sample size argument)
    clean = []
    for k in (1, 2, 5, 10, 20, 50):
        for rep in range(20):
            sel = draw_shots(d.y_adapt, k, rep)
            diff = d.za[sel][d.y_adapt[sel] == 1].mean(0) - d.za[sel][d.y_adapt[sel] == 0].mean(0)
            clean.append(dict(k=k, rep=rep, cos_to_full_direction=float(diff @ full / (np.linalg.norm(diff) * np.linalg.norm(full))),
                              prototypes=auc(y, proto_score(d.za[sel], d.y_adapt[sel], d.zt))))
    r, c = pd.DataFrame(rows), pd.DataFrame(clean)
    r.to_csv(OUT / "R2_5_label_noise_all.csv", index=False)
    c.to_csv(OUT / "R2_5_clean_small_k_direction.csv", index=False)
    s = r.groupby("rho")[["contrast", "cos_to_full_direction", "prototypes", "logistic_regression"]].agg(["mean", "std"]).round(3)
    s["k_eff"] = (50 * (1 - 2 * s.index.to_numpy()) ** 2).round(1)
    s.to_csv(OUT / "R2_5_label_noise_summary.csv")
    print(s.to_string())
    print(c.groupby("k")[["cos_to_full_direction", "prototypes"]].agg(["mean", "std"]).round(3).to_string())


# ===================================================================================== R2.7 deduplication
def dedup():
    idx = pd.read_csv(C.DEV_INDEX_CSV)
    pairs = pd.read_csv(C.RESULT_DIR / "revision" / "M6_near_duplicate_pairs.csv")
    # summary of where the pairs fall (training / validation / test, within or across collections)
    pairs["splits"] = [" - ".join(sorted((a, b))) for a, b in zip(pairs.split_i, pairs.split_j)]
    pairs["collections"] = np.where(pairs.ds_i == pairs.ds_j, "within collection", "across collections")
    summary = pairs.groupby(["collections", "splits"]).size().rename("pairs").reset_index()
    summary.to_csv(OUT / "R2_7_duplicate_pairs_by_split.csv", index=False)
    print(summary.to_string(index=False))
    # connected groups of near-duplicates; keep one image per group, preferring a training image
    parent = {}

    def find(a):
        while parent.setdefault(a, a) != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for i, j in zip(pairs.i, pairs.j):
        parent[find(i)] = find(j)
    groups = {}
    for n in set(pairs.i) | set(pairs.j):
        groups.setdefault(find(n), []).append(n)
    rank = {"train": 0, "val": 1, "test": 2}
    drop = set()
    for g in groups.values():
        keep = min(g, key=lambda n: (rank[idx.native_split[n]], n))
        drop |= set(g) - {keep}
    info = dict(images_in_pairs=len(set(pairs.i) | set(pairs.j)), groups=len(groups), images_removed=len(drop))
    removed_by_split = idx.loc[sorted(drop)].native_split.value_counts().to_dict()
    info.update({f"removed_{k}": v for k, v in removed_by_split.items()})
    pd.Series(info).to_csv(OUT / "R2_7_dedup_counts.csv")
    print(info)
    keep = ~idx.idx.isin(drop)
    rows = []
    for key in C.DEFAULT_BACKBONES + (["midnight"] if (C.FEATURE_DIR / "midnight" / "dev.npy").exists() else []):
        d = D(key)
        for name, mask in (("all images (first submission)", np.ones(len(idx), bool)), ("near-duplicates removed", keep.to_numpy())):
            tr = d.tr[mask[d.tr]]
            va = d.va[mask[d.va]]
            te = d.te[mask[d.te]]
            sc, clf, c, _ = fit_probe(d.f_dev[tr], d.y_dev[tr], d.f_dev[va], d.y_dev[va])
            rows.append(dict(encoder=key, source_data=name, n_train=len(tr), n_val=len(va), n_test=len(te), C=c,
                             internal_auroc=auc(d.y_dev[te], clf.predict_proba(sc.transform(d.f_dev[te]))[:, 1]),
                             external_auroc=auc(d.y_test, clf.predict_proba(sc.transform(d.f_test))[:, 1])))
        print(pd.DataFrame(rows[-2:]).round(3).to_string(index=False), flush=True)
    pd.DataFrame(rows).round(4).to_csv(OUT / "R2_7_direct_transfer_deduplicated.csv", index=False)


# ===================================================================================== R1 minor: operating points
def operating():
    """Sensitivity at fixed specificity (i) read retrospectively from the evaluation ROC curve, as in Table 12, and
    (ii) at thresholds chosen on a separate labelled calibration set from the adaptation pool (250 normal patches,
    disjoint from the support), then applied unchanged to the evaluation pool."""
    d = D("phikon")
    y = d.y_test
    neg = np.flatnonzero(d.y_adapt == 0)
    rows = []
    for rep in range(20):
        sel = draw_shots(d.y_adapt, 50, rep)
        rng = np.random.default_rng(C.SEED + 555 + rep)
        cal = rng.choice(np.setdiff1d(neg, sel), 250, replace=False)
        for head, fn in (("prototypes", lambda z: proto_score(d.za[sel], d.y_adapt[sel], z)),
                         ("logistic regression", lambda z: lr_score(d.za[sel], d.y_adapt[sel], z))):
            s_e, s_c = fn(d.zt), fn(d.za[cal])
            for spec in (0.90, 0.95, 0.98):
                thr_retro = np.quantile(s_e[y == 0], spec)
                thr_cal = np.quantile(s_c, spec)
                rows.append(dict(rep=rep, head=head, target_specificity=spec,
                                 retro_sensitivity=(s_e[y == 1] > thr_retro).mean(),
                                 cal_sensitivity=(s_e[y == 1] > thr_cal).mean(),
                                 cal_realised_specificity=(s_e[y == 0] <= thr_cal).mean()))
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "R1_minor_operating_points_all.csv", index=False)
    s = r.groupby(["head", "target_specificity"])[["retro_sensitivity", "cal_sensitivity", "cal_realised_specificity"]] \
        .agg(["mean", "std"]).round(3)
    s.to_csv(OUT / "R1_minor_operating_points_summary.csv")
    print(s.to_string())


# ===================================================================================== R1.1 revised reference, all encoders
def reference_all():
    """Revised target-supervised reference for every encoder (Table 6 'Reference' column, Table 13 row, Figures 5-6):
    logistic regression on all 20,000 standardised adaptation patches, C chosen by 5-fold slide-grouped CV on the
    adaptation pool (no evaluation data), threshold 0.5 on the predicted probability."""
    rows = []
    keys = C.DEFAULT_BACKBONES + (["midnight"] if (C.FEATURE_DIR / "midnight" / "pcam_adapt.npy").exists() else [])
    for key in keys:
        d = D(key)
        groups = d.adapt.wsi.to_numpy()
        cv = {}
        for c in C_GRID:
            sc = []
            for tr, va in GroupKFold(5).split(d.za, d.y_adapt, groups):
                m = LogisticRegression(C=c, max_iter=1000).fit(d.za[tr], d.y_adapt[tr])
                sc.append(auc(d.y_adapt[va], m.decision_function(d.za[va])))
            cv[c] = float(np.mean(sc))
        c_cv = max(cv, key=cv.get)
        m = LogisticRegression(C=c_cv, max_iter=1000).fit(d.za, d.y_adapt)
        p = m.predict_proba(d.zt)[:, 1]
        met = all_metrics(d.y_test, p, 0.5)
        rows.append(dict(encoder=key, C=c_cv, **{n: met[n] for n in ("accuracy", "sensitivity", "specificity", "precision",
                                                                    "f1", "balanced_accuracy", "mcc", "auroc", "auprc")}))
        if key in ("phikon", "phikon_v2", "midnight"):
            pd.DataFrame(dict(pcam_index=d.test.pcam_index, slide=d.test.wsi, label=d.y_test, probability=p)).to_csv(
                OUT / f"reference_cv_{key}.csv", index=False)
        print(key, c_cv, round(met["auroc"], 4), flush=True)
    pd.DataFrame(rows).round(4).to_csv(OUT / "R1_1_reference_all_encoders.csv", index=False)


if __name__ == "__main__":
    jobs = dict(reference=reference, pool=pool, smallk=smallk, centre=centre, table9=table9, noise=noise, dedup=dedup,
                operating=operating, reference_all=reference_all)
    names = list(jobs) if sys.argv[1:] == ["all"] else sys.argv[1:]
    for n in names:
        print(f"\n=================== {n}", flush=True)
        jobs[n]()
