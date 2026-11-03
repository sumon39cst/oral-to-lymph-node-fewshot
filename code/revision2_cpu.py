"""Round-2 revision analyses on cached embeddings (comments R1, R3, R5 and minor points).

    python revision2_cpu.py        -> result_prediction/revision/R2_*.csv
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import config as C
from linear_probe import fit_probe
from metrics import all_metrics
from revision_cpu import D, auc, draw_shots, proto_score, unit

warnings.filterwarnings("ignore")
REV = C.RESULT_DIR / "revision"
ENC = C.DEFAULT_BACKBONES
H_VECTOR = np.array([0.65, 0.70, 0.29]) / np.linalg.norm([0.65, 0.70, 0.29])   # haematoxylin (Ruifrok)


def density_proxies(x):
    """Two crude nuclear-density proxies per image: dark-pixel fraction and haematoxylin optical density."""
    dark, hod = np.empty(len(x), np.float32), np.empty(len(x), np.float32)
    for s in range(0, len(x), 2000):
        b = np.asarray(x[s:s + 2000]).astype(np.float32)
        lum = 0.299 * b[..., 0] + 0.587 * b[..., 1] + 0.114 * b[..., 2]
        dark[s:s + len(b)] = (lum < 110).reshape(len(b), -1).mean(1)
        od = -np.log((b + 1) / 256.0)
        hod[s:s + len(b)] = (od @ H_VECTOR).reshape(len(b), -1).mean(1)
    return dark, hod


def cluster_boot(y, s, groups, n=1000, seed=C.SEED):
    rng = np.random.default_rng(seed)
    u = np.unique(groups)
    members = [np.flatnonzero(groups == g) for g in u]
    vals = []
    for _ in range(n):
        idx = np.concatenate([members[i] for i in rng.integers(0, len(u), len(u))])
        if 0 < y[idx].sum() < len(idx):
            vals.append(auc(y[idx], s[idx]))
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


# ============================================================================== R1 + minor 7
def r1_mechanism():
    x = np.load(C.CACHE_DIR / "pcam_test_x.npy", mmap_mode="r")
    dark, hod = density_proxies(x)
    dirs = pd.read_csv(REV / "M1_direction_and_reverse.csv").set_index("encoder")
    rows = []
    rng = np.random.default_rng(C.SEED)
    for key in ENC:
        d = D(key)
        y = d.y_test
        sc, clf, c = d.source_probe()
        p = clf.predict_proba(sc.transform(d.f_test))[:, 1]
        normal = y == 0
        dim = d.f_test.shape[1]
        rand = np.abs(unit(rng.standard_normal((2000, dim))) @ unit(rng.standard_normal(dim)))
        row = dict(encoder=key, dim=dim, cos_wS_wT=dirs.loc[key, "cos_wS_wT"],
                   cos_class_means=dirs.loc[key, "cos_class_mean_directions"],
                   null_expected_abs_cos=float(np.sqrt(2 / (np.pi * dim))), null_95pct_abs_cos=float(np.quantile(rand, 0.95)))
        for name, prox in (("dark", dark), ("hod", hod)):
            lo, hi = np.quantile(prox[normal], [1 / 3, 2 / 3])
            sel = normal & ((prox <= lo) | (prox >= hi))
            row[f"auroc_score_for_dense_normals_{name}"] = auc((prox[sel] >= hi).astype(int), p[sel])
            row[f"spearman_score_density_normals_{name}"] = float(pd.Series(p[normal]).corr(pd.Series(prox[normal]), method="spearman"))
            row[f"auroc_density_for_metastasis_{name}"] = auc(y, prox.astype(float))
            t = pd.qcut(prox[normal], 3, labels=["low", "mid", "high"]).astype(str)
            for lab in ("low", "mid", "high"):
                row[f"fp_{name}_{lab}"] = float((p[normal] >= 0.5)[t == lab].mean())
        row["auroc_direct"] = auc(y, p)
        rows.append(row)
        print(f"[{key}] dense-normal AUROC {row['auroc_score_for_dense_normals_dark']:.3f} (H-OD {row['auroc_score_for_dense_normals_hod']:.3f}) "
              f"density->label AUROC {row['auroc_density_for_metastasis_dark']:.3f}/{row['auroc_density_for_metastasis_hod']:.3f} "
              f"null|cos| mean {row['null_expected_abs_cos']:.3f} p95 {row['null_95pct_abs_cos']:.3f} "
              f"FP(H-OD) {row['fp_hod_low']:.2f}/{row['fp_hod_mid']:.2f}/{row['fp_hod_high']:.2f}", flush=True)
    pd.DataFrame(rows).round(4).to_csv(REV / "R2_R1_density_mechanism_and_cosine_null.csv", index=False)


# ============================================================================== R3: lung and colon
def r3_tissues():
    index = pd.read_csv(C.DEV_INDEX_CSV)
    y_dev = index.label.to_numpy()
    tr = index[index.native_split == "train"].idx.to_numpy()
    va = index[index.native_split == "val"].idx.to_numpy()
    rows = []
    for tissue in ("lung", "colon"):
        meta = pd.read_csv(C.CACHE_DIR / f"{tissue}_eval_index.csv")
        y = meta.label.to_numpy()
        x = np.load(C.CACHE_DIR / f"{tissue}_eval_x.npy", mmap_mode="r")
        dark, hod = density_proxies(x)
        # groups of augmented copies: 250 original images per class in LC25000
        ph = unit(np.load(C.FEATURE_DIR / "phikon" / f"{tissue}_eval.npy"))
        groups = np.empty(len(y), int)
        for c in (0, 1):
            m = np.flatnonzero(y == c)
            lab = AgglomerativeClustering(n_clusters=250, linkage="average", metric="cosine").fit_predict(ph[m])
            groups[m] = lab + 250 * c
        for key in ENC:
            f_dev = np.load(C.FEATURE_DIR / key / "dev.npy")
            fe = np.load(C.FEATURE_DIR / key / f"{tissue}_eval.npy")
            sc, clf, _, _ = fit_probe(f_dev[tr], y_dev[tr], f_dev[va], y_dev[va])
            p = clf.predict_proba(sc.transform(fe))[:, 1]
            lo, hi = cluster_boot(y, p, groups)
            m = all_metrics(y, p)
            rows.append(dict(tissue=tissue, encoder=key, auroc=m["auroc"], ci_lo=lo, ci_hi=hi,
                             sensitivity=m["sensitivity"], specificity=m["specificity"],
                             auroc_density_for_malignant_dark=auc(y, dark.astype(float)),
                             auroc_density_for_malignant_hod=auc(y, hod.astype(float)),
                             spearman_score_density_dark=float(pd.Series(p).corr(pd.Series(dark), method="spearman")),
                             spearman_score_density_hod=float(pd.Series(p).corr(pd.Series(hod), method="spearman")),
                             spearman_score_density_benign_only=float(pd.Series(p[y == 0]).corr(pd.Series(hod[y == 0]), method="spearman"))))
            print(f"{tissue:<6}{key:<22} AUROC {m['auroc']:.3f} ({lo:.3f}-{hi:.3f}) density->malignant {rows[-1]['auroc_density_for_malignant_hod']:.3f} "
                  f"rho(score,density) {rows[-1]['spearman_score_density_hod']:+.2f}", flush=True)
    pd.DataFrame(rows).round(4).to_csv(REV / "R2_R3_lung_colon_clustered_ci_and_density.csv", index=False)


# ============================================================================== R5: field of view
def r5_fov():
    index = pd.read_csv(C.DEV_INDEX_CSV)
    y_dev = index.label.to_numpy()
    pc = pd.read_csv(C.CACHE_DIR / "pcam_test_index.csv")
    y_pc, wsi = pc.label.to_numpy(), pc.wsi.to_numpy()
    splits = {"all datasets": (index.native_split, np.ones(len(index), bool)),
              "DS3 only (400x)": (index.native_split, (index.dataset == "ds3").to_numpy())}
    rows = []
    for key in ("efficientnet_b0", "phikon", "phikon_v2"):
        f_pc = np.load(C.FEATURE_DIR / key / "pcam_test.npy")
        feats = {"full field": "dev", "1/2 field": "dev_fov50", "1/4 field": "dev_fov25", "1/8 field": "dev_fov12"}
        for sname, (split, mask) in splits.items():
            tr = index[(split == "train") & mask].idx.to_numpy()
            va = index[(split == "val") & mask].idx.to_numpy()
            te = index[(split == "test") & mask].idx.to_numpy()
            for fname, fkey in feats.items():
                f = np.load(C.FEATURE_DIR / key / f"{fkey}.npy")
                sc, clf, _, _ = fit_probe(f[tr], y_dev[tr], f[va], y_dev[va])
                p = clf.predict_proba(sc.transform(f_pc))[:, 1]
                lo, hi = cluster_boot(y_pc, p, wsi, n=500)
                rows.append(dict(encoder=key, source_training_set=sname, field_of_view=fname,
                                 internal_auroc=auc(y_dev[te], clf.predict_proba(sc.transform(f[te]))[:, 1]),
                                 external_auroc=auc(y_pc, p), ci_lo=lo, ci_hi=hi))
                print(f"{key:<16}{sname:<17}{fname:<11} internal {rows[-1]['internal_auroc']:.3f} external "
                      f"{rows[-1]['external_auroc']:.3f} ({lo:.3f}-{hi:.3f})", flush=True)
    pd.DataFrame(rows).round(4).to_csv(REV / "R2_R5_field_of_view_ci_and_ds3_only.csv", index=False)


# ============================================================================== minor points
def minors():
    # -- minor 3: logistic-regression few-shot for every configuration
    rows = []
    for key in ENC:
        d = D(key)
        for k in (5, 10, 50, 100):
            a = [auc(d.y_test, LogisticRegression(C=0.01, max_iter=500).fit(d.za[s], d.y_adapt[s]).decision_function(d.zt))
                 for s in (draw_shots(d.y_adapt, k, r) for r in range(20))]
            b = [auc(d.y_test, proto_score(d.za[s], d.y_adapt[s], d.zt)) for s in (draw_shots(d.y_adapt, k, r) for r in range(20))]
            rows.append(dict(encoder=key, k=k, logreg_mean=np.mean(a), logreg_sd=np.std(a, ddof=1),
                             centroid_mean=np.mean(b), centroid_sd=np.std(b, ddof=1)))
    lr = pd.DataFrame(rows).round(3)
    lr.to_csv(REV / "R2_minor3_logreg_fewshot_all_encoders.csv", index=False)
    print(lr.pivot_table(index="encoder", columns="k", values="logreg_mean").loc[ENC].to_string())

    # -- minor 4: internal results without near-duplicates, and two-fold leave-collection-out
    index = pd.read_csv(C.DEV_INDEX_CSV)
    y = index.label.to_numpy()
    pairs = pd.read_csv(REV / "M6_near_duplicate_pairs.csv")
    split = index.native_split.to_numpy()
    tr, va, te = (index[index.native_split == s].idx.to_numpy() for s in ("train", "val", "test"))
    dirty = set()
    for i, j in zip(pairs.i, pairs.j):
        if split[i] == "test" and split[j] != "test":
            dirty.add(i)
        if split[j] == "test" and split[i] != "test":
            dirty.add(j)
    te_clean = np.array([i for i in te if i not in dirty])
    ds = index.dataset.to_numpy()
    a_idx = np.flatnonzero(np.isin(ds, ["ds1", "ds2"]))
    b_idx = np.flatnonzero(ds == "ds3")
    rows = []
    for key in ENC:
        f = np.load(C.FEATURE_DIR / key / "dev.npy")
        sc, clf, _, _ = fit_probe(f[tr], y[tr], f[va], y[va])
        p = clf.predict_proba(sc.transform(f))[:, 1]
        row = dict(encoder=key, n_test=len(te), n_test_clean=len(te_clean),
                   internal_auroc=auc(y[te], p[te]), internal_auroc_clean=auc(y[te_clean], p[te_clean]),
                   internal_acc=float(((p[te] >= 0.5) == y[te]).mean()),
                   internal_acc_clean=float(((p[te_clean] >= 0.5) == y[te_clean]).mean()))
        for name, src, tgt in (("train_DS1+DS2_test_DS3", a_idx, b_idx), ("train_DS3_test_DS1+DS2", b_idx, a_idx)):
            s_tr = src[split[src] != "val"]
            s_va = src[split[src] == "val"]
            sc2, clf2, _, _ = fit_probe(f[s_tr], y[s_tr], f[s_va], y[s_va])
            row[name] = auc(y[tgt], clf2.predict_proba(sc2.transform(f[tgt]))[:, 1])
        rows.append(row)
    r = pd.DataFrame(rows).round(3)
    r.to_csv(REV / "R2_minor4_clean_internal_and_two_fold.csv", index=False)
    print(r.to_string(index=False))

    # -- minor 11: tumour-bearing slides in the evaluation pool
    t = pd.read_csv(C.CACHE_DIR / "pcam_test_index.csv")
    g = t.groupby("wsi").label.agg(n="size", pos="sum")
    named_tumour = g[g.index.str.contains("tumor")]
    print("slides:", len(g), "| named tumour:", len(named_tumour), "| with >=1 positive patch:", int((g.pos >= 1).sum()),
          "| with >=20 positive:", int((g.pos >= 20).sum()), "| named tumour with <20 positive:",
          int((named_tumour.pos < 20).sum()), "| min positives among tumour slides:", int(named_tumour.pos.min()))
    pd.Series(dict(slides=len(g), named_tumour=len(named_tumour), with_positive=int((g.pos >= 1).sum()),
                   with_20_positive=int((g.pos >= 20).sum()),
                   min_positive_on_tumour_slide=int(named_tumour.pos.min()))).to_csv(REV / "R2_minor11_tumour_slides.csv")

    # -- minor 8: one-class rule without statistics from the (tumour-containing) unlabelled pool
    d = D("phikon")
    neg = np.flatnonzero(d.y_adapt == 0)
    rows = []
    for k in (50, 1000):
        for rep in range(20 if k == 50 else 5):
            rng = np.random.default_rng(C.SEED + 7 * k + rep)
            sel = rng.choice(neg, k, replace=False)
            sc_n = StandardScaler().fit(d.f_adapt[neg])                 # statistics of normal patches only
            for name, za, zt in (("pool statistics (as reported)", d.za, d.zt),
                                 ("no standardisation", d.f_adapt, d.f_test),
                                 ("statistics of normal patches only", sc_n.transform(d.f_adapt), sc_n.transform(d.f_test))):
                c0 = unit(za[sel].mean(0))
                rows.append(dict(k=k, standardisation=name, auroc=auc(d.y_test, -(unit(zt) @ c0))))
    o = pd.DataFrame(rows).groupby(["k", "standardisation"]).auroc.agg(["mean", "std"]).round(3).reset_index()
    o.to_csv(REV / "R2_minor8_one_class_standardisation.csv", index=False)
    print(o.to_string(index=False))


if __name__ == "__main__":
    minors()
    r1_mechanism()
    r3_tissues()
    r5_fov()
