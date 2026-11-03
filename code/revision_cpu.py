"""Revision analyses on cached embeddings (review comments M1, M6-M9 and minor points).

    python revision_cpu.py core     everything that needs only the original embeddings
    python revision_cpu.py post     analyses of the new GPU outputs (scale control, extra tissues, PLIP)

Tables are written to result_prediction/revision/.
"""
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import config as C
from linear_probe import fit_probe
from metrics import all_metrics, expected_calibration_error

warnings.filterwarnings("ignore")
REV = C.RESULT_DIR / "revision"
REV.mkdir(parents=True, exist_ok=True)
ENC = C.DEFAULT_BACKBONES
TAU = 10.0


def unit(a):
    return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)


def auc(y, s):
    """Rank-based AUROC (ties averaged); fast enough for repeated bootstrap use."""
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    ss = s[order]
    i = 0
    r = np.arange(1, len(s) + 1, dtype=float)
    boundaries = np.flatnonzero(np.diff(ss) != 0) + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [len(s)]])
    avg = (r[starts] + r[ends - 1]) / 2
    ranks[order] = np.repeat(avg, ends - starts)
    n1 = y.sum()
    n0 = len(y) - n1
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def draw_shots(y_adapt, k, rep):
    rng = np.random.default_rng(C.SEED + 1000 * k + rep)
    pos, neg = np.flatnonzero(y_adapt == 1), np.flatnonzero(y_adapt == 0)
    return np.concatenate([rng.choice(neg, k, replace=False), rng.choice(pos, k, replace=False)])


def proto_score(za, ys, zt):
    c0, c1 = unit(za[ys == 0].mean(0)), unit(za[ys == 1].mean(0))
    zt = unit(zt)
    return zt @ c1 - zt @ c0


class D:
    def __init__(self, key):
        self.key = key
        self.index = pd.read_csv(C.DEV_INDEX_CSV)
        self.f_dev = np.load(C.FEATURE_DIR / key / "dev.npy")
        self.f_adapt = np.load(C.FEATURE_DIR / key / "pcam_adapt.npy")
        self.f_test = np.load(C.FEATURE_DIR / key / "pcam_test.npy")
        self.adapt = pd.read_csv(C.CACHE_DIR / "pcam_adapt_index.csv")
        self.test = pd.read_csv(C.CACHE_DIR / "pcam_test_index.csv")
        self.y_dev = self.index.label.to_numpy()
        self.y_adapt, self.y_test = self.adapt.label.to_numpy(), self.test.label.to_numpy()
        self.tr = self.index[self.index.native_split == "train"].idx.to_numpy()
        self.va = self.index[self.index.native_split == "val"].idx.to_numpy()
        self.te = self.index[self.index.native_split == "test"].idx.to_numpy()
        self.tsc = StandardScaler().fit(self.f_adapt)
        self.za, self.zt = self.tsc.transform(self.f_adapt), self.tsc.transform(self.f_test)

    def source_probe(self, train_idx=None):
        tr = self.tr if train_idx is None else train_idx
        sc, clf, c, _ = fit_probe(self.f_dev[tr], self.y_dev[tr], self.f_dev[self.va], self.y_dev[self.va])
        return sc, clf, c


# ============================================================================== M1 + minor 2
def task_shift_tests():
    rows_dir, rows_norm, rows_fp = [], [], []
    for key in ENC:
        d = D(key)
        sc, clf, c = d.source_probe()
        p_direct = clf.predict_proba(sc.transform(d.f_test))[:, 1]
        a_direct = auc(d.y_test, p_direct)

        # -- direction of the decision vectors. Both classifiers are refitted in one common
        #    standardised space (pooled source + target statistics) so that their weight vectors
        #    are comparable; dividing weights by per-domain SDs lets near-constant dimensions
        #    dominate the cosine.
        common = StandardScaler().fit(np.vstack([d.f_dev[d.tr], d.f_adapt]))
        w_s = LogisticRegression(C=c, max_iter=500, class_weight="balanced").fit(
            common.transform(d.f_dev[d.tr]), d.y_dev[d.tr]).coef_[0]
        w_t = LogisticRegression(C=c, max_iter=500).fit(common.transform(d.f_adapt), d.y_adapt).coef_[0]
        cos = float(w_s @ w_t / (np.linalg.norm(w_s) * np.linalg.norm(w_t)))
        # class-mean difference directions (no regularisation involved)
        xs_c, xa_c = common.transform(d.f_dev[d.tr]), common.transform(d.f_adapt)
        m_s = xs_c[d.y_dev[d.tr] == 1].mean(0) - xs_c[d.y_dev[d.tr] == 0].mean(0)
        m_t = xa_c[d.y_adapt == 1].mean(0) - xa_c[d.y_adapt == 0].mean(0)
        cos_means = float(m_s @ m_t / (np.linalg.norm(m_s) * np.linalg.norm(m_t)))
        tgt = LogisticRegression(C=c, max_iter=500).fit(d.za, d.y_adapt)
        # -- reverse direction: classifier trained on lymph node, applied to oral tissue
        p_rev = tgt.predict_proba(d.tsc.transform(d.f_dev[d.te]))[:, 1]
        rows_dir.append(dict(encoder=key, cos_wS_wT=cos, cos_class_mean_directions=cos_means,
                             auroc_oral_to_node=a_direct,
                             auroc_flipped=1 - a_direct,
                             auroc_node_to_oral=auc(d.y_dev[d.te], p_rev),
                             auroc_node_to_node=auc(d.y_test, tgt.predict_proba(d.zt)[:, 1]),
                             auroc_oral_to_oral=auc(d.y_dev[d.te], clf.predict_proba(sc.transform(d.f_dev[d.te]))[:, 1])))

        # -- normal-only target labels
        neg = np.flatnonzero(d.y_adapt == 0)
        xs, ys = sc.transform(d.f_dev[d.tr]), d.y_dev[d.tr]
        xt_src, xa_src = sc.transform(d.f_test), sc.transform(d.f_adapt)
        for k in (10, 50, 100, 1000):
            for rep in range(20 if k <= 100 else 5):
                rng = np.random.default_rng(C.SEED + 7 * k + rep)
                sel = rng.choice(neg, k, replace=False)
                # (a) one-class: distance from the mean of k normal lymph node patches
                c0 = unit(d.za[sel].mean(0))
                rows_norm.append(dict(encoder=key, method="one-class (distance from normal prototype)", k=k,
                                      rep=rep, auroc=auc(d.y_test, -(unit(d.zt) @ c0))))
                if rep < 3:
                    n_src_neg = int((ys == 0).sum())
                    # (b) source classifier refitted with the target normals as extra negatives
                    x = np.vstack([xs, xa_src[sel]])
                    y = np.concatenate([ys, np.zeros(k, int)])
                    w = np.concatenate([np.ones(len(ys)), np.full(k, n_src_neg / k)])
                    m = LogisticRegression(C=c, max_iter=300).fit(x, y, sample_weight=w)
                    rows_norm.append(dict(encoder=key, method="source + target normals as extra negatives", k=k,
                                          rep=rep, auroc=auc(d.y_test, m.predict_proba(xt_src)[:, 1])))
                    # (c) oral carcinoma against target normals only (oral normals removed)
                    x = np.vstack([xs[ys == 1], xa_src[sel]])
                    y = np.concatenate([np.ones(int((ys == 1).sum()), int), np.zeros(k, int)])
                    m = LogisticRegression(C=c, max_iter=300, class_weight="balanced").fit(x, y)
                    rows_norm.append(dict(encoder=key, method="oral carcinoma vs target normals", k=k, rep=rep,
                                          auroc=auc(d.y_test, m.predict_proba(xt_src)[:, 1])))
        rows_norm.append(dict(encoder=key, method="direct transfer", k=0, rep=0, auroc=a_direct))

        # -- where the false positives of direct transfer fall
        slide_type = np.where(d.test.wsi.str.contains("normal"), "normal slide", "tumour slide")
        x = np.load(C.CACHE_DIR / "pcam_test_x.npy", mmap_mode="r")
        lum = (0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2])
        dark = (lum < 110).reshape(len(x), -1).mean(1)          # proxy for nuclear density
        normal = d.y_test == 0
        tert = pd.qcut(dark[normal], 3, labels=["low", "middle", "high"]).astype(str)
        call = (p_direct >= 0.5)[normal]
        row = dict(encoder=key, fp_rate_all_normals=call.mean(),
                   fp_rate_normal_slides=call[slide_type[normal] == "normal slide"].mean(),
                   fp_rate_tumour_slide_normals=call[slide_type[normal] == "tumour slide"].mean(),
                   n_normal_slide=int((slide_type[normal] == "normal slide").sum()),
                   n_tumour_slide_normals=int((slide_type[normal] == "tumour slide").sum()))
        for t in ("low", "middle", "high"):
            row[f"fp_rate_nuclear_density_{t}"] = call[tert == t].mean()
        row["spearman_score_vs_density_normals"] = float(pd.Series(p_direct[normal]).corr(pd.Series(dark[normal]), method="spearman"))
        rows_fp.append(row)
        print(f"[{key}] cos(wS,wT)={cos:+.3f} cos(means)={cos_means:+.3f} direct={a_direct:.3f} reverse={rows_dir[-1]['auroc_node_to_oral']:.3f} "
              f"FP normal-slide={row['fp_rate_normal_slides']:.3f} tumour-slide={row['fp_rate_tumour_slide_normals']:.3f} "
              f"density low/mid/high={row['fp_rate_nuclear_density_low']:.2f}/{row['fp_rate_nuclear_density_middle']:.2f}/"
              f"{row['fp_rate_nuclear_density_high']:.2f}", flush=True)

    pd.DataFrame(rows_dir).round(4).to_csv(REV / "M1_direction_and_reverse.csv", index=False)
    pd.DataFrame(rows_fp).round(4).to_csv(REV / "M1_false_positive_breakdown.csv", index=False)
    n = pd.DataFrame(rows_norm)
    n.to_csv(REV / "M1_normal_only_labels_all.csv", index=False)
    s = n.groupby(["encoder", "method", "k"]).auroc.agg(["mean", "std"]).round(3).reset_index()
    s.to_csv(REV / "M1_normal_only_labels_summary.csv", index=False)
    print(s.pivot_table(index=["method", "k"], columns="encoder", values="mean").round(3).to_string())


# ============================================================================== M7: paired tests
def paired_tests(n_draws=10, n_boot=500):
    data = {k: D(k) for k in ENC}
    ref = data["phikon"]
    y, wsi = ref.y_test, ref.test.wsi.to_numpy()
    slides = np.unique(wsi)
    members = [np.flatnonzero(wsi == s) for s in slides]
    rng = np.random.default_rng(C.SEED)
    boots = [np.concatenate([members[i] for i in rng.integers(0, len(members), len(members))])
             for _ in range(n_boot)]
    rows = []
    for k in (10, 50):
        scores = {}
        for key, d in data.items():
            scores[key] = [proto_score(d.za[sel], d.y_adapt[sel], d.zt)
                           for sel in (draw_shots(d.y_adapt, k, r) for r in range(n_draws))]
        d = ref
        scores["phikon_logreg"] = [LogisticRegression(C=0.01, max_iter=500).fit(d.za[sel], d.y_adapt[sel])
                                   .decision_function(d.zt)
                                   for sel in (draw_shots(d.y_adapt, k, r) for r in range(n_draws))]
        pairs = [("phikon", e) for e in ENC if e != "phikon"] + [("phikon_logreg", "phikon")]
        for a, b in pairs:
            point = np.mean([auc(y, sa) - auc(y, sb) for sa, sb in zip(scores[a], scores[b])])
            diffs = np.array([np.mean([auc(y[i], sa[i]) - auc(y[i], sb[i])
                                       for sa, sb in zip(scores[a], scores[b])]) for i in boots])
            p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
            rows.append(dict(k=k, model_a=a, model_b=b, auroc_difference=point,
                             ci_lo=np.quantile(diffs, 0.025), ci_hi=np.quantile(diffs, 0.975),
                             p_value=max(p, 1 / n_boot)))
            print(f"k={k:<3} {a:<14} - {b:<22} dAUROC {point:+.3f} ({rows[-1]['ci_lo']:+.3f}, {rows[-1]['ci_hi']:+.3f}) "
                  f"p={rows[-1]['p_value']:.3f}", flush=True)
    pd.DataFrame(rows).round(4).to_csv(REV / "M7_paired_slide_bootstrap_differences.csv", index=False)


# ============================================================================== M8, M9, minors
def robustness():
    d = D("phikon")
    y = d.y_test
    out = []

    def both(sel, ys):
        res = {"prototype": auc(y, proto_score(d.za[sel], ys, d.zt))}
        if len(np.unique(ys)) == 2:
            res["logistic regression"] = auc(y, LogisticRegression(C=0.01, max_iter=500).fit(d.za[sel], ys)
                                             .decision_function(d.zt))
        return res

    # -- supports restricted to a few slides
    g = d.adapt.groupby("wsi").label.agg(n="size", pos="sum")
    ok = g[(g.pos >= 10) & (g.n - g.pos >= 10)].index.to_numpy()
    wsi_a = d.adapt.wsi.to_numpy()
    for n_slides in (1, 2, 5):
        for k in (10, 50):
            for rep in range(20):
                rng = np.random.default_rng(C.SEED + 31 * n_slides + 7 * k + rep)
                chosen = rng.choice(ok, n_slides, replace=False)
                pool = np.flatnonzero(np.isin(wsi_a, chosen))
                sel = np.concatenate([rng.choice(pool[d.y_adapt[pool] == c], min(k, int((d.y_adapt[pool] == c).sum())),
                                                 replace=False) for c in (0, 1)])
                for head, a in both(sel, d.y_adapt[sel]).items():
                    out.append(dict(experiment="support slides", setting=f"{n_slides} slide(s)", k=k, head=head, auroc=a))
    for k in (10, 50):
        for rep in range(20):
            sel = draw_shots(d.y_adapt, k, rep)
            for head, a in both(sel, d.y_adapt[sel]).items():
                out.append(dict(experiment="support slides", setting="all 108 slides", k=k, head=head, auroc=a))
    print(f"slides usable for restricted supports: {len(ok)} of {g.shape[0]}")

    # -- label noise in the support set
    for rho in (0.0, 0.1, 0.2, 0.3, 0.4):
        for rep in range(20):
            sel = draw_shots(d.y_adapt, 50, rep)
            ys = d.y_adapt[sel].copy()
            rng = np.random.default_rng(C.SEED + rep)
            flip = rng.choice(len(ys), int(round(rho * len(ys))), replace=False)
            ys[flip] = 1 - ys[flip]
            for head, a in both(sel, ys).items():
                out.append(dict(experiment="label noise", setting=f"{int(rho * 100)}% flipped", k=50, head=head, auroc=a))

    # -- large supports: where does the prototype head level off?
    for k in (100, 200, 500, 1000, 2000, 5000):
        for rep in range(10):
            sel = draw_shots(d.y_adapt, k, rep)
            for head, a in both(sel, d.y_adapt[sel]).items():
                out.append(dict(experiment="large support", setting=f"k={k}", k=k, head=head, auroc=a))
    full = np.arange(len(d.y_adapt))
    for head, a in both(full, d.y_adapt).items():
        out.append(dict(experiment="large support", setting="all 20,000 adaptation patches", k=10000, head=head, auroc=a))

    r = pd.DataFrame(out)
    r.to_csv(REV / "M9_robustness_all.csv", index=False)
    s = r.groupby(["experiment", "setting", "k", "head"], sort=False).auroc.agg(["mean", "std"]).round(3).reset_index()
    s.to_csv(REV / "M9_robustness_summary.csv", index=False)
    print(s.to_string(index=False))

    # -- temperature fitted on the support set by leave-one-out (minor 6)
    rows = []
    grid = np.logspace(0, 2.5, 80)
    for k in (10, 50):
        for rep in range(20):
            sel = draw_shots(d.y_adapt, k, rep)
            zs, ys = d.za[sel], d.y_adapt[sel]
            loo = np.empty(len(sel))
            for i in range(len(sel)):
                m = np.ones(len(sel), bool)
                m[i] = False
                loo[i] = proto_score(zs[m], ys[m], zs[i:i + 1])[0]
            nll = [np.mean(np.log1p(np.exp(-(2 * ys - 1) * t * loo))) for t in grid]
            tau = float(grid[int(np.argmin(nll))])
            sc = proto_score(zs, ys, d.zt)
            for name, t in (("fixed tau = 10", TAU), ("tau fitted on support (leave-one-out)", tau)):
                p = 1 / (1 + np.exp(-t * sc))
                rows.append(dict(k=k, rep=rep, temperature=name, tau=t, ece=expected_calibration_error(y, p),
                                 brier=float(np.mean((p - y) ** 2))))
    t = pd.DataFrame(rows)
    ts = t.groupby(["k", "temperature"])[["tau", "ece", "brier"]].agg(["mean", "std"]).round(3)
    ts.to_csv(REV / "minor6_temperature_calibration.csv")
    print(ts.to_string())

    # -- gap closed against chance (M8) and source-plus-shots values (minor 10)
    ad = pd.read_csv(C.TABLE_DIR / "adaptation_metrics.csv")
    rows = []
    for key in ("phikon", "phikon_v2"):
        a = ad[ad.backbone == key]
        direct = a[a.method == "direct"].auroc.iloc[0]
        sup = a[a.method == "oracle_target_supervised"].auroc.iloc[0]
        for k in (10, 50):
            ak = a[(a.method == "fewshot_prototype") & (a.k == k)].auroc.mean()
            rows.append(dict(encoder=key, k=k, auroc_k=ak, direct=direct, upper=sup,
                             G_vs_direct=(ak - direct) / (sup - direct), G_vs_chance=(ak - 0.5) / (sup - 0.5)))
    pd.DataFrame(rows).round(3).to_csv(REV / "M8_gap_closed.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    sp = ad[ad.method.isin(["fewshot_source_plus_shots", "fewshot_prototype"]) & ad.backbone.isin(["phikon", "phikon_v2"])]
    sp.groupby(["backbone", "method", "k"]).auroc.agg(["mean", "std"]).round(3).reset_index().to_csv(
        REV / "minor10_source_plus_shots.csv", index=False)

    # -- per-slide sensitivity on tumour slides, specificity on normal slides (minor 11)
    pr = pd.read_csv(C.ROOT / "prposed model resul and prediction" / "predictions" / "predictions_phikon_k50.csv")
    pos = pr[pr.label == 1].groupby("slide").agg(n_pos=("label", "size"), sensitivity=("correct", "mean"))
    pos = pos[pos.n_pos >= 20]
    neg = pr[pr.label == 0].groupby("slide").agg(n_neg=("label", "size"), specificity=("correct", "mean"))
    neg = neg[neg.n_neg >= 20]
    neg_n = neg[neg.index.str.contains("normal")]
    neg_t = neg[neg.index.str.contains("tumor")]
    summ = dict(tumour_slides=len(pos), sens_median=pos.sensitivity.median(), sens_q25=pos.sensitivity.quantile(.25),
                sens_q75=pos.sensitivity.quantile(.75), tumour_slides_sens_below_0_7=int((pos.sensitivity < 0.7).sum()),
                tumour_slides_sens_below_0_5=int((pos.sensitivity < 0.5).sum()),
                normal_slides=len(neg_n), spec_normal_slides_median=neg_n.specificity.median(),
                spec_tumour_slide_normals_median=neg_t.specificity.median())
    pos.round(4).to_csv(REV / "minor11_per_tumour_slide_sensitivity.csv")
    pd.Series(summ).round(3).to_csv(REV / "minor11_summary.csv")
    print(pd.Series(summ).round(3).to_string())


# ============================================================================== M6: duplicates
def duplicate_audit():
    index = pd.read_csv(C.DEV_INDEX_CSV)
    x = np.load(C.DEV_CACHE_NPY, mmap_mode="r")
    hashes = np.empty(len(x), np.uint64)
    weights = (1 << np.arange(64, dtype=np.uint64))
    for s in range(0, len(x), 512):
        b = np.asarray(x[s:s + 512]).astype(np.float32).mean(-1)[:, :, :252]         # grey, 256 x 252
        small = b.reshape(len(b), 8, 32, 9, 28).mean((2, 4))                          # 8 x 9 block means
        bits = (small[:, :, 1:] > small[:, :, :-1]).reshape(len(b), 64)               # difference hash
        hashes[s:s + len(b)] = (bits.astype(np.uint64) * weights).sum(1, dtype=np.uint64)
    f = unit(np.load(C.FEATURE_DIR / "phikon" / "dev.npy"))
    ds, sp = index.dataset.to_numpy(), index.native_split.to_numpy()
    pairs = []
    for s in range(0, len(x), 256):
        ham = np.bitwise_count(hashes[s:s + 256, None] ^ hashes[None, :])
        cos = f[s:s + 256] @ f.T
        ii, jj = np.nonzero((ham <= 6) & (cos >= 0.90))
        for i, j in zip(ii + s, jj):
            if i < j:
                pairs.append((i, j, int(ham[i - s, j]), float(cos[i - s, j])))
    p = pd.DataFrame(pairs, columns=["i", "j", "hamming", "cosine"])
    p["ds_i"], p["ds_j"], p["split_i"], p["split_j"] = ds[p.i], ds[p.j], sp[p.i], sp[p.j]
    p.to_csv(REV / "M6_near_duplicate_pairs.csv", index=False)
    cross = p[p.ds_i != p.ds_j]
    within = p[p.ds_i == p.ds_j]
    leak = within[((within.split_i == "train") & (within.split_j != "train")) |
                  ((within.split_j == "train") & (within.split_i != "train"))]
    rows = [dict(scope="pairs across different datasets", pairs=len(cross),
                 images_involved=len(set(cross.i) | set(cross.j)))]
    for dname in ("ds1", "ds2", "ds3"):
        w, l = within[within.ds_i == dname], leak[leak.ds_i == dname]
        held = set(l.i[l.split_i != "train"]) | set(l.j[l.split_j != "train"])
        n_held = int(((ds == dname) & (sp != "train")).sum())
        rows.append(dict(scope=f"{dname}: pairs within the dataset", pairs=len(w),
                         images_involved=len(set(w.i) | set(w.j))))
        rows.append(dict(scope=f"{dname}: training image paired with a validation/test image", pairs=len(l),
                         images_involved=len(held), of_total=n_held))
    r = pd.DataFrame(rows)
    r.to_csv(REV / "M6_near_duplicate_summary.csv", index=False)
    print(r.to_string(index=False))
    print(cross.groupby(["ds_i", "ds_j"]).size().to_string() if len(cross) else "no cross-dataset near-duplicates")


def overlap_embedding():
    """Augmentation-tolerant overlap check: nearest-neighbour cosine in Phikon embedding space.

    A difference hash misses rotated or flipped copies, so this complements duplicate_audit():
    for every image of dataset B, the highest cosine similarity to any image of dataset A.
    """
    index = pd.read_csv(C.DEV_INDEX_CSV)
    f = unit(np.load(C.FEATURE_DIR / "phikon" / "dev.npy"))
    ds, sp = index.dataset.to_numpy(), index.native_split.to_numpy()

    def nn_max(a_idx, b_idx):
        out = np.empty(len(b_idx), np.float32)
        for s in range(0, len(b_idx), 1024):
            out[s:s + 1024] = (f[b_idx[s:s + 1024]] @ f[a_idx].T).max(1)
        return out

    rows = []
    for a in ("ds1", "ds2", "ds3"):
        for b in ("ds1", "ds2", "ds3"):
            if a == b:
                ai = np.flatnonzero((ds == a) & (sp == "train"))
                bi = np.flatnonzero((ds == a) & (sp != "train"))
                name = f"{a}: validation/test -> training"
            else:
                ai, bi = np.flatnonzero(ds == a), np.flatnonzero(ds == b)
                name = f"{b} -> {a}"
            m = nn_max(ai, bi)
            rows.append(dict(query_to_reference=name, n_query=len(bi), median_nn_cosine=np.median(m),
                             frac_ge_0_90=(m >= 0.90).mean(), frac_ge_0_95=(m >= 0.95).mean(),
                             frac_ge_0_98=(m >= 0.98).mean()))
    r = pd.DataFrame(rows).round(3)
    r.to_csv(REV / "M6_embedding_nearest_neighbour_overlap.csv", index=False)
    print(r.to_string(index=False))


# ============================================================================== post-GPU analyses
def post():
    index = pd.read_csv(C.DEV_INDEX_CSV)
    y_dev = index.label.to_numpy()
    tr = index[index.native_split == "train"].idx.to_numpy()
    va = index[index.native_split == "val"].idx.to_numpy()
    te = index[index.native_split == "test"].idx.to_numpy()
    y_pc = pd.read_csv(C.CACHE_DIR / "pcam_test_index.csv").label.to_numpy()

    # -- field-of-view (physical scale) control
    rows = []
    for key in ("efficientnet_b0", "phikon", "phikon_v2"):
        if not (C.FEATURE_DIR / key / "dev_fov12.npy").exists():
            continue
        f_pc = np.load(C.FEATURE_DIR / key / "pcam_test.npy")
        feats = {"full field": np.load(C.FEATURE_DIR / key / "dev.npy")}
        for k, name in (("fov50", "1/2 field"), ("fov25", "1/4 field"), ("fov12", "1/8 field")):
            feats[name] = np.load(C.FEATURE_DIR / key / f"dev_{k}.npy")
        for name, f in feats.items():
            sc, clf, _, _ = fit_probe(f[tr], y_dev[tr], f[va], y_dev[va])
            p = clf.predict_proba(sc.transform(f_pc))[:, 1]
            m = all_metrics(y_pc, p)
            rows.append(dict(encoder=key, source_field_of_view=name,
                             internal_auroc=auc(y_dev[te], clf.predict_proba(sc.transform(f[te]))[:, 1]),
                             external_auroc=m["auroc"], external_sensitivity=m["sensitivity"],
                             external_specificity=m["specificity"], positive_rate=m["positive_rate"]))
        allf = np.vstack([f[tr] for f in feats.values()])
        ally = np.concatenate([y_dev[tr]] * len(feats))
        sc, clf, _, _ = fit_probe(allf, ally, np.vstack([f[va] for f in feats.values()]),
                                  np.concatenate([y_dev[va]] * len(feats)))
        m = all_metrics(y_pc, clf.predict_proba(sc.transform(f_pc))[:, 1])
        rows.append(dict(encoder=key, source_field_of_view="all four scales pooled",
                         internal_auroc=auc(y_dev[te], clf.predict_proba(sc.transform(feats["full field"][te]))[:, 1]),
                         external_auroc=m["auroc"], external_sensitivity=m["sensitivity"],
                         external_specificity=m["specificity"], positive_rate=m["positive_rate"]))
    if rows:
        r = pd.DataFrame(rows).round(3)
        r.to_csv(REV / "M2_field_of_view_control.csv", index=False)
        print(r.to_string(index=False))

    # -- further target tissues
    rows = []
    no_ds2 = index[(index.native_split == "train") & (index.dataset != "ds2")].idx.to_numpy()
    for tissue in ("lung", "colon"):
        if not (C.CACHE_DIR / f"{tissue}_eval_index.csv").exists():
            continue
        ye = pd.read_csv(C.CACHE_DIR / f"{tissue}_eval_index.csv").label.to_numpy()
        ya = pd.read_csv(C.CACHE_DIR / f"{tissue}_adapt_index.csv").label.to_numpy()
        for key in ENC:
            if not (C.FEATURE_DIR / key / f"{tissue}_eval.npy").exists():
                continue
            f_dev = np.load(C.FEATURE_DIR / key / "dev.npy")
            fe, fa = np.load(C.FEATURE_DIR / key / f"{tissue}_eval.npy"), np.load(C.FEATURE_DIR / key / f"{tissue}_adapt.npy")
            row = dict(tissue=tissue, encoder=key)
            for name, idx in (("direct", tr), ("direct_without_ds2", no_ds2)):
                sc, clf, _, _ = fit_probe(f_dev[idx], y_dev[idx], f_dev[va], y_dev[va])
                p = clf.predict_proba(sc.transform(fe))[:, 1]
                m = all_metrics(ye, p)
                row[f"{name}_auroc"], row[f"{name}_sens"], row[f"{name}_spec"] = m["auroc"], m["sensitivity"], m["specificity"]
            tsc = StandardScaler().fit(fa)
            za, ze = tsc.transform(fa), tsc.transform(fe)
            for k in (1, 5, 10, 50):
                row[f"prototype_k{k}"] = np.mean([auc(ye, proto_score(za[s], ya[s], ze))
                                                  for s in (draw_shots(ya, k, r) for r in range(20))])
            row["upper_bound"] = auc(ye, LogisticRegression(C=0.01, max_iter=500).fit(za, ya).decision_function(ze))
            rows.append(row)
    if rows:
        r = pd.DataFrame(rows).round(3)
        r.to_csv(REV / "M4_further_tissues.csv", index=False)
        print(r.to_string(index=False))

    # -- PLIP as an encoder for few-shot prototypes (context for the zero-shot result)
    if (REV / "plip_pcam_adapt_embed.npy").exists():
        fa, ft = np.load(REV / "plip_pcam_adapt_embed.npy"), np.load(REV / "plip_pcam_test_embed.npy")
        ya = pd.read_csv(C.CACHE_DIR / "pcam_adapt_index.csv").label.to_numpy()
        tsc = StandardScaler().fit(fa)
        za, zt = tsc.transform(fa), tsc.transform(ft)
        res = {f"prototype_k{k}": np.mean([auc(y_pc, proto_score(za[s], ya[s], zt))
                                           for s in (draw_shots(ya, k, r) for r in range(20))]) for k in (10, 50)}
        res["upper_bound"] = auc(y_pc, LogisticRegression(C=0.01, max_iter=500).fit(za, ya).decision_function(zt))
        pd.Series(res).round(3).to_csv(REV / "plip_few_shot.csv")
        print("PLIP image encoder, few-shot:", {k: round(v, 3) for k, v in res.items()})


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "core":
        duplicate_audit()
        task_shift_tests()
        robustness()
        paired_tests()
    elif mode == "post":
        post()
    else:
        {"dup": duplicate_audit, "shift": task_shift_tests, "robust": robustness, "paired": paired_tests,
         "overlap": overlap_embedding}[mode]()
