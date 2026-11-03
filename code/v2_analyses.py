"""Analyses for the revision of the first manuscript version (MDPI review, comments 3-8, 10, 17).

All on cached embeddings; results go to paper/version2/analysis/.
    python v2_analyses.py
"""
import re
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve

import config as C
from revision_cpu import D, auc, draw_shots, proto_score, unit

warnings.filterwarnings("ignore")
OUT = C.ROOT / "paper" / "version2" / "analysis"
OUT.mkdir(parents=True, exist_ok=True)
ENC = C.DEFAULT_BACKBONES


def lr_score(xa, ya, xt):
    return LogisticRegression(C=0.01, max_iter=500).fit(xa, ya).decision_function(xt)


def slide_boot_index(wsi, n, seed=C.SEED):
    rng = np.random.default_rng(seed)
    u = np.unique(wsi)
    members = [np.flatnonzero(wsi == s) for s in u]
    return [np.concatenate([members[i] for i in rng.integers(0, len(u), len(u))]) for _ in range(n)]


def centre_of(wsi):
    """CAMELYON16 training slides: normal 1-100 and tumour 1-70 from Radboud UMC, the rest from UMC Utrecht."""
    kind, num = re.match(r".*_(normal|tumor)_(\d+)$", wsi).groups()
    num = int(num)
    return "Radboud" if (kind == "normal" and num <= 100) or (kind == "tumor" and num <= 70) else "Utrecht"


# ============================================================================== C3: two centres
def cross_centre():
    d = D("phikon")
    ca = d.adapt.wsi.map(centre_of).to_numpy()
    ct = d.test.wsi.map(centre_of).to_numpy()
    # check the assignment against colour: the two centres used different scanners
    x = np.load(C.CACHE_DIR / "pcam_test_x.npy", mmap_mode="r")
    rgb = np.array([np.asarray(x[i]).reshape(-1, 3).mean(0) for i in range(0, len(x), 10)])
    col = pd.DataFrame(rgb, columns=["R", "G", "B"]).assign(centre=ct[::10], wsi=d.test.wsi.to_numpy()[::10])
    per_slide = col.groupby(["wsi", "centre"])[["R", "G", "B"]].mean().reset_index()
    sep = {c: auc((per_slide.centre == "Utrecht").to_numpy().astype(int), per_slide[c].to_numpy()) for c in "RGB"}
    print("slides per centre (evaluation pool):", pd.Series(ct).groupby(ct).size().to_dict(),
          "| per-slide colour separates the centres, AUROC R/G/B:", {k: round(v, 3) for k, v in sep.items()})
    rows = []
    for key in ("efficientnet_b0", "dinov2_base", "phikon", "phikon_v2"):
        d = D(key)
        for src in ("Radboud", "Utrecht", "both"):
            pool = np.arange(len(d.y_adapt)) if src == "both" else np.flatnonzero(ca == src)
            for tgt in ("Radboud", "Utrecht"):
                ev = np.flatnonzero(ct == tgt)
                for k in (10, 50):
                    res = {"nearest centroid": [], "logistic regression": []}
                    for rep in range(20):
                        rng = np.random.default_rng(C.SEED + 100 * k + rep)
                        sel = np.concatenate([rng.choice(pool[d.y_adapt[pool] == c], k, replace=False) for c in (0, 1)])
                        ys = d.y_adapt[sel]
                        res["nearest centroid"].append(auc(d.y_test[ev], proto_score(d.za[sel], ys, d.zt[ev])))
                        res["logistic regression"].append(auc(d.y_test[ev], lr_score(d.za[sel], ys, d.zt[ev])))
                    for head, v in res.items():
                        rows.append(dict(encoder=key, support_centre=src, evaluation_centre=tgt, k=k, head=head,
                                         auroc_mean=np.mean(v), auroc_sd=np.std(v, ddof=1)))
    r = pd.DataFrame(rows).round(3)
    r.to_csv(OUT / "C3_cross_centre.csv", index=False)
    pd.Series(sep).round(3).to_csv(OUT / "C3_centre_colour_check.csv")
    print(r[(r.encoder == "phikon") & (r.k == 50)].to_string(index=False))


# ============================================================================== C4: support composition
def support_composition():
    d = D("phikon")
    wa = d.adapt.wsi.to_numpy()
    rows, slides = [], []
    for k in (1, 5, 10, 20, 50, 100):
        for rep in range(20):
            sel = draw_shots(d.y_adapt, k, rep)
            slides.append(dict(k=k, rep=rep, unique_slides=len(np.unique(wa[sel])),
                               unique_slides_normal=len(np.unique(wa[sel][d.y_adapt[sel] == 0])),
                               unique_slides_tumour=len(np.unique(wa[sel][d.y_adapt[sel] == 1]))))
    s = pd.DataFrame(slides).groupby("k")[["unique_slides", "unique_slides_normal", "unique_slides_tumour"]].agg(["mean", "min", "max"])
    s.round(1).to_csv(OUT / "C4_unique_slides_random_supports.csv")
    print(s.round(1).to_string())

    # sampling schemes with the same number of labelled patches
    neg_s = {w: np.flatnonzero((wa == w) & (d.y_adapt == 0)) for w in np.unique(wa)}
    pos_s = {w: np.flatnonzero((wa == w) & (d.y_adapt == 1)) for w in np.unique(wa)}
    km_cache = {}
    for k in (5, 10, 20, 50):
        for rep in range(20):
            rng = np.random.default_rng(C.SEED + 13 * k + rep)
            schemes = {"random": draw_shots(d.y_adapt, k, rep)}
            # slide-balanced: one patch per slide in turn, as many different slides as possible
            sel = []
            for cls_sets in (neg_s, pos_s):
                order = [w for w in rng.permutation(list(cls_sets)) if len(cls_sets[w])]
                picked, i = [], 0
                while len(picked) < k:
                    w = order[i % len(order)]
                    cand = np.setdiff1d(cls_sets[w], picked)
                    if len(cand):
                        picked.append(int(rng.choice(cand)))
                    i += 1
                sel += picked
            schemes["slide-balanced"] = np.array(sel)
            # diversity-based selection, label-agnostic: k-means with 2k clusters on the unlabelled pool,
            # the patch nearest to each centre is sent for labelling
            if (k, rep) not in km_cache:
                sub = rng.choice(len(d.za), 5000, replace=False)
                xs = unit(d.za[sub])
                km = KMeans(n_clusters=2 * k, n_init=1, random_state=C.SEED + rep).fit(xs)
                km_cache[(k, rep)] = np.unique(sub[np.argmin(km.transform(xs), axis=0)])
            schemes["diversity (k-means, label-agnostic)"] = km_cache[(k, rep)]
            for name, sel in schemes.items():
                ys = d.y_adapt[sel]
                row = dict(k=k, rep=rep, scheme=name, n_labelled=len(sel), n_tumour=int(ys.sum()),
                           unique_slides=len(np.unique(wa[sel])))
                if 0 < ys.sum() < len(ys):
                    row["nearest centroid"] = auc(d.y_test, proto_score(d.za[sel], ys, d.zt))
                    row["logistic regression"] = auc(d.y_test, lr_score(d.za[sel], ys, d.zt))
                rows.append(row)
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "C4_sampling_schemes_all.csv", index=False)
    g = r.groupby(["k", "scheme"])[["n_labelled", "n_tumour", "unique_slides", "nearest centroid", "logistic regression"]] \
        .agg(["mean", "std"]).round(3)
    g.to_csv(OUT / "C4_sampling_schemes_summary.csv")
    print(g.to_string())


# ============================================================================== C5: embedding distributions
def mmd2(a, b, gamma):
    def k(x, y):
        return np.exp(-gamma * ((x ** 2).sum(1)[:, None] + (y ** 2).sum(1)[None] - 2 * x @ y.T))
    return float(k(a, a).mean() + k(b, b).mean() - 2 * k(a, b).mean())


def distribution_analysis():
    rows = []
    rng = np.random.default_rng(C.SEED)
    for key in ("efficientnet_b0", "dinov2_base", "phikon", "phikon_v2"):
        d = D(key)
        sc = np.vstack([d.f_dev[d.tr], d.f_adapt]).std(0) + 1e-6
        mu = np.vstack([d.f_dev[d.tr], d.f_adapt]).mean(0)
        z = lambda f: (f - mu) / sc
        groups = {"oral normal": z(d.f_dev[d.tr][d.y_dev[d.tr] == 0]), "oral SCC": z(d.f_dev[d.tr][d.y_dev[d.tr] == 1]),
                  "lymph node normal": z(d.f_adapt[d.y_adapt == 0]), "lymph node metastasis": z(d.f_adapt[d.y_adapt == 1])}
        groups = {g: v[rng.choice(len(v), min(1500, len(v)), replace=False)] for g, v in groups.items()}
        allv = np.vstack(list(groups.values()))
        sub = allv[rng.choice(len(allv), 1000, replace=False)]
        med = np.median(((sub[:, None] - sub[None]) ** 2).sum(-1))
        gamma = 1.0 / med
        names = list(groups)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                ca, cb = groups[a].mean(0), groups[b].mean(0)
                rows.append(dict(encoder=key, group_a=a, group_b=b, mmd2=mmd2(groups[a], groups[b], gamma),
                                 centroid_distance=float(np.linalg.norm(ca - cb)),
                                 centroid_cosine=float(unit(ca) @ unit(cb))))
    r = pd.DataFrame(rows).round(4)
    r.to_csv(OUT / "C5_embedding_distances.csv", index=False)
    print(r.pivot_table(index=["group_a", "group_b"], columns="encoder", values="mmd2").round(3).to_string())


# ============================================================================== C6/C7: heads, statistics
def heads_and_statistics(n_draws=10, n_boot=500):
    d = D("phikon")
    y, wsi = d.y_test, d.test.wsi.to_numpy()
    boots = slide_boot_index(wsi, n_boot)
    rows = []
    for k in (5, 10, 20, 50, 100):
        sels = [draw_shots(d.y_adapt, k, r) for r in range(n_draws)]
        pa = [lr_score(d.za[s], d.y_adapt[s], d.zt) for s in sels]
        pb = [proto_score(d.za[s], d.y_adapt[s], d.zt) for s in sels]
        point = np.mean([auc(y, a) - auc(y, b) for a, b in zip(pa, pb)])
        diffs = np.array([np.mean([auc(y[i], a[i]) - auc(y[i], b[i]) for a, b in zip(pa, pb)]) for i in boots])
        p = max(2 * min((diffs <= 0).mean(), (diffs >= 0).mean()), 1 / n_boot)
        # variance components for the nearest-centroid AUROC
        draw_sd = np.std([auc(y, b) for b in pb], ddof=1)
        eval_sd = np.std([auc(y[i], pb[0][i]) for i in boots[:200]], ddof=1)
        rows.append(dict(k=k, auroc_logreg_minus_centroid=point, ci_lo=np.quantile(diffs, .025),
                         ci_hi=np.quantile(diffs, .975), p_value=p, sd_between_support_draws=draw_sd,
                         sd_between_evaluation_resamples=eval_sd))
        print(rows[-1], flush=True)
    r = pd.DataFrame(rows)
    # Holm correction over every paired comparison reported in the manuscript
    enc = pd.read_csv(C.RESULT_DIR / "revision" / "M7_paired_slide_bootstrap_differences.csv")
    allp = [("head", int(k), p) for k, p in zip(r.k, r.p_value)] + \
           [(f"{a}-{b}", int(k), p) for a, b, k, p in zip(enc.model_a, enc.model_b, enc.k, enc.p_value)
            if a != "phikon_logreg"]
    order = np.argsort([p for *_, p in allp])
    m = len(allp)
    adj, running = np.empty(m), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * allp[i][2]))
        adj[i] = running
    holm = pd.DataFrame([dict(comparison=c, k=k, p_value=p, p_holm=a) for (c, k, p), a in zip(allp, adj)])
    holm.round(4).to_csv(OUT / "C6_holm_adjusted.csv", index=False)
    r.round(4).to_csv(OUT / "C6_C7_logreg_vs_centroid_paired.csv", index=False)
    print(holm.round(4).to_string(index=False))

    # multiple centroids per class: does a richer description of each class remove the plateau?
    rows = []
    for k in (50, 100, 500, 2000):
        for m_c in (1, 2, 4, 8):
            if m_c > k:
                continue
            vals = []
            for rep in range(10):
                sel = draw_shots(d.y_adapt, k, rep)
                zs, ys = unit(d.za[sel]), d.y_adapt[sel]
                zt = unit(d.zt)
                cents = {}
                for c in (0, 1):
                    xs = zs[ys == c]
                    cents[c] = unit(KMeans(n_clusters=m_c, n_init=3, random_state=rep).fit(xs).cluster_centers_) if m_c > 1 \
                        else unit(xs.mean(0, keepdims=True))
                s = (zt @ cents[1].T).max(1) - (zt @ cents[0].T).max(1)
                vals.append(auc(y, s))
            rows.append(dict(k=k, centroids_per_class=m_c, auroc_mean=np.mean(vals), auroc_sd=np.std(vals, ddof=1)))
    mc = pd.DataFrame(rows).round(3)
    mc.to_csv(OUT / "C7_multiple_centroids.csv", index=False)
    print(mc.to_string(index=False))

    # computational cost on a CPU, for 2k support patches and 20,000 queries
    rows = []
    for k in (10, 50, 100, 1000):
        sel = draw_shots(d.y_adapt, k, 0)
        t0 = time.perf_counter()
        for _ in range(5):
            proto_score(d.za[sel], d.y_adapt[sel], d.zt)
        t1 = time.perf_counter()
        for _ in range(5):
            lr_score(d.za[sel], d.y_adapt[sel], d.zt)
        t2 = time.perf_counter()
        rows.append(dict(k=k, centroid_ms=(t1 - t0) / 5 * 1000, logreg_ms=(t2 - t1) / 5 * 1000))
    pd.DataFrame(rows).round(1).to_csv(OUT / "C7_compute_time.csv", index=False)
    print(pd.DataFrame(rows).round(1).to_string(index=False))


# ============================================================================== C8: clinical operating points
def clinical():
    d = D("phikon")
    y, wsi = d.y_test, d.test.wsi.to_numpy()
    pr = pd.read_csv(C.ROOT / "prposed model resul and prediction" / "predictions" / "predictions_phikon_k50.csv")
    assert (pr.pcam_index.to_numpy() == d.test.pcam_index.to_numpy()).all()
    scores = {"nearest centroid, k = 50 (median draw)": pr.probability.to_numpy()}
    sel = draw_shots(d.y_adapt, 50, int(pd.read_csv(C.ROOT / "prposed model resul and prediction" / "tables" /
                                                   "Table2_representative_run_95CI.csv").query("Encoder.str.startswith('Phikon (') and `k per class` == 50", engine="python").Draw.iloc[0]))
    scores["logistic regression, same support set"] = lr_score(d.za[sel], d.y_adapt[sel], d.zt)
    boots = slide_boot_index(wsi, 1000)

    def sens_at_spec(yy, ss, spec):
        fpr, tpr, _ = roc_curve(yy, ss)
        return float(np.interp(1 - spec, fpr, tpr))

    rows = []
    for name, s in scores.items():
        for spec in (0.90, 0.95, 0.98):
            v = [sens_at_spec(y[i], s[i], spec) for i in boots]
            rows.append(dict(classifier=name, specificity=spec, sensitivity=sens_at_spec(y, s, spec),
                             ci_lo=np.quantile(v, .025), ci_hi=np.quantile(v, .975)))
    op = pd.DataFrame(rows).round(3)
    op.to_csv(OUT / "C8_sensitivity_at_specificity.csv", index=False)
    print(op.to_string(index=False))

    # slide-level aggregation over the sampled patches of each slide
    df = pd.DataFrame(dict(wsi=wsi, p=scores["nearest centroid, k = 50 (median draw)"], y=y))
    slide = df.groupby("wsi").agg(tumour_slide=("wsi", lambda s: "tumor" in s.iloc[0]), n=("y", "size"),
                                  max=("p", "max"), top5=("p", lambda s: np.sort(s.to_numpy())[-max(1, len(s) // 20):].mean()),
                                  frac=("p", lambda s: (s >= 0.5).mean()), mean=("p", "mean"))
    yt = slide.tumour_slide.astype(int).to_numpy()
    agg = {a: auc(yt, slide[a].to_numpy()) for a in ("max", "top5", "frac", "mean")}
    pd.Series(agg).round(3).to_csv(OUT / "C8_slide_level_aggregation.csv")
    print("slide-level AUROC by aggregation:", {k: round(v, 3) for k, v in agg.items()}, "| slides", len(slide))

    # sensitivity by metastatic burden of the slide (fraction of the slide's sampled patches that are tumour)
    pos = df[df.y == 1].groupby("wsi").p.apply(lambda s: (s >= 0.5).mean())
    burden = df.groupby("wsi").y.mean().loc[pos.index]
    t = pd.DataFrame(dict(sensitivity=pos, burden=burden))
    t["burden_tertile"] = pd.qcut(t.burden, 3, labels=["lowest", "middle", "highest"])
    bt = t.groupby("burden_tertile").agg(slides=("sensitivity", "size"), burden_range_min=("burden", "min"),
                                         burden_range_max=("burden", "max"), median_sensitivity=("sensitivity", "median"))
    bt.round(3).to_csv(OUT / "C8_sensitivity_by_burden.csv")
    print(bt.round(3).to_string())


# ============================================================================== C17: temperature
def temperature():
    d = D("phikon")
    from metrics import expected_calibration_error
    rows = []
    for tau in (2, 5, 10, 20, 50):
        e = []
        for rep in range(20):
            sel = draw_shots(d.y_adapt, 50, rep)
            p = 1 / (1 + np.exp(-tau * proto_score(d.za[sel], d.y_adapt[sel], d.zt)))
            e.append(expected_calibration_error(d.y_test, p))
        rows.append(dict(tau=tau, ece_mean=np.mean(e), ece_sd=np.std(e, ddof=1)))
    pd.DataFrame(rows).round(3).to_csv(OUT / "C17_temperature_sweep.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False))


# ============================================================================== C10: model versions
def versions():
    """Revisions actually used: the snapshot hashes present in the local Hugging Face cache."""
    from huggingface_hub import scan_cache_dir
    cached = {r.repo_id: sorted(rev.commit_hash for rev in r.revisions) for r in scan_cache_dir().repos}
    repos = {"efficientnet_b0": "timm/tf_efficientnet_b0.in1k", "convnextv2_tiny": "timm/convnextv2_tiny.fcmae_ft_in22k_in1k",
             "swinv2_tiny": "timm/swinv2_tiny_window8_256.ms_in1k", "swinv2_base": "timm/swinv2_base_window8_256.ms_in1k",
             "mambavision_tiny": "nvidia/MambaVision-T-1K", "dinov2_base": "timm/vit_base_patch14_dinov2.lvd142m",
             "phikon": "owkin/phikon", "phikon_v2": "owkin/phikon-v2", "plip": "vinid/plip"}
    rows = []
    for k, r in repos.items():
        rows.append(dict(model=k, repository=r, revision=", ".join(cached.get(r, ["not in local cache"]))))
    pd.DataFrame(rows).to_csv(OUT / "C10_model_revisions.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    versions()
    cross_centre()
    clinical()
    temperature()
    distribution_analysis()
    heads_and_statistics()
    support_composition()
