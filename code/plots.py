"""Step 6 - result figures and paper-ready summary tables from the saved metrics.

Reads tables/*.csv and predictions/**.csv; writes figures/*.png|pdf and
tables/summary_*.csv. Each figure is independent: missing inputs skip that figure only.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

import config as C

FAMILY_COLOR = {
    "CNN (ImageNet)": "#8c6d31",
    "Hierarchical transformer (ImageNet)": "#3d5a99",
    "State-space (ImageNet)": "#6b4a9c",
    "General SSL ViT": "#2f7d68",
    "Pathology foundation model": "#b5533c",
}
LABEL = {
    "efficientnet_b0_96px": "EfficientNet-B0 (96 px)", "efficientnet_b0": "EfficientNet-B0",
    "convnextv2_tiny": "ConvNeXt V2-T", "swinv2_tiny": "Swin V2-T", "swinv2_base": "Swin V2-B",
    "mambavision_tiny": "MambaVision-T", "dinov2_base": "DINOv2 ViT-B", "phikon": "Phikon",
    "phikon_v2": "Phikon-v2", "uni": "UNI", "uni2_h": "UNI2-h", "h_optimus_0": "H-optimus-0",
    "h_optimus_1": "H-optimus-1", "virchow2": "Virchow2",
}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
                     "savefig.facecolor": "white"})


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(C.FIG_DIR / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  figure:", name)


def order(keys):
    reg = list(C.BACKBONES)
    return sorted(keys, key=reg.index)


def lab(k):
    return LABEL.get(k, k)


def col(k):
    return FAMILY_COLOR.get(C.BACKBONES[k]["family"], "#52606d")


def mark(k):
    """Marker that tells apart models sharing a family colour."""
    fam = C.BACKBONES[k]["family"]
    same = [b for b in C.BACKBONES if C.BACKBONES[b]["family"] == fam]
    return "osD^vP*X"[same.index(k) % 8]


# ----------------------------------------------------------------------------- tables
def summary_tables(lp):
    cols = ["accuracy", "sensitivity", "specificity", "precision", "f1", "balanced_accuracy",
            "mcc", "auroc", "auprc", "ece"]
    nat = lp[(lp.protocol == "native")]
    out = nat[["backbone", "family", "eval_set", "threshold_rule", "threshold", "n"] + cols +
              [c for c in nat.columns if c.endswith(("_lo", "_hi"))]].copy()
    out.insert(0, "model", out.backbone.map(lab))
    out.to_csv(C.TABLE_DIR / "summary_native_split.csv", index=False)

    rnd = lp[lp.protocol == "random"].groupby(["backbone", "eval_set", "threshold_rule"])[cols] \
        .agg(["mean", "std"]).round(4)
    rnd.columns = [f"{m}_{s}" for m, s in rnd.columns]
    rnd.reset_index().to_csv(C.TABLE_DIR / "summary_random_split_5seeds.csv", index=False)

    lodo = lp[lp.protocol.str.startswith("lodo")]
    lodo[["backbone", "protocol", "eval_set", "threshold_rule"] + cols].to_csv(
        C.TABLE_DIR / "summary_leave_one_dataset_out.csv", index=False)

    # headline table: one row per backbone
    f = nat[nat.threshold_rule == "fixed_0.5"]
    i, e = f[f.eval_set == "internal"].set_index("backbone"), f[f.eval_set == "external"].set_index("backbone")
    head = pd.DataFrame({
        "model": [lab(k) for k in order(i.index)],
        "family": [C.BACKBONES[k]["family"] for k in order(i.index)],
        "internal_acc": i.loc[order(i.index), "accuracy"].values,
        "internal_auroc": i.loc[order(i.index), "auroc"].values,
        "external_acc": e.loc[order(i.index), "accuracy"].values,
        "external_sens": e.loc[order(i.index), "sensitivity"].values,
        "external_spec": e.loc[order(i.index), "specificity"].values,
        "external_bal_acc": e.loc[order(i.index), "balanced_accuracy"].values,
        "external_mcc": e.loc[order(i.index), "mcc"].values,
        "external_auroc": e.loc[order(i.index), "auroc"].values,
    }).round(4)
    head["auroc_gap"] = (head.internal_auroc - head.external_auroc).round(4)
    head.to_csv(C.TABLE_DIR / "headline_results.csv", index=False)
    print(head.to_string(index=False))


# ----------------------------------------------------------------------------- figures
def fig_internal_vs_external(lp):
    f = lp[(lp.protocol == "native") & (lp.threshold_rule == "fixed_0.5")]
    keys = order(f.backbone.unique())
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9), sharey=True)
    for ax, metric, title in zip(axes, ("auroc", "balanced_accuracy", "specificity"),
                                 ("AUROC", "Balanced accuracy", "Specificity")):
        y = np.arange(len(keys))
        for eval_set, off, alpha in (("internal", -0.19, 0.45), ("external", 0.19, 1.0)):
            g = f[f.eval_set == eval_set].set_index("backbone").loc[keys]
            ax.barh(y + off, g[metric], 0.36, color=[col(k) for k in keys], alpha=alpha,
                    edgecolor="white", linewidth=0.6)
            if f"{metric}_lo" in g:
                err = np.vstack([g[metric] - g[f"{metric}_lo"], g[f"{metric}_hi"] - g[metric]])
                ax.errorbar(g[metric], y + off, xerr=np.nan_to_num(err), fmt="none",
                            ecolor="#1f2933", lw=0.8, capsize=1.5)
            hi = g[f"{metric}_hi"].fillna(g[metric]) if f"{metric}_hi" in g else g[metric]
            for yy, v, h in zip(y + off, g[metric], hi):
                ax.text(max(v, h) + 0.012, yy, f"{v:.2f}", va="center", fontsize=6.6, color="#1f2933")
        ax.set_yticks(y)
        ax.set_yticklabels([lab(k) for k in keys])
        ax.set_xlim(0, 1.12)
        ax.set_title(title, fontweight="bold")
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
    axes[0].text(0.0, -0.13, "pale = internal oral test    solid = external lymph node test",
                 transform=axes[0].transAxes, fontsize=7.5, color="#52606d")
    save(fig, "fig_internal_vs_external")


def fig_roc_external(lp):
    keys = order(lp.backbone.unique())
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    for ax, part, title in zip(axes, ("internal", "external"),
                               ("Internal test (oral)", "External test (lymph node)")):
        for k in keys:
            p = C.PRED_DIR / "linear_probe" / f"{k}__native__{part}.csv"
            if not p.exists():
                continue
            d = pd.read_csv(p)
            fpr, tpr, _ = roc_curve(d.label, d.prob)
            auc = lp[(lp.backbone == k) & (lp.protocol == "native") & (lp.eval_set == part)].auroc.iloc[0]
            ax.plot(fpr, tpr, color=col(k), lw=1.6,
                    ls="-" if "foundation" in C.BACKBONES[k]["family"] or "SSL" in C.BACKBONES[k]["family"] else "--",
                    label=f"{lab(k)} ({auc:.3f})", alpha=0.9)
        ax.plot([0, 1], [0, 1], color="#9aa5b1", lw=0.8, ls=":")
        ax.set_xlabel("1 - specificity")
        ax.set_ylabel("Sensitivity")
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=6.8, loc="lower right", frameon=False, title="AUROC", title_fontsize=7)
    save(fig, "fig_roc_curves")


def fig_gap_scatter(lp):
    f = lp[(lp.protocol == "native") & (lp.threshold_rule == "fixed_0.5")]
    i = f[f.eval_set == "internal"].set_index("backbone")
    e = f[f.eval_set == "external"].set_index("backbone")
    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    for k in order(i.index):
        ax.scatter(i.loc[k, "auroc"], e.loc[k, "auroc"], s=70, color=col(k), marker=mark(k),
                   edgecolor="white", zorder=3)
        ax.annotate(lab(k), (i.loc[k, "auroc"], e.loc[k, "auroc"]), xytext=(5, 4),
                    textcoords="offset points", fontsize=7)
    lo = min(e.auroc.min(), i.auroc.min()) - 0.05
    ax.plot([lo, 1], [lo, 1], color="#9aa5b1", lw=0.8, ls=":")
    ax.set_xlim(lo, 1.005)
    ax.set_ylim(lo, 1.005)
    ax.set_xlabel("Internal AUROC (oral test)")
    ax.set_ylabel("External AUROC (lymph node)")
    ax.set_title("Generalisation gap (distance below the diagonal)", fontweight="bold", fontsize=9.5)
    for fam, c in FAMILY_COLOR.items():
        ax.scatter([], [], color=c, label=fam, s=40)
    ax.legend(fontsize=6.6, frameon=False, loc="lower left")
    save(fig, "fig_generalisation_gap")


def fig_fewshot(ad):
    keys = order(ad.backbone.unique())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for ax, metric, title in zip(axes, ("auroc", "balanced_accuracy"), ("AUROC", "Balanced accuracy")):
        for k in keys:
            g = ad[(ad.backbone == k) & (ad.method == "fewshot_prototype")].groupby("k")[metric]
            m, s = g.mean(), g.std()
            ax.plot(m.index, m.values, marker=mark(k), ms=4.5, lw=1.5, color=col(k), label=lab(k),
                    ls="-" if "foundation" in C.BACKBONES[k]["family"] else "--")
            ax.fill_between(m.index, m - s, m + s, color=col(k), alpha=0.10)
            d = ad[(ad.backbone == k) & (ad.method == "direct")][metric].iloc[0]
            ax.scatter([0.55], [d], marker=mark(k), s=26, color=col(k), zorder=3)
        ax.set_xscale("log")
        ax.set_xticks([0.55] + list(C.FEWSHOT_K))
        ax.set_xticklabels(["0\n(direct)"] + [str(k) for k in C.FEWSHOT_K])
        ax.set_xlabel("Labelled lymph node patches per class")
        ax.set_ylabel(f"External {title}")
        ax.set_title(f"Few-shot adaptation: {title}", fontweight="bold")
    axes[1].legend(fontsize=6.8, frameon=False, loc="lower right", ncol=2)
    save(fig, "fig_fewshot_curves")


def fig_adaptation_ladder(ad):
    keys = order(ad.backbone.unique())
    steps = [("direct", 0, "Direct"), ("target_standardise", 0, "Target\nstandardise"),
             ("coral", 0, "CORAL"), ("self_training", 0, "Self-\ntraining"),
             ("fewshot_prototype", 10, "10-shot"), ("fewshot_prototype", 100, "100-shot"),
             ("oracle_target_supervised", 0, "Oracle")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), sharey=True)
    for ax, metric, title in zip(axes, ("balanced_accuracy", "specificity"),
                                 ("Balanced accuracy", "Specificity")):
        x = np.arange(len(steps))
        for k in keys:
            vals = [ad[(ad.backbone == k) & (ad.method == m) & (ad.k == kk)][metric].mean()
                    for m, kk, _ in steps]
            ax.plot(x, vals, marker=mark(k), ms=4.5, lw=1.4, color=col(k), label=lab(k),
                    ls="-" if "foundation" in C.BACKBONES[k]["family"] else "--")
        ax.set_xticks(x)
        ax.set_xticklabels([s[2] for s in steps], fontsize=7.5)
        ax.axvspan(0.5, 3.5, color="#2f7d68", alpha=0.05)
        ax.axvspan(3.5, 5.5, color="#3d5a99", alpha=0.05)
        ax.set_title(f"External {title}", fontweight="bold")
        ax.set_ylim(0.3, 1.0)
    axes[0].text(2.0, 0.315, "unlabelled target data", ha="center", fontsize=7, color="#2f7d68")
    axes[0].text(4.5, 0.315, "labelled shots", ha="center", fontsize=7, color="#3d5a99")
    axes[1].legend(fontsize=6.8, frameon=False, loc="lower right", ncol=2)
    save(fig, "fig_adaptation_ladder")


def fig_confusion(lp):
    f = lp[(lp.protocol == "native") & (lp.threshold_rule == "fixed_0.5") & (lp.eval_set == "external")]
    keys = order(f.backbone.unique())
    n = len(keys)
    ncol = min(5, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.5 * ncol, 2.7 * nrow), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, k in zip(axes.ravel(), keys):
        r = f[f.backbone == k].iloc[0]
        m = np.array([[r.tn, r.fp], [r.fn, r.tp]], float)
        pct = m / m.sum(1, keepdims=True)
        ax.axis("on")
        ax.grid(False)
        ax.imshow(pct, cmap="Blues", vmin=0, vmax=1)
        for (a, b), v in np.ndenumerate(m):
            ax.text(b, a, f"{int(v):,}\n{pct[a, b]:.0%}", ha="center", va="center", fontsize=7.5,
                    color="white" if pct[a, b] > 0.55 else "#1f2933")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Normal", "Metastasis"], fontsize=7)
        ax.set_yticklabels(["Normal", "Metastasis"], fontsize=7, rotation=90, va="center")
        ax.set_title(lab(k), fontsize=8.2, fontweight="bold")
        for s in ax.spines.values():
            s.set_visible(False)
    fig.supxlabel("Predicted", fontsize=9)
    fig.supylabel("Actual", fontsize=9)
    save(fig, "fig_confusion_external")


def fig_calibration(lp):
    keys = order(lp.backbone.unique())
    fig, ax = plt.subplots(figsize=(5, 4.6))
    for k in keys:
        p = C.PRED_DIR / "linear_probe" / f"{k}__native__external.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p)
        bins = np.minimum((d.prob * 10).astype(int), 9)
        g = d.groupby(bins).agg(conf=("prob", "mean"), acc=("label", "mean"))
        ax.plot(g.conf, g.acc, marker=mark(k), ms=3.5, lw=1.3, color=col(k), label=lab(k),
                ls="-" if "foundation" in C.BACKBONES[k]["family"] else "--")
    ax.plot([0, 1], [0, 1], color="#9aa5b1", lw=0.8, ls=":")
    ax.set_xlabel("Predicted probability of metastasis")
    ax.set_ylabel("Observed fraction metastatic")
    ax.set_title("Calibration on the external test", fontweight="bold")
    ax.legend(fontsize=6.6, frameon=False)
    save(fig, "fig_calibration_external")


def fig_embedding_shift(keys):
    """2-D projection of source and target embeddings: where the domain gap is visible."""
    try:
        import umap
    except Exception:
        print("  (umap not installed - embedding figure skipped)")
        return
    index = pd.read_csv(C.DEV_INDEX_CSV)
    pcam = pd.read_csv(C.CACHE_DIR / "pcam_test_index.csv")
    rng = np.random.default_rng(C.SEED)
    si = rng.choice(len(index), 3000, replace=False)
    ti = rng.choice(len(pcam), 3000, replace=False)
    fig, axes = plt.subplots(1, len(keys), figsize=(4.4 * len(keys), 4.2), squeeze=False)
    groups = [("Oral normal", index.label.to_numpy()[si] == 0, "#2f7d68", True),
              ("Oral SCC", index.label.to_numpy()[si] == 1, "#b5533c", True),
              ("Lymph node normal", pcam.label.to_numpy()[ti] == 0, "#7fb8e0", False),
              ("Lymph node metastasis", pcam.label.to_numpy()[ti] == 1, "#6b4a9c", False)]
    for ax, k in zip(axes[0], keys):
        fs = np.load(C.FEATURE_DIR / k / "dev.npy", mmap_mode="r")[np.sort(si)]
        ft = np.load(C.FEATURE_DIR / k / "pcam_test.npy", mmap_mode="r")[np.sort(ti)]
        x = np.vstack([fs, ft]).astype(np.float32)
        x = (x - x.mean(0)) / (x.std(0) + 1e-6)
        z = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=C.SEED).fit_transform(x)
        zs, zt = z[:len(si)], z[len(si):]
        order_s, order_t = np.argsort(np.argsort(si)), np.argsort(np.argsort(ti))
        zs, zt = zs[order_s], zt[order_t]
        for name, mask, c, is_src in groups:
            pts = zs[mask] if is_src else zt[mask]
            ax.scatter(pts[:, 0], pts[:, 1], s=3, color=c, alpha=0.55, label=name, linewidths=0)
        ax.set_title(lab(k), fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    axes[0][0].legend(fontsize=7, frameon=False, markerscale=4, loc="best")
    save(fig, "fig_embedding_domain_shift")


def fig_finetune(ft):
    f = ft[ft.threshold_rule == "fixed_0.5"]
    runs = f.run.unique()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    for ax, metric, title in zip(axes, ("auroc", "balanced_accuracy", "specificity"),
                                 ("AUROC", "Balanced accuracy", "Specificity")):
        y = np.arange(len(runs))
        for eval_set, off, alpha in (("internal", -0.19, 0.45), ("external", 0.19, 1.0)):
            g = f[f.eval_set == eval_set].set_index("run").loc[runs]
            ax.barh(y + off, g[metric], 0.36, color=[col(b) for b in g.backbone], alpha=alpha)
            for yy, v in zip(y + off, g[metric]):
                ax.text(v + 0.012, yy, f"{v:.2f}", va="center", fontsize=6.6)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{lab(r.split('__')[0])}\n{'+ stain aug.' if 'stainaug' in r else 'no stain aug.'}"
                            for r in runs], fontsize=7.5)
        ax.set_xlim(0, 1.12)
        ax.set_title(f"Fine-tuned: {title}", fontweight="bold")
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
    save(fig, "fig_finetune_architectures")


def main():
    jobs = []
    lp_path, ad_path, ft_path = (C.TABLE_DIR / n for n in
                                 ("linear_probe_metrics.csv", "adaptation_metrics.csv", "finetune_metrics.csv"))
    if lp_path.exists():
        lp = pd.read_csv(lp_path)
        jobs += [(summary_tables, lp), (fig_internal_vs_external, lp), (fig_roc_external, lp),
                 (fig_gap_scatter, lp), (fig_confusion, lp), (fig_calibration, lp)]
        nat = lp[(lp.protocol == "native") & (lp.eval_set == "external") & (lp.threshold_rule == "fixed_0.5")]
        picks = ["efficientnet_b0"] if "efficientnet_b0" in set(nat.backbone) else []
        best = nat.sort_values("auroc", ascending=False).backbone.iloc[0]
        picks += [k for k in ("swinv2_tiny", best) if k in set(nat.backbone) and k not in picks]
        jobs.append((fig_embedding_shift, picks))
    if ad_path.exists():
        ad = pd.read_csv(ad_path)
        jobs += [(fig_fewshot, ad), (fig_adaptation_ladder, ad)]
    if ft_path.exists() and len(pd.read_csv(ft_path)):
        jobs.append((fig_finetune, pd.read_csv(ft_path)))
    for fn, arg in jobs:
        try:
            fn(arg)
        except Exception as exc:
            print(f"  {fn.__name__} failed: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
