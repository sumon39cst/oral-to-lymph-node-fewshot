"""Figures changed in the second review round.

    Figure 4   (a) redrawn as a dot plot (Reviewer 2: too many overlapping bars); (b), (c) unchanged in content
    Figure 9   (b) label noise extended to 45% and 50%, with the effective-sample-size curve (Reviewer 2)
    Figure 10  dashed square = central 32 x 32 px region that defines the PatchCamelyon label (Reviewer 1)
    Figure 13  new: unlabelled-pool composition for normal-only labels (Reviewer 1)
Output: Majore Revision/revised/figures/ (PNG 600 dpi + PDF)
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
import config as C  # noqa: E402
from plots import LABEL  # noqa: E402

MR = Path(__file__).resolve().parents[2] / "results"
AN, FIG = MR / "analysis", MR / "revised" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
REV = C.RESULT_DIR / "revision"
INK, MUTED, GRID = "#1a1a1a", "#5f6b76", "#e6e9ec"
RED, LBLUE, BLUE, PURPLE, GREY, GREEN = "#c0392b", "#9ecae1", "#3182bd", "#6a51a3", "#636363", "#2e8b57"
STYLE = {"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 8.5,
         "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
         "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
         "legend.frameon": False, "legend.fontsize": 7.6, "axes.titlesize": 9, "axes.titleweight": "bold",
         "savefig.facecolor": "white"}


def letter(ax, s, x=-0.09, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom", ha="left")


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=600 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


# ================================================================================ Figure 4
def figure4():
    ENC = C.DEFAULT_BACKBONES
    norm = pd.read_csv(REV / "M1_normal_only_labels_summary.csv")
    fp = pd.read_csv(REV / "M1_false_positive_breakdown.csv").set_index("encoder")
    ad = pd.read_csv(C.TABLE_DIR / "adaptation_metrics.csv")
    fov = pd.read_csv(REV / "R2_R5_field_of_view_ci_and_ds3_only.csv")
    fov = fov[fov.source_training_set == "all datasets"]
    names = [LABEL[k] for k in ENC]

    def one_class(k_, n):
        return norm[(norm.encoder == k_) & norm.method.str.startswith("one-class") & (norm.k == n)]["mean"].iloc[0]

    series = [("Direct transfer (no target label)", RED, "o",
               lambda k_: norm[(norm.encoder == k_) & (norm.method == "direct transfer")]["mean"].iloc[0]),
              ("50 normal patches only", LBLUE, "v", lambda k_: one_class(k_, 50)),
              ("1,000 normal patches only", BLUE, "^", lambda k_: one_class(k_, 1000)),
              ("50 + 50 labelled (prototypes)", PURPLE, "D",
               lambda k_: ad[(ad.backbone == k_) & (ad.method == "fewshot_prototype") & (ad.k == 50)].auroc.mean())]
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(7.2, 7.2))
        gs = fig.add_gridspec(2, 2, height_ratios=[1.12, 1.0], width_ratios=[1, 1.08], hspace=0.5, wspace=0.42)
        # (a) dot plot: one row per encoder, one marker per amount of target information
        ax = fig.add_subplot(gs[0, :])
        yy = np.arange(len(ENC))[::-1]
        vals = np.array([[fn(k) for k in ENC] for *_, fn in series])
        for j in range(len(ENC)):
            ax.plot([vals[:, j].min(), vals[:, j].max()], [yy[j]] * 2, color="#c9ced3", lw=1.6, zorder=1)
        for (name, c, mk, _), v in zip(series, vals):
            ax.scatter(v, yy, s=34, color=c, marker=mk, edgecolor="white", linewidth=0.5, zorder=3, label=name)
        ax.axvline(0.5, color=INK, lw=0.7, ls=(0, (3, 3)))
        ax.set_yticks(yy)
        ax.set_yticklabels(names, fontsize=7.8)
        ax.set_xlim(0.3, 1.0)
        ax.set_xlabel("External AUROC (PatchCamelyon evaluation pool)")
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
        ax.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0), handletextpad=0.4, columnspacing=1.6)
        letter(ax, "a", x=-0.2, y=1.1)
        # (b) heat map (unchanged)
        ax = fig.add_subplot(gs[1, 0])
        m = np.array([[fp.loc[k, f"fp_rate_nuclear_density_{t}"] for t in ("low", "middle", "high")] for k in ENC])
        cmap = LinearSegmentedColormap.from_list("fp", ["#fff5f0", "#fcbba1", "#fb6a4a", "#a50f15"])
        im = ax.imshow(m, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for (i, j), v in np.ndenumerate(m):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.2, color="white" if v > 0.7 else INK)
        ax.set_xticks(range(3))
        ax.set_xticklabels(["Low", "Middle", "High"])
        ax.set_yticks(range(len(ENC)))
        ax.set_yticklabels(names, fontsize=7.6)
        ax.set_xlabel("Nuclear density of normal patch (tertile)")
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
        cb = fig.colorbar(im, ax=ax, fraction=0.06, pad=0.03)
        cb.set_label("False-positive rate", fontsize=7.6)
        cb.ax.tick_params(labelsize=7, width=0.6, length=2)
        cb.outline.set_linewidth(0.6)
        ax.set_title("False positives of direct transfer", loc="left", fontsize=8.5)
        letter(ax, "b", x=-0.62, y=1.03)
        # (c) field of view (unchanged)
        ax = fig.add_subplot(gs[1, 1])
        order = ["full field", "1/2 field", "1/4 field", "1/8 field"]
        styles = {"efficientnet_b0": ("#8c6d31", "o"), "phikon": (RED, "s"), "phikon_v2": (BLUE, "D")}
        for j, k in enumerate(styles):
            c, mk = styles[k]
            d = fov[fov.encoder == k].set_index("field_of_view").loc[order]
            xs = np.arange(4) + (j - 1) * 0.08
            ax.errorbar(xs, d.external_auroc, yerr=[d.external_auroc - d.ci_lo, d.ci_hi - d.external_auroc],
                        color=c, marker=mk, ms=4.2, lw=1.3, capsize=2.2, elinewidth=0.9, label=LABEL[k])
            ax.plot(xs, d.internal_auroc, color=c, marker=mk, ms=3.6, lw=0.9, ls=(0, (2, 2)), mfc="white")
        ax.axhline(0.5, color=INK, lw=0.7, ls=(0, (3, 3)))
        ax.set_xticks(range(4))
        ax.set_xticklabels(["Full", "1/2", "1/4", "1/8"])
        ax.set_xlabel("Side of oral field used for training")
        ax.set_ylabel("AUROC")
        ax.set_ylim(0.2, 1.03)
        ax.grid(axis="x", visible=False)
        h, l = ax.get_legend_handles_labels()
        h += [plt.Line2D([], [], color=INK, lw=1.3), plt.Line2D([], [], color=INK, lw=0.9, ls=(0, (2, 2)))]
        l += ["External (lymph node)", "Internal (oral)"]
        ax.legend(h, l, loc="center", bbox_to_anchor=(0.5, 0.715), ncol=2, handlelength=1.8, fontsize=6.9,
                  columnspacing=1.0)
        ax.set_title("Field-of-view control", loc="left", fontsize=8.5)
        letter(ax, "c", x=-0.2, y=1.03)
        save(fig, "Figure4_sources_of_shift")


# ================================================================================ Figure 9
def figure9():
    rob = pd.read_csv(REV / "M9_robustness_summary.csv")
    nz = pd.read_csv(AN / "R2_5_label_noise_all.csv")
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.3))
        # (a) support slides
        ax = axes[0]
        sub = rob[rob.experiment == "support slides"]
        order = ["1 slide(s)", "2 slide(s)", "5 slide(s)", "all 108 slides"]
        for head, c, mk in (("prototype", PURPLE, "o"), ("logistic regression", GREY, "s")):
            for k, ls in ((10, (0, (3, 2))), (50, "-")):
                d = sub[(sub["head"] == head) & (sub.k == k)].set_index("setting").loc[order]
                ax.errorbar(range(4), d["mean"], yerr=d["std"], color=c, marker=mk, ms=3.8, lw=1.2, ls=ls, capsize=2,
                            label=f"{'prototypes' if head == 'prototype' else 'logistic regr.'}, k = {k}")
        ax.set_xticks(range(4))
        ax.set_xticklabels(["1", "2", "5", "108"])
        ax.set_xlabel("Slides supplying the support set")
        ax.set_ylabel("External AUROC")
        ax.set_ylim(0.6, 1.0)
        ax.legend(fontsize=6.6, loc="lower right")
        letter(ax, "a", x=-0.2, y=1.02)
        # (b) label noise, now 0-50%, with the effective sample size k(1-2rho)^2 on a second axis
        ax = axes[1]
        g = nz.groupby("rho")[["prototypes", "logistic_regression"]].agg(["mean", "std"])
        x = g.index.to_numpy() * 100
        for col, c, mk, lab in (("prototypes", PURPLE, "o", "prototypes"), ("logistic_regression", GREY, "s", "logistic regr.")):
            ax.errorbar(x, g[col]["mean"], yerr=g[col]["std"], color=c, marker=mk, ms=3.8, lw=1.2, capsize=2, label=lab)
        ax.axhline(0.5, color=INK, lw=0.7, ls=(0, (3, 3)))
        ax.set_xlabel("Support labels flipped (%)  [k = 50 per class]")
        ax.set_ylabel("External AUROC")
        ax.set_ylim(0.3, 1.0)
        top = ax.secondary_xaxis("top", functions=(lambda v: v, lambda v: v))
        top.set_xticks([0, 10, 20, 30, 40, 50])
        top.set_xticklabels([f"{50 * (1 - 2 * r / 100) ** 2:.0f}" for r in [0, 10, 20, 30, 40, 50]], fontsize=7)
        top.set_xlabel("Effective clean patches per class, k(1 \u2212 2\u03c1)\u00b2", fontsize=7.4)
        ax.legend(fontsize=6.8, loc="lower left")
        letter(ax, "b", x=-0.2, y=1.12)
        # (c) large supports (unchanged)
        ax = axes[2]
        sub = rob[rob.experiment == "large support"].copy()
        for head, c, mk, lab in (("prototype", PURPLE, "o", "prototypes"), ("logistic regression", GREY, "s", "logistic regr.")):
            d = sub[sub["head"] == head].sort_values("k")
            ax.errorbar(d.k, d["mean"], yerr=d["std"].fillna(0), color=c, marker=mk, ms=3.8, lw=1.2, capsize=2, label=lab)
        ax.set_xscale("log")
        ax.set_xlabel("Labelled patches per class (log scale)")
        ax.set_ylabel("External AUROC")
        ax.set_ylim(0.88, 0.99)
        ax.legend(fontsize=6.8, loc="lower right")
        letter(ax, "c", x=-0.2, y=1.02)
        fig.tight_layout()
        save(fig, "Figure9_robustness")


# ================================================================================ Figure 10
def figure10():
    x = np.load(C.CACHE_DIR / "pcam_test_x.npy", mmap_mode="r")
    pr = pd.read_csv(C.ROOT / "prposed model resul and prediction" / "predictions" / "predictions_phikon_k50.csv")
    test = pd.read_csv(C.CACHE_DIR / "pcam_test_index.csv")
    assert (pr.pcam_index.to_numpy() == test.pcam_index.to_numpy()).all()
    y, p = pr.label.to_numpy(), pr.probability.to_numpy()
    pred = p >= 0.5
    TUM, NOR, ERR = "#7b3fa0", "#2b7bba", "#b5533c"
    groups = [("True positive", (y == 1) & pred, TUM), ("True negative", (y == 0) & ~pred, NOR),
              ("False positive", (y == 0) & pred, ERR), ("False negative", (y == 1) & ~pred, ERR)]
    rng = np.random.default_rng(C.SEED)                    # same selection as the first submission
    with plt.rc_context({**STYLE, "axes.grid": False}):
        fig, axes = plt.subplots(4, 8, figsize=(13.2, 7.6))
        for r, (name, mask, col) in enumerate(groups):
            idx = np.flatnonzero(mask)
            idx = np.array([i for i in rng.permutation(idx)[:400] if x[i].mean() < 190 and x[i].std() > 30][:8])
            for c in range(8):
                ax = axes[r, c]
                ax.set_xticks([])
                ax.set_yticks([])
                if c < len(idx):
                    ax.imshow(np.asarray(x[idx[c]]), interpolation="lanczos")
                    ax.add_patch(Rectangle((31.5, 31.5), 32, 32, fill=False, ec="white", lw=1.4, ls=(0, (3, 2))))
                    ax.add_patch(Rectangle((31.5, 31.5), 32, 32, fill=False, ec="black", lw=0.5, ls=(0, (3, 2)),
                                           alpha=0.6))
                    ax.set_xlabel(f"p = {p[idx[c]]:.2f}", fontsize=8, color=col, fontweight="bold", labelpad=2)
                for s in ax.spines.values():
                    s.set_visible(True)
                    s.set_color(col)
                    s.set_linewidth(2.2)
            axes[r, 0].set_ylabel(f"{name}\n(n = {int(mask.sum()):,})", fontsize=9, fontweight="bold", color=col)
        fig.suptitle("Prototype-classifier predictions on unseen slides (Phikon, k = 50 per class, median draw)",
                     fontsize=10.5, fontweight="bold", y=0.945)
        fig.text(0.5, 0.905, "Dashed square: central 32 \u00d7 32 pixel region; a patch is labelled metastatic only if "
                             "tumour is present inside this region.", ha="center", fontsize=8.5, color=MUTED)
        save(fig, "Figure10_prediction_gallery")


# ================================================================================ Figure 13 (new)
def figure13():
    r = pd.read_csv(AN / "R1_2_pool_composition_all.csv")
    sub = r[r.pool_size == 9000]
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(4.6, 3.2))
        fracs = [0.0, 0.01, 0.10, 0.25, 0.50]                 # categorical positions, equally spaced
        pos = np.arange(len(fracs))
        for j, (head, c, mk, lab) in enumerate((("normal-only centroid (-s0), k = 50 normals", BLUE, "o",
                                                 "50 labelled normal patches + pool"),
                                                ("two-class prototypes, k = 50 per class", PURPLE, "D",
                                                 "50 + 50 labelled patches + pool"))):
            d = sub[sub["head"] == head].groupby("tumour_fraction").auroc.agg(["mean", "std"]).reindex(fracs)
            ax.errorbar(pos + (j - 0.5) * 0.08, d["mean"], yerr=d["std"], color=c, marker=mk, ms=4, lw=1.3,
                        capsize=2, label=lab)
        oc = r[r["head"].str.startswith("one-class")].auroc
        ax.axhline(oc.mean(), color=GREY, lw=1, ls=(0, (4, 2)), label="50 normal patches, no pool (one-class)")
        ax.axhline(0.5, color=INK, lw=0.7, ls=(0, (3, 3)))
        ax.set_xlabel("Tumour patches in the unlabelled pool (%), pool of 9,000 patches")
        ax.set_ylabel("External AUROC (mean \u00b1 SD, 20 draws)")
        ax.set_xticks(pos)
        ax.set_xticklabels(["0", "1", "10", "25", "50"])
        ax.set_ylim(0.4, 1.0)
        ax.grid(axis="x", visible=False)
        ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)
        save(fig, "Figure13_pool_composition")


# ================================================================================ Figures 6 and 7 (old 5 and 6)
def figure_perf():
    """Old Figure 5 (now 6) redrawn with the revised target-supervised reference (Reviewer 1)."""
    ad = pd.read_csv(C.TABLE_DIR / "adaptation_metrics.csv")
    ref = pd.read_csv(AN / "R1_1_reference_all_encoders.csv").set_index("encoder")
    ks = [1, 2, 5, 10, 20, 50, 100]
    panels = [("auroc", "AUROC"), ("balanced_accuracy", "Balanced accuracy"), ("sensitivity", "Sensitivity"),
              ("specificity", "Specificity")]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 4, figsize=(12.0, 3.4), sharey=False)
        for ax, (m, title) in zip(axes, panels):
            for key, c, mk, lab in (("phikon", "#7b3fa0", "o", "Phikon (ViT-B/16)"), ("phikon_v2", "#3a5a98", "s", "Phikon-v2 (ViT-L/16)")):
                a = ad[(ad.backbone == key) & (ad.method == "fewshot_prototype")].groupby("k")[m].agg(["mean", "std"]).loc[ks]
                ax.plot(ks, a["mean"], color=c, marker=mk, ms=4, lw=1.5, label=lab)
                ax.fill_between(ks, a["mean"] - a["std"], a["mean"] + a["std"], color=c, alpha=0.13, lw=0)
            direct = ad[(ad.backbone == "phikon") & (ad.method == "direct")][m].iloc[0]
            sup = ref.loc["phikon", m]
            ax.axhline(sup, color="#4d4d4d", lw=0.9, ls=":")
            ax.axhline(direct, color=RED, lw=0.9, ls=":")
            ax.text(1.05, sup + 0.008, f"all labels {sup:.2f}", fontsize=7, color="#4d4d4d")
            ax.text(1.05, direct + 0.008, f"direct transfer {direct:.2f}", fontsize=7, color=RED)
            ax.set_xscale("log")
            ax.set_xticks(ks)
            ax.set_xticklabels([str(k) for k in ks])
            ax.minorticks_off()
            ax.set_ylim(0.25, 1.03)
            ax.set_title(title)
            ax.set_xlabel("Labelled patches per class (k)")
        axes[0].set_ylabel("External test (lymph node)")
        axes[0].legend(fontsize=7, loc="lower right")
        fig.tight_layout()
        save(fig, "Figure6_performance_vs_k")


def figure_roc():
    """Old Figure 6 (now 7): ROC and precision-recall curves with the revised reference (Reviewer 1)."""
    from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
    pr = C.ROOT / "prposed model resul and prediction" / "predictions"
    curves = [("Direct transfer (k = 0)", pd.read_csv(pr / "reference_direct_phikon.csv"), "#8c6d31", (0, (5, 2))),
              ("Prototypes, k = 10", pd.read_csv(pr / "predictions_phikon_k10.csv"), "#3a5a98", (0, (5, 2))),
              ("Prototypes, k = 50", pd.read_csv(pr / "predictions_phikon_k50.csv"), "#c0553f", "-"),
              ("Prototypes, k = 100", pd.read_csv(pr / "predictions_phikon_k100.csv"), "#7b3fa0", (0, (5, 2))),
              ("All 20,000 labels", pd.read_csv(AN / "reference_cv_phikon.csv"), "#2f7d68", "-")]
    with plt.rc_context(STYLE):
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 4.6))
        for name, d, c, ls in curves:
            y, p = d.label.to_numpy(), d.probability.to_numpy()
            fpr, tpr, _ = roc_curve(y, p)
            a1.plot(fpr, tpr, color=c, ls=ls, lw=1.6, label=f"{name} ({roc_auc_score(y, p):.3f})")
            prec, rec, _ = precision_recall_curve(y, p)
            a2.plot(rec, prec, color=c, ls=ls, lw=1.6, label=f"{name} ({average_precision_score(y, p):.3f})")
        a1.plot([0, 1], [0, 1], color="#9a9a9a", lw=0.8, ls=":")
        a2.axhline(curves[0][1].label.mean(), color="#9a9a9a", lw=0.8, ls=":")
        a1.set(xlabel="1 - specificity", ylabel="Sensitivity", title="ROC curve (lymph node)")
        a2.set(xlabel="Recall (sensitivity)", ylabel="Precision", title="Precision-recall curve (lymph node)")
        a1.legend(title="AUROC", fontsize=7.4, title_fontsize=7.4, loc="lower right")
        a2.legend(title="AUPRC", fontsize=7.4, title_fontsize=7.4, loc="center left", bbox_to_anchor=(0.12, 0.62))
        fig.tight_layout()
        save(fig, "Figure7_ROC_PR")


if __name__ == "__main__":
    for f in (sys.argv[1:] or ["figure4", "figure9", "figure10", "figure13", "figure_perf", "figure_roc"]):
        globals()[f]()
