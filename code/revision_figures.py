"""Figures added in the revision, drawn from result_prediction/revision/*.csv.

    Fig_task_shift_tests   (a) normal-only target labels, (b) false positives of direct transfer
                           by nuclear density, (c) field-of-view control
    Fig_support_robustness (a) supports restricted to few slides, (b) label noise, (c) large supports
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as C
from plots import LABEL, col, mark

REV = C.RESULT_DIR / "revision"
OUT = C.ROOT / "prposed model resul and prediction" / "figures"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6, "savefig.facecolor": "white"})
ENC = C.DEFAULT_BACKBONES


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=400, bbox_inches="tight")
    plt.close(fig)
    print("figure:", name)


def task_shift():
    norm = pd.read_csv(REV / "M1_normal_only_labels_summary.csv")
    fp = pd.read_csv(REV / "M1_false_positive_breakdown.csv").set_index("encoder")
    ad = pd.read_csv(C.TABLE_DIR / "adaptation_metrics.csv")
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.2), gridspec_kw=dict(width_ratios=[1.45, 1, 1]))

    ax = axes[0]
    series = [("Direct transfer (no target data)", "#b5533c", lambda k_: norm[(norm.encoder == k_) & (norm.method == "direct transfer")]["mean"].iloc[0]),
              ("50 normal target patches, no tumour labels", "#7fb8e0", lambda k_: norm[(norm.encoder == k_) & (norm.method.str.startswith("one-class")) & (norm.k == 50)]["mean"].iloc[0]),
              ("1,000 normal target patches, no tumour labels", "#2b7bba", lambda k_: norm[(norm.encoder == k_) & (norm.method.str.startswith("one-class")) & (norm.k == 1000)]["mean"].iloc[0]),
              ("50 normal + 50 tumour patches (nearest centroid)", "#7b3fa0", lambda k_: ad[(ad.backbone == k_) & (ad.method == "fewshot_prototype") & (ad.k == 50)].auroc.mean()),
              ("All labels (supervised reference)", "#52606d", lambda k_: ad[(ad.backbone == k_) & (ad.method == "oracle_target_supervised")].auroc.iloc[0])]
    x = np.arange(len(ENC))
    w = 0.16
    for i, (name, c, fn) in enumerate(series):
        ax.bar(x + (i - 2) * w, [fn(k) for k in ENC], w * 0.94, color=c, label=name)
    ax.axhline(0.5, color="black", lw=0.7, ls=(0, (2, 3)))
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[k].replace(" (96 px)", "\n(96 px)") for k in ENC], rotation=35, ha="right", fontsize=7.5)
    ax.set_ylim(0.3, 1.0)
    ax.set_ylabel("External AUROC (lymph node)")
    ax.set_title("(a) Labelling only the normal class", fontweight="bold", loc="left")
    ax.legend(fontsize=7, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.36))
    ax.grid(axis="x", visible=False)

    ax = axes[1]
    for k in ENC:
        ax.plot([0, 1, 2], [fp.loc[k, f"fp_rate_nuclear_density_{t}"] for t in ("low", "middle", "high")],
                marker=mark(k), ms=5, lw=1.4, color=col(k), label=LABEL[k])
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["lowest third", "middle third", "highest third"])
    ax.set_xlabel("Nuclear density of the normal patch (dark-pixel fraction)")
    ax.set_ylabel("Fraction of normal patches called malignant")
    ax.set_ylim(0, 1.03)
    ax.set_title("(b) False positives of direct transfer", fontweight="bold", loc="left")
    ax.legend(fontsize=7, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.22))

    ax = axes[2]
    p = REV / "R2_R5_field_of_view_ci_and_ds3_only.csv"
    if p.exists():
        sc = pd.read_csv(p)
        sc = sc[sc.source_training_set == "all datasets"]
        order = ["full field", "1/2 field", "1/4 field", "1/8 field"]
        for j, k in enumerate(sc.encoder.unique()):
            d = sc[sc.encoder == k].set_index("field_of_view").loc[order]
            xs = np.arange(4) + (j - 1) * 0.06                      # small offset so intervals do not overlap
            ax.errorbar(xs, d.external_auroc, yerr=[d.external_auroc - d.ci_lo, d.ci_hi - d.external_auroc],
                        marker=mark(k), ms=5, lw=1.6, capsize=2.5, color=col(k), label=f"{LABEL[k]}, external")
            ax.plot(range(4), [d.loc[o, "internal_auroc"] for o in order], marker=mark(k), ms=4, lw=1.0,
                    color=col(k), ls=(0, (2, 2)), alpha=0.7, label=f"{LABEL[k]}, internal")
        ax.axhline(0.5, color="black", lw=0.7, ls=(0, (2, 3)))
        ax.set_xticks(range(4))
        ax.set_xticklabels(["full", "1/2", "1/4", "1/8"])
        ax.set_xlabel("Side of the oral field used for training (fraction of the image)")
        ax.set_ylabel("AUROC")
        ax.set_ylim(0.2, 1.02)
        ax.legend(fontsize=7, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    ax.set_title("(c) Field-of-view control", fontweight="bold", loc="left")
    save(fig, "Fig11_task_shift_tests")


def robustness():
    r = pd.read_csv(REV / "M9_robustness_summary.csv")
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.9), sharey=True)
    styles = {"prototype": ("#7b3fa0", "o", "-"), "logistic regression": ("#2f7d68", "s", (0, (4, 2)))}

    ax = axes[0]
    d = r[r.experiment == "support slides"]
    order = ["1 slide(s)", "2 slide(s)", "5 slide(s)", "all 108 slides"]
    for head, (c, m, ls) in styles.items():
        for k, alpha in ((10, 0.45), (50, 1.0)):
            g = d[(d["head"] == head) & (d.k == k)].set_index("setting").loc[order]
            ax.errorbar(range(4), g["mean"], yerr=g["std"], color=c, marker=m, ls=ls, lw=1.5, ms=4.5, capsize=2,
                        alpha=alpha, label=f"{head}, k = {k}")
    ax.set_xticks(range(4))
    ax.set_xticklabels(["1", "2", "5", "108 (all)"])
    ax.set_xlabel("Slides from which the support patches are drawn")
    ax.set_ylabel("External AUROC (Phikon)")
    ax.set_title("(a) Supports from few slides", fontweight="bold", loc="left")
    ax.legend(fontsize=7, frameon=False, loc="lower right")

    ax = axes[1]
    d = r[r.experiment == "label noise"]
    xs = [0, 10, 20, 30, 40]
    for head, (c, m, ls) in styles.items():
        g = d[d["head"] == head].set_index("setting").loc[[f"{v}% flipped" for v in xs]]
        ax.errorbar(xs, g["mean"], yerr=g["std"], color=c, marker=m, ls=ls, lw=1.5, ms=4.5, capsize=2, label=head)
    ax.set_xlabel("Support labels flipped (%), k = 50")
    ax.set_title("(b) Label noise in the support set", fontweight="bold", loc="left")
    ax.legend(fontsize=7, frameon=False, loc="lower left")

    ax = axes[2]
    d = r[r.experiment == "large support"]
    for head, (c, m, ls) in styles.items():
        g = d[d["head"] == head].sort_values("k")
        ax.errorbar(g.k, g["mean"], yerr=g["std"].fillna(0), color=c, marker=m, ls=ls, lw=1.5, ms=4.5, capsize=2, label=head)
    ax.set_xscale("log")
    ax.set_xticks([100, 200, 500, 1000, 2000, 5000, 10000])
    ax.set_xticklabels(["100", "200", "500", "1k", "2k", "5k", "10k"])
    ax.minorticks_off()
    ax.set_xlabel("Labelled patches per class (k)")
    ax.set_title("(c) Large support sets", fontweight="bold", loc="left")
    ax.legend(fontsize=7, frameon=False, loc="center right")
    ax.set_ylim(0.55, 1.0)
    save(fig, "Fig12_support_robustness")


if __name__ == "__main__":
    task_shift()
    robustness()
