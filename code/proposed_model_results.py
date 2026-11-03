"""All results, predictions, tables and figures for the PROPOSED model only:

    frozen pathology foundation encoder (Phikon; Phikon-v2 as variant)
    -> target standardisation -> class prototypes from k labelled patches -> cosine head

Uses the cached embeddings, so it runs in a few minutes on CPU. Few-shot draws use the same
seeds as adapt_fewshot.py, so the numbers agree with tables/adaptation_metrics.csv.

Output folder:  <project>/prposed model resul and prediction/{tables,predictions,figures}
"""
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.preprocessing import StandardScaler

import config as C
from metrics import all_metrics, bootstrap_ci

warnings.filterwarnings("ignore")
OUT = C.ROOT / "prposed model resul and prediction"
TAB, PRED, FIG = OUT / "tables", OUT / "predictions", OUT / "figures"
for d in (TAB, PRED, FIG):
    d.mkdir(parents=True, exist_ok=True)

ENCODERS = {"phikon": "Phikon (ViT-B/16)", "phikon_v2": "Phikon-v2 (ViT-L/16)"}
MAIN = "phikon"
KS = C.FEWSHOT_K
REPS = C.FEWSHOT_REPEATS
TAU = 10.0
SHOW_K = (10, 50, 100)
METRICS = ["accuracy", "sensitivity", "specificity", "precision", "f1", "balanced_accuracy", "mcc",
           "auroc", "auprc", "ece", "brier"]
NICE = dict(accuracy="Accuracy", sensitivity="Sensitivity", specificity="Specificity",
            precision="Precision", f1="F1-score", balanced_accuracy="Balanced accuracy", mcc="MCC",
            auroc="AUROC", auprc="AUPRC", ece="ECE", brier="Brier score")
# Paun et al., Cancers 2026, 18, 2752, external validation, Table 4.
PAPER = dict(accuracy=0.5691, sensitivity=0.9066, specificity=0.3332, precision=0.4871, f1=0.6337,
             balanced_accuracy=0.6199, mcc=0.279)
NORMAL, TUMOUR, GOLD, INK, MUTED, RED = "#2b7bba", "#7b3fa0", "#b7791f", "#1f2933", "#52606d", "#b5533c"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
                     "savefig.facecolor": "white"})


def save(fig, name):
    for ext in ("png", "pdf"):
        try:
            fig.savefig(FIG / f"{name}.{ext}", dpi=400, bbox_inches="tight")
        except OSError:                      # file open in a viewer: keep the old one, write a copy
            fig.savefig(FIG / f"{name}_new.{ext}", dpi=400, bbox_inches="tight")
            print(f"  ({name}.{ext} is open elsewhere; wrote {name}_new.{ext})")
    plt.close(fig)
    print("  figure:", name)


def unit(a):
    return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)


def draw_shots(y_adapt, k, rep):
    """Identical draw to adapt_fewshot.py."""
    rng = np.random.default_rng(C.SEED + 1000 * k + rep)
    pos, neg = np.flatnonzero(y_adapt == 1), np.flatnonzero(y_adapt == 0)
    return np.concatenate([rng.choice(neg, k, replace=False), rng.choice(pos, k, replace=False)])


def prototype_predict(za_shots, y_shots, zt):
    c0, c1 = unit(za_shots[y_shots == 0].mean(0)), unit(za_shots[y_shots == 1].mean(0))
    zt = unit(zt)
    s0, s1 = zt @ c0, zt @ c1
    return s0, s1, 1.0 / (1.0 + np.exp(-TAU * (s1 - s0)))


class Data:
    def __init__(self, key):
        self.key = key
        self.f_adapt = np.load(C.FEATURE_DIR / key / "pcam_adapt.npy")
        self.f_test = np.load(C.FEATURE_DIR / key / "pcam_test.npy")
        self.adapt = pd.read_csv(C.CACHE_DIR / "pcam_adapt_index.csv")
        self.test = pd.read_csv(C.CACHE_DIR / "pcam_test_index.csv")
        self.y_adapt, self.y_test = self.adapt.label.to_numpy(), self.test.label.to_numpy()
        sc = StandardScaler().fit(self.f_adapt)
        self.za, self.zt = sc.transform(self.f_adapt), sc.transform(self.f_test)


# =============================================================================== 1. main runs
def run_all_draws(data):
    rows, store = [], {}
    for k in KS:
        for rep in range(REPS):
            sel = draw_shots(data.y_adapt, k, rep)
            s0, s1, p = prototype_predict(data.za[sel], data.y_adapt[sel], data.zt)
            m = all_metrics(data.y_test, p, 0.5)
            rows.append(dict(encoder=data.key, k=k, draw=rep, **{x: m[x] for x in METRICS},
                             tp=m["tp"], tn=m["tn"], fp=m["fp"], fn=m["fn"]))
            store[(k, rep)] = (s0, s1, p)
    return pd.DataFrame(rows), store


def representative(draws, k):
    """The draw whose AUROC is the median of the repeated draws (neither lucky nor unlucky)."""
    d = draws[draws.k == k].sort_values("auroc").reset_index(drop=True)
    return int(d.loc[len(d) // 2, "draw"])


def reference_points(data):
    """Direct transfer (oral-trained head) and the all-label upper bound, same encoder."""
    # Read back the predictions written by linear_probe.py / adapt_fewshot.py so that these
    # reference rows are exactly the ones reported in the study-wide tables.
    direct = pd.read_csv(C.PRED_DIR / "linear_probe" / f"{data.key}__native__external.csv")
    oracle = pd.read_csv(C.PRED_DIR / "adaptation" / f"{data.key}__oracle_target_supervised.csv")
    assert (direct.pcam_index.to_numpy() == data.test.pcam_index.to_numpy()).all()
    assert (oracle.pcam_index.to_numpy() == data.test.pcam_index.to_numpy()).all()
    return direct.prob.to_numpy(), oracle.prob.to_numpy()


# =============================================================================== 2. ablations
def ablations(data):
    index = pd.read_csv(C.DEV_INDEX_CSV)
    f_dev = np.load(C.FEATURE_DIR / data.key / "dev.npy")
    src = StandardScaler().fit(f_dev[index[index.native_split == "train"].idx.to_numpy()])
    spaces = {"none (raw embeddings)": (data.f_adapt, data.f_test),
              "source statistics (oral images)": (src.transform(data.f_adapt), src.transform(data.f_test)),
              "target statistics": (data.za, data.zt)}
    rows = []
    for k in (5, 10, 50, 100):
        for rep in range(REPS):
            sel = draw_shots(data.y_adapt, k, rep)
            ys = data.y_adapt[sel]
            for name, (xa, xt) in spaces.items():
                _, _, p = prototype_predict(xa[sel], ys, xt)
                rows.append(dict(component="standardisation", variant=name, k=k, draw=rep,
                                 **{m: all_metrics(data.y_test, p)[m] for m in METRICS}))
            xa, xt = data.za, data.zt
            m0, m1 = xa[sel][ys == 0].mean(0), xa[sel][ys == 1].mean(0)
            d = np.linalg.norm(xt - m0, axis=1) - np.linalg.norm(xt - m1, axis=1)
            heads = {"cosine nearest centroid": prototype_predict(xa[sel], ys, xt)[2],
                     "Euclidean nearest centroid": 1 / (1 + np.exp(-d / (np.std(d) + 1e-8))),
                     "logistic regression on the shots":
                         LogisticRegression(C=0.01, max_iter=500).fit(xa[sel], ys).predict_proba(xt)[:, 1]}
            for name, p in heads.items():
                rows.append(dict(component="classifier head", variant=name, k=k, draw=rep,
                                 **{m: all_metrics(data.y_test, p)[m] for m in METRICS}))
    return pd.DataFrame(rows)


# =============================================================================== tables
def mean_sd_table(draws):
    g = draws.groupby(["encoder", "k"])[METRICS].agg(["mean", "std"])
    out = pd.DataFrame({"Encoder": [ENCODERS[e] for e, _ in g.index], "k per class": [k for _, k in g.index],
                        "Labelled patches (total)": [2 * k for _, k in g.index]})
    for m in METRICS:
        out[NICE[m]] = [f"{a:.3f} ± {b:.3f}" for a, b in zip(g[(m, "mean")], g[(m, "std")])]
    return out


def fmt_ci(row, m):
    return f"{row[m]:.3f} ({row[m + '_lo']:.3f}–{row[m + '_hi']:.3f})"


# =============================================================================== figures
def fig_performance_vs_k(draws, refs):
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), sharex=True)
    for ax, m in zip(axes, ("auroc", "balanced_accuracy", "sensitivity", "specificity")):
        for key, col, mk in (("phikon", TUMOUR, "o"), ("phikon_v2", "#3d5a99", "s")):
            g = draws[draws.encoder == key].groupby("k")[m]
            mu, sd = g.mean(), g.std()
            ax.plot(mu.index, mu.values, color=col, marker=mk, ms=4, lw=1.6, label=ENCODERS[key])
            ax.fill_between(mu.index, mu - sd, mu + sd, color=col, alpha=0.14, linewidth=0)
        d, o = refs[MAIN]["direct"][m], refs[MAIN]["oracle"][m]
        ax.axhline(o, color=MUTED, lw=0.9, ls=":")
        ax.axhline(d, color=RED, lw=0.9, ls=":")
        ax.text(1.02, o + 0.012, f"all labels {o:.2f}", fontsize=6.6, color=MUTED)
        ax.text(1.02, d + 0.012, f"direct transfer {d:.2f}", fontsize=6.6, color=RED)
        ax.set_xscale("log")
        ax.set_xticks(list(KS))
        ax.set_xticklabels([str(k) for k in KS])
        ax.minorticks_off()
        ax.set_ylim(0.25, 1.03)
        ax.set_title(NICE[m], fontweight="bold")
        ax.set_xlabel("Labelled patches per class (k)")
    axes[0].set_ylabel("External test (lymph node)")
    axes[0].legend(fontsize=7, frameon=False, loc="lower right", bbox_to_anchor=(1, 0.12))
    save(fig, "Fig1_performance_vs_k")


def fig_roc_pr(y, curves):
    """Journal-style ROC and precision-recall panels.

    Serif type, boxed square axes, inward ticks, panel letters, and line styles that stay
    distinguishable in greyscale. Legends sit below the axes so that no curve is covered.
    """
    styles = [("#a8322d", (0, (5, 2.5))),            # direct transfer
              ("#6f9bd1", (0, (4, 1.5, 1, 1.5))),    # k = 10
              ("#1f4e9c", "-"),                      # k = 50
              ("#0b1f47", (0, (9, 2))),              # k = 100
              ("#555555", (0, (1, 1.6)))]            # upper bound
    rc_formal = {"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                 "mathtext.fontset": "stix", "font.size": 10, "axes.linewidth": 0.8,
                 "axes.spines.top": True, "axes.spines.right": True, "axes.grid": False,
                 "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True,
                 "ytick.right": True, "xtick.minor.visible": True, "ytick.minor.visible": True,
                 "xtick.major.size": 4, "ytick.major.size": 4, "xtick.minor.size": 2,
                 "ytick.minor.size": 2, "legend.fontsize": 8.5}
    with plt.rc_context(rc_formal):
        fig, axes = plt.subplots(1, 2, figsize=(7.4, 4.7))
        for (name, (p, _, _)), (col, ls) in zip(curves.items(), styles):
            fpr, tpr, _ = roc_curve(y, p)
            pr, rc, _ = precision_recall_curve(y, p)
            m = all_metrics(y, p)
            axes[0].plot(fpr, tpr, color=col, ls=ls, lw=1.4, label=f"{name}: AUROC = {m['auroc']:.3f}")
            axes[1].plot(rc[:-1], pr[:-1], color=col, ls=ls, lw=1.4,
                         label=f"{name}: AUPRC = {m['auprc']:.3f}")
        axes[0].plot([0, 1], [0, 1], color="black", lw=0.6, ls=(0, (2, 3)))
        axes[0].text(0.60, 0.555, "chance", rotation=45, fontsize=8, ha="center", va="center",
                     color="black", rotation_mode="anchor")
        axes[1].axhline(y.mean(), color="black", lw=0.6, ls=(0, (2, 3)))
        axes[1].text(0.985, y.mean() + 0.018, f"prevalence = {y.mean():.3f}", fontsize=8, ha="right")
        axes[0].set(xlabel="1 − Specificity (false-positive rate)", ylabel="Sensitivity (true-positive rate)")
        axes[1].set(xlabel="Recall (sensitivity)", ylabel="Precision (positive predictive value)")
        for ax, letter in zip(axes, "ab"):
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.02)
            ax.set_xticks(np.linspace(0, 1, 6))
            ax.set_yticks(np.linspace(0, 1, 6))
            ax.set_aspect("equal", adjustable="box")
            ax.text(-0.17, 1.03, f"({letter})", transform=ax.transAxes, fontsize=12, fontweight="bold")
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.19), frameon=True, fancybox=False,
                      edgecolor="black", framealpha=1, borderpad=0.6, handlelength=3.2).get_frame().set_linewidth(0.6)
        fig.subplots_adjust(wspace=0.34)
        for ext in ("png", "pdf", "tiff"):
            fig.savefig(FIG / f"Fig2_ROC_PR_curves.{ext}", dpi=600, bbox_inches="tight",
                        **({"pil_kwargs": {"compression": "tiff_lzw"}} if ext == "tiff" else {}))
        plt.close(fig)
    print("  figure: Fig2_ROC_PR_curves")


def fig_confusion(y, preds):
    fig, axes = plt.subplots(1, len(preds), figsize=(3.1 * len(preds), 3.3))
    for ax, (name, p) in zip(axes, preds.items()):
        m = all_metrics(y, p)
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]], float)
        pct = cm / cm.sum(1, keepdims=True)
        ax.imshow(pct, cmap="Purples", vmin=0, vmax=1)
        ax.grid(False)
        for (a, b), v in np.ndenumerate(cm):
            ax.text(b, a, f"{int(v):,}\n{pct[a, b]:.1%}", ha="center", va="center", fontsize=9,
                    color="white" if pct[a, b] > 0.55 else INK)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Normal", "Metastasis"])
        ax.set_yticklabels(["Normal", "Metastasis"], rotation=90, va="center")
        ax.set_title(f"{name}\nacc. {m['accuracy']:.3f}", fontsize=9, fontweight="bold")
        for s in ax.spines.values():
            s.set_visible(False)
    fig.supxlabel("Predicted", fontsize=9.5)
    fig.supylabel("Actual", fontsize=9.5)
    save(fig, "Fig3_confusion_matrices")


def fig_probability_histograms(y, preds):
    fig, axes = plt.subplots(1, len(preds), figsize=(3.6 * len(preds), 3.2), sharey=True)
    bins = np.linspace(0, 1, 41)
    for ax, (name, p) in zip(axes, preds.items()):
        ax.hist(p[y == 0], bins, color=NORMAL, alpha=0.65, label="Normal")
        ax.hist(p[y == 1], bins, color=TUMOUR, alpha=0.65, label="Metastasis")
        ax.axvline(0.5, color=INK, lw=0.9, ls="--")
        ax.set_title(name, fontsize=9, fontweight="bold")
        ax.set_xlabel("Predicted probability of metastasis")
    axes[0].set_ylabel("Patches")
    axes[0].legend(fontsize=7.5, frameon=False)
    save(fig, "Fig4_probability_distributions")


def fig_calibration(y, preds):
    fig, ax = plt.subplots(figsize=(4.8, 4.5))
    cols = [RED, "#c08bd6", TUMOUR, "#3d1f5c"]
    for (name, p), col in zip(preds.items(), cols):
        b = np.minimum((p * 10).astype(int), 9)
        g = pd.DataFrame(dict(b=b, p=p, y=y)).groupby("b").agg(conf=("p", "mean"), acc=("y", "mean"), n=("y", "size"))
        g = g[g.n >= 50]
        ax.plot(g.conf, g.acc, marker="o", ms=4, lw=1.5, color=col,
                label=f"{name}  (ECE {all_metrics(y, p)['ece']:.3f})")
    ax.plot([0, 1], [0, 1], color="#9aa5b1", lw=0.8, ls=":")
    ax.set(xlabel="Predicted probability of metastasis", ylabel="Observed fraction metastatic")
    ax.set_title("Calibration (reliability diagram)", fontweight="bold")
    ax.legend(fontsize=7, frameon=False)
    save(fig, "Fig5_calibration")


def fig_gallery(data, p, k):
    x = np.load(C.CACHE_DIR / "pcam_test_x.npy", mmap_mode="r")
    y, pred = data.y_test, p >= 0.5
    groups = [("True positive", (y == 1) & pred, TUMOUR), ("True negative", (y == 0) & ~pred, NORMAL),
              ("False positive", (y == 0) & pred, RED), ("False negative", (y == 1) & ~pred, RED)]
    rng = np.random.default_rng(C.SEED)
    fig, axes = plt.subplots(4, 8, figsize=(13.2, 7.4))
    for r, (name, mask, col) in enumerate(groups):
        idx = np.flatnonzero(mask)
        idx = np.array([i for i in rng.permutation(idx)[:400] if x[i].mean() < 190 and x[i].std() > 30][:8])
        for c in range(8):
            ax = axes[r, c]
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            if c < len(idx):
                ax.imshow(np.asarray(x[idx[c]]), interpolation="lanczos")
                ax.set_xlabel(f"p = {p[idx[c]]:.2f}", fontsize=8, color=col, fontweight="bold", labelpad=2)
            for s in ax.spines.values():
                s.set_visible(True)
                s.set_color(col)
                s.set_linewidth(2.2)
        axes[r, 0].set_ylabel(f"{name}\n(n = {int(mask.sum()):,})", fontsize=9, fontweight="bold", color=col)
    fig.suptitle(f"Nearest-centroid predictions on unseen slides (Phikon, k = {k} per class)",
                 fontsize=10.5, fontweight="bold", y=0.93)
    save(fig, "Fig6_prediction_gallery")


def fig_vs_paper(rows):
    ms = ["accuracy", "sensitivity", "specificity", "precision", "f1", "balanced_accuracy", "mcc"]
    fig, ax = plt.subplots(figsize=(11, 4))
    names = list(rows)
    cols = ["#9aa5b1", RED, "#c08bd6", TUMOUR, "#3d1f5c"][:len(names)]
    w = 0.8 / len(names)
    for i, (name, col) in enumerate(zip(names, cols)):
        vals = [rows[name][m] for m in ms]
        xs = np.arange(len(ms)) + (i - (len(names) - 1) / 2) * w
        ax.bar(xs, vals, w * 0.92, color=col, label=name)
        for xx, v in zip(xs, vals):
            ax.text(xx, max(v, 0) + 0.012, f"{v:.2f}", ha="center", fontsize=6.2, rotation=90)
    ax.axhline(0, color=INK, lw=0.6)
    ax.set_xticks(np.arange(len(ms)))
    ax.set_xticklabels([NICE[m] for m in ms])
    ax.set_ylim(-0.2, 1.12)
    ax.set_ylabel("External test (lymph node)")
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=7.5, frameon=False, ncol=len(names), loc="upper center", bbox_to_anchor=(0.5, 1.13))
    save(fig, "Fig7_comparison_with_reference_paper")


def fig_embedding(data, sel, p):
    import umap
    rng = np.random.default_rng(C.SEED)
    idx = rng.choice(len(data.zt), 3000, replace=False)
    ys = data.y_adapt[sel]
    c0, c1 = unit(data.za[sel][ys == 0].mean(0)), unit(data.za[sel][ys == 1].mean(0))
    emb = umap.UMAP(n_neighbors=30, min_dist=0.25, metric="cosine", random_state=C.SEED).fit_transform(
        np.vstack([unit(data.zt[idx]), c0, c1]))
    z, y, pp = emb[:-2], data.y_test[idx], p[idx]
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.3))
    for lab, col, name in ((0, NORMAL, "Normal"), (1, TUMOUR, "Metastasis")):
        axes[0].scatter(z[y == lab, 0], z[y == lab, 1], s=3, color=col, alpha=0.5, linewidths=0, label=name)
    sc = axes[1].scatter(z[:, 0], z[:, 1], s=3, c=pp, cmap="RdBu_r", vmin=0, vmax=1, linewidths=0)
    ok = (pp >= 0.5) == (y == 1)
    axes[2].scatter(z[ok, 0], z[ok, 1], s=3, color="#b8c2cc", alpha=0.6, linewidths=0, label="Correct")
    axes[2].scatter(z[~ok, 0], z[~ok, 1], s=5, color=RED, alpha=0.8, linewidths=0, label="Misclassified")
    for ax, title in zip(axes, ("Reference label", "Predicted probability", "Errors")):
        ax.scatter(*emb[-2], marker="*", s=260, color=NORMAL, edgecolor="white", linewidth=1.2, zorder=5)
        ax.scatter(*emb[-1], marker="*", s=260, color=TUMOUR, edgecolor="white", linewidth=1.2, zorder=5)
        ax.set_title(title, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    axes[0].legend(fontsize=7.5, frameon=False, markerscale=4)
    axes[2].legend(fontsize=7.5, frameon=False, markerscale=3)
    fig.colorbar(sc, ax=axes[1], fraction=0.045, pad=0.02, label="p (metastasis)")
    save(fig, "Fig8_embedding_space_predictions")


def fig_ablation(abl, enc_tab):
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.9), sharey=True)
    for ax, comp in zip(axes[:2], ("standardisation", "classifier head")):
        g = abl[abl.component == comp].groupby(["variant", "k"]).auroc.agg(["mean", "std"]).reset_index()
        for (name, d), col in zip(g.groupby("variant", sort=False), ("#9aa5b1", GOLD, "#2f7d68", "#3d5a99")):
            main = name in ("target statistics", "cosine nearest centroid")
            euclid = "Euclidean" in name                 # coincides with the cosine curve: draw on top, dashed
            ax.errorbar(d.k, d["mean"], yerr=None if euclid else d["std"], marker="x" if euclid else "o",
                        ms=7 if euclid else 4, lw=1.6 if main else 1.1, capsize=2,
                        color="black" if euclid else col, ls=(0, (4, 3)) if euclid else "-",
                        zorder=5 if euclid else 2, label=name)
        ax.set_xscale("log")
        ax.set_xticks([5, 10, 50, 100])
        ax.set_xticklabels(["5", "10", "50", "100"])
        ax.minorticks_off()
        ax.set_title(comp.capitalize(), fontweight="bold")
        ax.set_xlabel("Labelled patches per class (k)")
        ax.legend(fontsize=7, frameon=False, loc="lower right")
    ax = axes[2]
    palette = {"Pathology foundation model": TUMOUR, "General SSL ViT": "#2f7d68",
               "Hierarchical transformer (ImageNet)": "#3d5a99", "State-space (ImageNet)": "#6b4a9c",
               "CNN (ImageNet)": "#8c6d31"}
    for (bk, d) in enc_tab.groupby("backbone", sort=False):
        fam = C.BACKBONES[bk]["family"]
        ax.plot(d.k, d.auroc, marker="o", ms=3.5, lw=1.8 if "foundation" in fam else 1.0, color=palette[fam],
                alpha=1.0 if "foundation" in fam else 0.75)
    for fam, col in palette.items():
        ax.plot([], [], color=col, lw=1.6, label=fam)
    ax.set_xscale("log")
    ax.set_xticks(list(KS))
    ax.set_xticklabels([str(k) for k in KS])
    ax.minorticks_off()
    ax.set_title("Encoder (same nearest-centroid head)", fontweight="bold")
    ax.set_xlabel("Labelled patches per class (k)")
    ax.legend(fontsize=6.8, frameon=False, loc="lower right")
    axes[0].set_ylabel("External AUROC")
    save(fig, "Fig9_ablation_studies")


def fig_per_slide(slide_tab):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    axes[0].hist(slide_tab.accuracy, bins=np.linspace(0, 1, 21), color=TUMOUR, alpha=0.8, edgecolor="white")
    axes[0].axvline(slide_tab.accuracy.median(), color=INK, lw=1, ls="--")
    axes[0].text(slide_tab.accuracy.median() - 0.02, axes[0].get_ylim()[1] * 0.92,
                 f"median {slide_tab.accuracy.median():.2f}", ha="right", fontsize=8)
    axes[0].set(xlabel="Accuracy within a slide", ylabel="Slides")
    axes[0].set_title("Per-slide accuracy", fontweight="bold")
    axes[1].scatter(slide_tab.n_patches, slide_tab.accuracy, s=22 + 60 * slide_tab.frac_metastatic,
                    c=slide_tab.frac_metastatic, cmap="RdBu_r", vmin=0, vmax=1, edgecolor="white", linewidth=0.5)
    axes[1].set(xlabel="Patches from the slide in the evaluation pool", ylabel="Accuracy within the slide",
                xscale="log")
    axes[1].set_title("Slide size, composition and accuracy", fontweight="bold")
    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(0, 1))
    fig.colorbar(sm, ax=axes[1], fraction=0.045, pad=0.02, label="fraction metastatic")
    save(fig, "Fig10_per_slide_performance")


# =============================================================================== main
def main():
    data = {k: Data(k) for k in ENCODERS}
    draws, stores, refs, ref_probs = [], {}, {}, {}
    for key, d in data.items():
        df, store = run_all_draws(d)
        draws.append(df)
        stores[key] = store
        p_direct, p_oracle = reference_points(d)
        ref_probs[key] = dict(direct=p_direct, oracle=p_oracle)
        refs[key] = dict(direct=all_metrics(d.y_test, p_direct), oracle=all_metrics(d.y_test, p_oracle))
    draws = pd.concat(draws, ignore_index=True)
    draws.to_csv(TAB / "TableS1_all_few_shot_draws.csv", index=False)

    # ---- Table 1: performance vs k (mean ± SD over draws)
    t1 = mean_sd_table(draws)
    t1.to_csv(TAB / "Table1_performance_vs_k_mean_sd.csv", index=False, encoding="utf-8-sig")

    # ---- representative runs: predictions + Table 2 with slide-cluster bootstrap intervals
    y = data[MAIN].y_test
    t2, rep_probs = [], {}
    for key, d in data.items():
        for k in SHOW_K:
            rep = representative(draws[draws.encoder == key], k)
            s0, s1, p = stores[key][(k, rep)]
            rep_probs[(key, k)] = p
            sel = draw_shots(d.y_adapt, k, rep)
            pd.DataFrame(dict(pcam_index=d.test.pcam_index, slide=d.test.wsi, label=d.y_test,
                              sim_normal=s0.round(5), sim_metastasis=s1.round(5), probability=p,
                              prediction=(p >= 0.5).astype(int),
                              correct=((p >= 0.5).astype(int) == d.y_test).astype(int))
                         ).to_csv(PRED / f"predictions_{key}_k{k}.csv", index=False)
            pd.DataFrame(dict(pcam_index=d.adapt.pcam_index.to_numpy()[sel], slide=d.adapt.wsi.to_numpy()[sel],
                              label=d.y_adapt[sel])).to_csv(PRED / f"support_set_{key}_k{k}.csv", index=False)
            m = all_metrics(d.y_test, p)
            m.update(bootstrap_ci(d.y_test, p, 0.5, groups=d.test.wsi.to_numpy(), n_boot=1000))
            row = {"Encoder": ENCODERS[key], "k per class": k, "Draw": rep,
                   "TP": m["tp"], "TN": m["tn"], "FP": m["fp"], "FN": m["fn"]}
            row.update({NICE[x]: fmt_ci(m, x) for x in METRICS})
            t2.append(row)
    pd.DataFrame(t2).to_csv(TAB / "Table2_representative_run_95CI.csv", index=False, encoding="utf-8-sig")
    for key in data:
        for name, p in ref_probs[key].items():
            pd.DataFrame(dict(pcam_index=data[key].test.pcam_index, slide=data[key].test.wsi,
                              label=data[key].y_test, probability=p)
                         ).to_csv(PRED / f"reference_{name}_{key}.csv", index=False)

    # ---- Table 3: comparison with the earlier study of Paun et al. (2026) and reference points
    mean = draws[draws.encoder == MAIN].groupby("k")[METRICS].mean()
    cmp_rows = {"Paun et al. (EfficientNet-B0, direct)": PAPER,
                "Direct transfer, same encoder (k = 0)": refs[MAIN]["direct"]}
    for k in SHOW_K:
        cmp_rows[f"Nearest centroid, k = {k}"] = mean.loc[k].to_dict()
    t3 = pd.DataFrame([{"Model": n, **{NICE[m]: (round(v[m], 3) if m in v else "not reported") for m in METRICS}}
                       for n, v in {**cmp_rows, "All 20,000 labels (reference)": refs[MAIN]["oracle"]}.items()])
    t3.to_csv(TAB / "Table3_comparison_with_reference_paper.csv", index=False)

    # ---- Table 4: ablations
    abl = ablations(data[MAIN])
    t4 = abl.groupby(["component", "variant", "k"])[["auroc", "balanced_accuracy", "sensitivity", "specificity"]] \
        .agg(["mean", "std"]).round(3)
    t4.columns = [f"{NICE[m]} {s}" for m, s in t4.columns]
    t4.reset_index().to_csv(TAB / "Table4_ablation_standardisation_and_head.csv", index=False)
    ad = pd.read_csv(C.TABLE_DIR / "adaptation_metrics.csv")
    enc_tab = ad[ad.method == "fewshot_prototype"].groupby(["backbone", "family", "k"])[
        ["auroc", "balanced_accuracy"]].mean().round(3).reset_index()
    enc_tab.to_csv(TAB / "Table5_ablation_encoder.csv", index=False)

    # ---- Table 6: per-slide performance of the main representative run
    k_main = 50
    p_main = rep_probs[(MAIN, k_main)]
    t = data[MAIN].test.assign(p=p_main, ok=((p_main >= 0.5).astype(int) == y))
    slide_tab = t.groupby("wsi").agg(n_patches=("label", "size"), frac_metastatic=("label", "mean"),
                                     accuracy=("ok", "mean"), mean_probability=("p", "mean")).round(4).reset_index()
    slide_tab.rename(columns={"wsi": "slide"}).to_csv(TAB / "Table6_per_slide_performance_k50.csv", index=False)

    # ---- figures
    fig_performance_vs_k(draws, refs)
    curves = {"Direct transfer ($k$ = 0)": (ref_probs[MAIN]["direct"], RED, "--")}
    for k, col in zip(SHOW_K, ("#c08bd6", TUMOUR, "#3d1f5c")):
        curves[f"Nearest centroid, $k$ = {k}"] = (rep_probs[(MAIN, k)], col, "-")
    curves["All labels (reference)"] = (ref_probs[MAIN]["oracle"], MUTED, ":")
    fig_roc_pr(y, curves)
    preds = {"Direct transfer (k = 0)": ref_probs[MAIN]["direct"]}
    preds.update({f"Nearest centroid, k = {k}": rep_probs[(MAIN, k)] for k in SHOW_K})
    fig_confusion(y, preds)
    fig_probability_histograms(y, preds)
    fig_calibration(y, preds)
    fig_gallery(data[MAIN], p_main, k_main)
    fig_vs_paper(cmp_rows)
    rep = representative(draws[draws.encoder == MAIN], k_main)
    fig_embedding(data[MAIN], draw_shots(data[MAIN].y_adapt, k_main, rep), p_main)
    fig_ablation(abl, enc_tab)
    fig_per_slide(slide_tab)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(t1[["Encoder", "k per class", "Accuracy", "Sensitivity", "Specificity", "MCC", "AUROC"]].to_string(index=False))
    print(pd.DataFrame(t2)[["Encoder", "k per class", "Draw", "TP", "TN", "FP", "FN", "Accuracy", "AUROC"]].to_string(index=False))
    print(t3.to_string(index=False))
    print(t4.reset_index()[["component", "variant", "k", "AUROC mean", "AUROC std"]].to_string(index=False))
    print("per-slide accuracy: median", slide_tab.accuracy.median(), "| slides", len(slide_tab),
          "| slides below 0.5:", int((slide_tab.accuracy < 0.5).sum()))


if __name__ == "__main__":
    main()
