"""Second review round, Reviewer 2: a newer, openly licensed pathology foundation model (Midnight-12k, kaiko.ai 2025).

Runs, for the new encoder, every step that the manuscript reports for the other encoders, with identical code:
  1. the target-information ladder of code/adapt_fewshot.py (direct transfer, target standardisation, CORAL,
     self-training, prototype few-shot k = 1..100 over 20 draws, reference)  -> rows added to adaptation_metrics.csv
  2. logistic regression on the same support sets (as Supplementary Table S11)
  3. normal-only labels: centroid rule with the unlabelled pool, and the one-class detector (as Figure 4a)
  4. paired slide-bootstrap differences Midnight - Phikon at k = 10 and 50 (prototypes and logistic regression)
Results go to Majore Revision/analysis/.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.argv = [sys.argv[0]]
import adapt_fewshot  # noqa: E402
import config as C  # noqa: E402
from revision_cpu import D, auc, draw_shots, proto_score, unit  # noqa: E402
from v2_analyses import lr_score, slide_boot_index  # noqa: E402

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parents[2] / "results" / "revision_analysis"


def paired(y, boots, sa, sb):
    point = np.mean([auc(y, a) - auc(y, b) for a, b in zip(sa, sb)])
    diffs = np.array([np.mean([auc(y[i], a[i]) - auc(y[i], b[i]) for a, b in zip(sa, sb)]) for i in boots])
    return point, np.quantile(diffs, .025), np.quantile(diffs, .975), max(2 * min((diffs <= 0).mean(), (diffs >= 0).mean()), 1 / len(boots))


def main():
    # 1. ladder (adds/replaces the midnight rows of adaptation_metrics.csv; other encoders untouched)
    sys.argv = [sys.argv[0], "--models", "midnight"]
    adapt_fewshot.main()
    ad = pd.read_csv(C.TABLE_DIR / "adaptation_metrics.csv")
    m = ad[ad.backbone == "midnight"].groupby(["method", "k"]).auroc.agg(["mean", "std"]).round(3)
    m.to_csv(OUT / "R2_1_midnight_ladder.csv")
    print(m.to_string())

    dm, dp = D("midnight"), D("phikon")
    y = dm.y_test
    # internal (oral test fold) performance of the direct-transfer probe, as in Table 3
    from metrics import all_metrics
    sc, clf, c = dm.source_probe()
    mi = all_metrics(dm.y_dev[dm.te], clf.predict_proba(sc.transform(dm.f_dev[dm.te]))[:, 1], 0.5)
    me = all_metrics(y, clf.predict_proba(sc.transform(dm.f_test))[:, 1], 0.5)
    pd.DataFrame([dict(encoder="midnight", C=c, internal_accuracy=mi["accuracy"], internal_auroc=mi["auroc"],
                       external_accuracy=me["accuracy"], sensitivity=me["sensitivity"], specificity=me["specificity"],
                       positive_rate=me["positive_rate"], external_auroc=me["auroc"])]).round(4).to_csv(
        OUT / "R2_1_midnight_internal.csv", index=False)
    # 2. logistic regression few-shot (C = 0.01, same draws)
    rows = []
    for key, d in (("midnight", dm), ("phikon", dp)):
        for k in (1, 2, 5, 10, 20, 50, 100):
            for rep in range(20):
                sel = draw_shots(d.y_adapt, k, rep)
                rows.append(dict(encoder=key, k=k, rep=rep,
                                 prototypes=auc(y, proto_score(d.za[sel], d.y_adapt[sel], d.zt)),
                                 logistic_regression=auc(y, lr_score(d.za[sel], d.y_adapt[sel], d.zt))))
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "R2_1_midnight_vs_phikon_fewshot_all.csv", index=False)
    s = r.groupby(["encoder", "k"])[["prototypes", "logistic_regression"]].agg(["mean", "std"]).round(3)
    s.to_csv(OUT / "R2_1_midnight_vs_phikon_fewshot_summary.csv")
    print(s.to_string())

    # 3. normal-only labels (same draws as the manuscript: seed SEED + 7k + rep)
    rows = []
    neg = np.flatnonzero(dm.y_adapt == 0)
    for k in (10, 50, 100, 1000):
        for rep in range(20 if k <= 100 else 5):
            sel = np.random.default_rng(C.SEED + 7 * k + rep).choice(neg, k, replace=False)
            rows.append(dict(encoder="midnight", k=k, rep=rep, method="normal-only centroid with unlabelled pool",
                             auroc=auc(y, -(unit(dm.zt) @ unit(dm.za[sel].mean(0))))))
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler().fit(dm.f_adapt[sel])
            rows.append(dict(encoder="midnight", k=k, rep=rep, method="one-class, normals only",
                             auroc=auc(y, np.linalg.norm(sc.transform(dm.f_test), axis=1))))
    n = pd.DataFrame(rows)
    n.to_csv(OUT / "R2_1_midnight_normal_only_all.csv", index=False)
    print(n.groupby(["method", "k"]).auroc.agg(["mean", "std"]).round(3).to_string())

    # 4. paired differences Midnight - Phikon
    boots = slide_boot_index(dm.test.wsi.to_numpy(), 500)
    tests = []
    for k in (10, 50):
        sels = [draw_shots(dm.y_adapt, k, r) for r in range(10)]
        for head, fn in (("prototypes", proto_score), ("logistic regression", lr_score)):
            sa = [fn(dm.za[s], dm.y_adapt[s], dm.zt) for s in sels]
            sb = [fn(dp.za[s], dp.y_adapt[s], dp.zt) for s in sels]
            pt, lo, hi, p = paired(y, boots, sa, sb)
            tests.append(dict(k=k, head=head, comparison="midnight - phikon", auroc_difference=pt, ci_lo=lo, ci_hi=hi,
                              p_value=p))
            print(tests[-1], flush=True)
    pd.DataFrame(tests).round(4).to_csv(OUT / "R2_1_midnight_minus_phikon_paired.csv", index=False)


if __name__ == "__main__":
    main()
