"""One-class rule with and without information from the unlabelled (tumour-containing) pool.

The rule reported in the first revision centres embeddings on the mean of the unlabelled
adaptation pool, about half of which is tumour. This script separates what comes from the
k labelled normal patches from what comes from that pool.
"""
import numpy as np
import pandas as pd

import config as C
from revision_cpu import D, auc, unit

REV = C.RESULT_DIR / "revision"
rows = []
for key in C.DEFAULT_BACKBONES:
    d = D(key)
    neg = np.flatnonzero(d.y_adapt == 0)
    for k in (50, 1000):
        for rep in range(20 if k == 50 else 5):
            rng = np.random.default_rng(C.SEED + 7 * k + rep)
            sel = rng.choice(neg, k, replace=False)
            fn = d.f_adapt[sel]
            mu, sd = fn.mean(0), fn.std(0) + 1e-6
            res = {
                # as reported: pool statistics (unlabelled, contains tumour) + cosine to the normal centroid
                "cosine to normal centroid, pool statistics": -(unit(d.zt) @ unit(d.za[sel].mean(0))),
                # normal patches only: standardised Euclidean distance from the normal mean
                "distance from normal mean, normal-only statistics": np.linalg.norm((d.f_test - mu) / sd, axis=1),
                # normal patches only: plain Euclidean distance in the raw embedding space
                "distance from normal mean, raw embeddings": np.linalg.norm(d.f_test - mu, axis=1),
                # no labels at all: distance from the mean of the unlabelled pool
                "distance from pool mean, no labels": np.linalg.norm(d.zt, axis=1),
            }
            for name, s in res.items():
                rows.append(dict(encoder=key, k=k, rule=name, auroc=auc(d.y_test, s)))
t = pd.DataFrame(rows)
s = t.groupby(["encoder", "rule", "k"]).auroc.agg(["mean", "std"]).round(3).reset_index()
s.to_csv(REV / "R2_one_class_variants.csv", index=False)
pd.set_option("display.width", 250)
print(s.pivot_table(index=["rule", "k"], columns="encoder", values="mean")[C.DEFAULT_BACKBONES].to_string())
