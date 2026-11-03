"""Experiment C driver: fine-tunes each architecture with and without stain augmentation.

Runs already present in tables/finetune_metrics.csv are skipped, so the script can be
restarted after an interruption.

    python run_finetune_study.py --epochs 6
"""
import argparse
import subprocess
import sys

import pandas as pd

import config as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=C.FINETUNE_BACKBONES)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--seed", type=int, default=C.SEED)
    a = ap.parse_args()

    table = C.TABLE_DIR / "finetune_metrics.csv"
    for model in a.models:
        for stain in (False, True):
            run = f"{model}__{'stainaug' if stain else 'noaug'}__seed{a.seed}"
            if table.exists() and run in set(pd.read_csv(table).run):
                print(f"[skip] {run}", flush=True)
                continue
            print(f"\n===== {run} =====", flush=True)
            cmd = [sys.executable, "finetune.py", "--model", model, "--epochs", str(a.epochs),
                   "--patience", str(a.patience), "--seed", str(a.seed)]
            if stain:
                cmd.append("--stain_aug")
            if subprocess.call(cmd, cwd=C.CODE_DIR) != 0:
                print(f"[failed] {run}", flush=True)


if __name__ == "__main__":
    main()
