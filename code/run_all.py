"""Runs the whole study end to end. Each step skips work that is already on disk.

    python run_all.py                 data -> features -> probes -> adaptation -> figures
    python run_all.py --finetune      additionally runs the fine-tuning architecture study
    python run_all.py --models uni2_h h_optimus_1     add newly unlocked (gated) backbones
"""
import argparse
import subprocess
import sys

import config as C


def step(title, args):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}", flush=True)
    code = subprocess.call([sys.executable] + args, cwd=C.CODE_DIR)
    if code != 0:
        raise SystemExit(f"step failed: {' '.join(args)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--finetune", action="store_true")
    ap.add_argument("--epochs", type=int, default=15)
    a = ap.parse_args()
    extra = (["--models"] + a.models) if a.models else []

    if not C.DEV_CACHE_NPY.exists() or not C.DEV_INDEX_CSV.exists():
        step("1a. development image cache", ["prepare_data.py", "--dev"])
    if not (C.CACHE_DIR / "pcam_adapt_x.npy").exists():
        step("1b. PatchCamelyon pools", ["prepare_data.py", "--pcam"])
    step("2. frozen embeddings", ["extract_features.py"] + extra)
    step("3. linear probes (Experiment A)", ["linear_probe.py"] + extra)
    step("4. adaptation ladder and few-shot (Experiment B)", ["adapt_fewshot.py"] + extra)
    if a.finetune:
        for model in C.FINETUNE_BACKBONES:
            for flag in ([], ["--stain_aug"]):
                step(f"5. fine-tune {model} {' '.join(flag)}",
                     ["finetune.py", "--model", model, "--epochs", str(a.epochs)] + flag)
    step("6. figures and summary tables", ["plots.py"])
    step("7. architecture figure", ["make_architecture_figure.py"])


if __name__ == "__main__":
    main()
