"""Step 2 - frozen embeddings for every backbone on every image set.

Writes  dataset/cache/features/<backbone>/{dev,pcam_test,pcam_adapt}.npy  (float32 [N, D]).
Backbones that cannot be loaded (gated weights, missing package) are skipped and listed.

Usage:  python extract_features.py                       (default, non-gated backbones)
        python extract_features.py --models uni2_h h_optimus_1
        python extract_features.py --all                 (everything in the registry)
"""
import argparse
import json
import time

import numpy as np
import torch
from tqdm import tqdm

import config as C
from models import BackboneUnavailable, load_backbone, preprocess

SETS = {"dev": C.DEV_CACHE_NPY,
        "pcam_test": C.CACHE_DIR / "pcam_test_x.npy",
        "pcam_adapt": C.CACHE_DIR / "pcam_adapt_x.npy"}


@torch.no_grad()
def embed_array(backbone, arr, batch_size, device, desc):
    feats = np.empty((len(arr), backbone.embed_dim), dtype=np.float32)
    for start in tqdm(range(0, len(arr), batch_size), desc=desc, leave=False):
        x = torch.from_numpy(np.ascontiguousarray(arr[start:start + batch_size]))
        x = preprocess(x, backbone)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
            f = backbone(x)
        feats[start:start + len(x)] = f.float().cpu().numpy()
    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    keys = list(C.BACKBONES) if a.all else (a.models or C.DEFAULT_BACKBONES)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True
    skipped = {}

    for key in keys:
        out_dir = C.FEATURE_DIR / key
        todo = [s for s in SETS if a.overwrite or not (out_dir / f"{s}.npy").exists()]
        if not todo:
            print(f"[{key}] already extracted")
            continue
        try:
            backbone = load_backbone(key, device).eval()
        except BackboneUnavailable as exc:
            print(f"[{key}] SKIPPED - {exc}")
            skipped[key] = str(exc)
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        n_params = sum(p.numel() for p in backbone.parameters()) / 1e6
        info = dict(backbone=key, source=C.BACKBONES[key]["name"], embed_dim=backbone.embed_dim,
                    img_size=backbone.img_size, params_million=round(n_params, 2),
                    family=C.BACKBONES[key]["family"])
        print(f"[{key}] dim={backbone.embed_dim} input={backbone.img_size} params={n_params:.1f}M")
        for s in todo:
            arr = np.load(SETS[s], mmap_mode="r")
            t0 = time.time()
            feats = embed_array(backbone, arr, a.batch_size, device, f"{key}/{s}")
            np.save(out_dir / f"{s}.npy", feats)
            info[f"{s}_images_per_s"] = round(len(arr) / (time.time() - t0), 1)
        (out_dir / "info.json").write_text(json.dumps(info, indent=2))
        del backbone
        torch.cuda.empty_cache()

    if skipped:
        (C.LOG_DIR / "skipped_backbones.json").write_text(json.dumps(skipped, indent=2))
        print("\nSkipped backbones:", ", ".join(skipped))


if __name__ == "__main__":
    main()
