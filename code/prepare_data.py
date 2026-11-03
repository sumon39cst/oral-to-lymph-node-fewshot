"""Step 1 - build the image caches used by every later experiment.

Development domain (three oral / oropharyngeal datasets, read straight from the zips):
    dev_index.csv   one row per image: dataset, label, subtype, native split, source path
    dev_256.npy     uint8 [N, 256, 256, 3]; short side resized to 256, centre crop

External domain (PatchCamelyon lymph node patches):
    pcam_test_*.npy   20,000 patches for evaluation
    pcam_adapt_*.npy  20,000 patches for adaptation / few-shot, from *different slides*

Oral submucous fibrosis (OSMF) images in dataset 3 are excluded: they are a premalignant
condition, not carcinoma, and leaving them out reproduces the 14,760 / 8,530 image totals
of the earlier study of Paun et al. (2026).

Usage:  python prepare_data.py            (everything)
        python prepare_data.py --dev      (development cache only)
        python prepare_data.py --pcam     (PatchCamelyon pools only)
"""
import argparse
import gzip
import io
import shutil
import urllib.request
import zipfile
from multiprocessing import Pool

import h5py
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import config as C

Image.MAX_IMAGE_PIXELS = None


# ----------------------------------------------------------------------------- dev index
def build_dev_index() -> pd.DataFrame:
    rows = []

    def add(zip_path, dataset, entry, label, subtype, split):
        rows.append(dict(dataset=dataset, label=label, subtype=subtype, native_split=split,
                         zip=zip_path.name, entry=entry))

    with zipfile.ZipFile(C.ZIP_DS1) as z:
        for n in z.namelist():
            p = n.split("/")
            if len(p) == 3 and p[2].lower().endswith((".jpg", ".jpeg", ".png")):
                label = {"Normal": 0, "OSCC": 1}[p[1]]
                add(C.ZIP_DS1, "ds1", n, label, "normal" if label == 0 else "oscc", p[0])

    with zipfile.ZipFile(C.ZIP_DS2) as z:
        for n in z.namelist():
            p = n.split("/")
            if len(p) == 3 and p[1] in ("oral_normal", "oral_scc") and p[2].lower().endswith(".jpg"):
                label = 0 if p[1] == "oral_normal" else 1
                add(C.ZIP_DS2, "ds2", n, label, "normal" if label == 0 else "scc", p[0])

    with zipfile.ZipFile(C.ZIP_DS3) as z:
        for n in z.namelist():
            p = n.split("/")
            if len(p) == 4 and p[3].lower().endswith(".jpg"):
                sub = p[2].lower()
                if sub == "osmf":          # premalignant, not carcinoma -> excluded
                    continue
                label = 0 if sub == "normal" else 1
                add(C.ZIP_DS3, "ds3", n, label, sub, p[1].lower())

    df = pd.DataFrame(rows)

    # Dataset 3 has no validation fold: carve one out of its training fold, per subtype.
    rng = np.random.default_rng(C.SEED)
    ds3_train = df[(df.dataset == "ds3") & (df.native_split == "train")]
    for _, grp in ds3_train.groupby("subtype"):
        n_val = int(round(len(grp) * C.DS3_VAL_FRACTION))
        df.loc[rng.choice(grp.index.to_numpy(), n_val, replace=False), "native_split"] = "val"

    df = df.sort_values(["dataset", "entry"]).reset_index(drop=True)
    df.insert(0, "idx", np.arange(len(df)))
    return df


# ----------------------------------------------------------------------------- dev cache
_ZIPS = {}


def _load_one(args):
    idx, zip_name, entry = args
    if zip_name not in _ZIPS:
        _ZIPS[zip_name] = zipfile.ZipFile(C.DATA_DIR / zip_name)
    s = C.CACHE_SIZE
    try:
        im = Image.open(io.BytesIO(_ZIPS[zip_name].read(entry)))
        im.draft("RGB", (s * 2, s * 2))            # fast JPEG down-scaling at decode time
        im = im.convert("RGB")
        w, h = im.size
        scale = s / min(w, h)
        nw, nh = max(s, round(w * scale)), max(s, round(h * scale))
        im = im.resize((nw, nh), Image.BICUBIC, reducing_gap=3.0)
        left, top = (nw - s) // 2, (nh - s) // 2
        return idx, np.asarray(im.crop((left, top, left + s, top + s)), dtype=np.uint8), (w, h)
    except Exception as exc:                       # corrupted file -> reported, not fatal
        return idx, None, str(exc)


def build_dev_cache(df: pd.DataFrame, workers: int = 6):
    s = C.CACHE_SIZE
    arr = np.lib.format.open_memmap(C.DEV_CACHE_NPY, mode="w+", dtype=np.uint8,
                                    shape=(len(df), s, s, 3))
    tasks = list(zip(df.idx, df.zip, df.entry))
    bad, sizes = [], {}
    with Pool(workers) as pool:
        for idx, img, info in tqdm(pool.imap_unordered(_load_one, tasks, chunksize=16),
                                   total=len(tasks), desc="decoding development images"):
            if img is None:
                bad.append((idx, info))
            else:
                arr[idx] = img
                sizes[idx] = info
    arr.flush()
    df["decoded_w"] = df.idx.map(lambda i: sizes.get(i, (0, 0))[0])
    df["decoded_h"] = df.idx.map(lambda i: sizes.get(i, (0, 0))[1])
    df["decoded"] = ~df.idx.isin([b[0] for b in bad])
    if bad:
        print(f"WARNING: {len(bad)} images could not be decoded:", bad[:5])
    return df


# ----------------------------------------------------------------------------- PCam
def _ensure_pcam_files():
    x_path = C.PCAM_DIR / C.PCAM_X_NAME
    if not x_path.exists():
        print("extracting PatchCamelyon image file from the zip (about 6.9 GB) ...")
        with zipfile.ZipFile(C.ZIP_PCAM) as z:
            member = [n for n in z.namelist() if n.endswith(C.PCAM_X_NAME)][0]
            with z.open(member) as src, open(x_path, "wb") as dst:
                shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)

    y_path = C.PCAM_DIR / C.PCAM_Y_NAME
    if not y_path.exists():
        # The Kaggle mirror ships images only; labels come from the official release.
        url = C.PCAM_ZENODO + C.PCAM_Y_NAME + ".gz?download=1"
        print("downloading labels:", url)
        with urllib.request.urlopen(url) as r:
            y_path.write_bytes(gzip.decompress(r.read()))

    meta_path = C.PCAM_DIR / C.PCAM_META_NAME
    if not meta_path.exists():
        url = C.PCAM_ZENODO + C.PCAM_META_NAME + "?download=1"
        print("downloading slide metadata:", url)
        try:
            with urllib.request.urlopen(url) as r:
                meta_path.write_bytes(r.read())
        except Exception as exc:
            print("  metadata unavailable (", exc, ") - falling back to a random split")
    return x_path, y_path, meta_path


def build_pcam_pools():
    x_path, y_path, meta_path = _ensure_pcam_files()
    with h5py.File(y_path, "r") as f:
        y = f["y"][:].reshape(-1).astype(np.int64)
    n = len(y)
    rng = np.random.default_rng(C.SEED)

    wsi = None
    if meta_path.exists():
        meta = pd.read_csv(meta_path)
        if len(meta) == n and "wsi" in meta.columns:
            wsi = meta["wsi"].astype(str).to_numpy()

    if wsi is not None:
        # Slide-disjoint pools: no slide contributes patches to both pools.
        slides = np.unique(wsi)
        rng.shuffle(slides)
        half = len(slides) // 2
        in_test = np.isin(wsi, slides[:half])
        cand = {"pcam_test": np.flatnonzero(in_test), "pcam_adapt": np.flatnonzero(~in_test)}
    else:
        perm = rng.permutation(n)
        cand = {"pcam_test": perm[: n // 2], "pcam_adapt": perm[n // 2:]}
        wsi = np.array(["unknown"] * n)

    chosen = {k: np.sort(rng.choice(v, C.PCAM_POOL_SIZE, replace=False)) for k, v in cand.items()}

    with h5py.File(x_path, "r") as f:
        ds = f["x"]
        assert ds.shape[0] == n, f"image/label count mismatch: {ds.shape[0]} vs {n}"
        for name, idx in chosen.items():
            out = np.empty((len(idx), 96, 96, 3), dtype=np.uint8)
            block = 4096
            pos = 0
            for start in tqdm(range(0, n, block), desc=f"reading {name}"):
                sel = idx[(idx >= start) & (idx < start + block)]
                if len(sel) == 0:
                    continue
                chunk = ds[start:start + block]
                out[pos:pos + len(sel)] = chunk[sel - start]
                pos += len(sel)
            np.save(C.CACHE_DIR / f"{name}_x.npy", out)
            pd.DataFrame(dict(pcam_index=idx, label=y[idx], wsi=wsi[idx])).to_csv(
                C.CACHE_DIR / f"{name}_index.csv", index=False)
            print(f"{name}: {len(idx)} patches, {int(y[idx].sum())} metastatic, "
                  f"{len(np.unique(wsi[idx]))} slides")


# ----------------------------------------------------------------------------- main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", action="store_true")
    ap.add_argument("--pcam", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    do_all = not (a.dev or a.pcam)

    if a.dev or do_all:
        index = build_dev_index()
        print(index.groupby(["dataset", "label"]).size().unstack())
        print("total:", len(index), "| malignant:", int(index.label.sum()),
              "| normal:", int((index.label == 0).sum()))
        index = build_dev_cache(index, a.workers)
        index.to_csv(C.DEV_INDEX_CSV, index=False)
        print("saved", C.DEV_INDEX_CSV, "and", C.DEV_CACHE_NPY)

    if a.pcam or do_all:
        build_pcam_pools()
