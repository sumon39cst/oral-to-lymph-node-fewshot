"""Revision experiments that need new image decoding or a GPU (review comments M2, M4, M5).

    python revision_gpu.py scale     field-of-view control: oral images re-cropped to 1/2, 1/4 and
                                     1/8 of the field before embedding (physical-scale confound)
    python revision_gpu.py tissues   two further target tissues from the multi-cancer collection:
                                     lung (squamous cell carcinoma vs benign) and colon
                                     (adenocarcinoma vs benign), embedded with every encoder
    python revision_gpu.py plip      zero-shot vision-language baseline (PLIP), no target labels
    python revision_gpu.py adabn     AdaBN on the fine-tuned networks that contain batch norm

Arrays go to dataset/cache, embeddings to dataset/cache/features/<encoder>/, tables to
result_prediction/revision/.
"""
import io
import sys
import zipfile
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

import config as C
from extract_features import embed_array
from metrics import all_metrics, bootstrap_ci
from models import Classifier, load_backbone, preprocess

REV = C.RESULT_DIR / "revision"
REV.mkdir(parents=True, exist_ok=True)
FOVS = {"fov50": 0.5, "fov25": 0.25, "fov12": 0.125}
SCALE_ENCODERS = ["efficientnet_b0", "phikon", "phikon_v2"]
TISSUES = {"lung": ("lung_bnt", "lung_scc"), "colon": ("colon_bnt", "colon_aca")}
Image.MAX_IMAGE_PIXELS = None
_ZIPS = {}


def _zip(name):
    if name not in _ZIPS:
        _ZIPS[name] = zipfile.ZipFile(C.DATA_DIR / name)
    return _ZIPS[name]


# ============================================================================== scale control
def _crops(args):
    idx, zip_name, entry = args
    im = Image.open(io.BytesIO(_zip(zip_name).read(entry))).convert("RGB")
    w, h = im.size
    s = min(w, h)
    out = []
    for frac in FOVS.values():
        c = max(8, int(round(s * frac)))
        left, top = (w - c) // 2, (h - c) // 2
        out.append(np.asarray(im.crop((left, top, left + c, top + c)).resize((224, 224), Image.BICUBIC),
                              dtype=np.uint8))
    return idx, out, s


def run_scale():
    index = pd.read_csv(C.DEV_INDEX_CSV)
    paths = {k: C.CACHE_DIR / f"dev_{k}.npy" for k in FOVS}
    if not all(p.exists() for p in paths.values()):
        arrs = {k: np.lib.format.open_memmap(p, mode="w+", dtype=np.uint8, shape=(len(index), 224, 224, 3))
                for k, p in paths.items()}
        short = np.zeros(len(index), int)
        with Pool(6) as pool:
            for idx, crops, s in tqdm(pool.imap_unordered(_crops, list(zip(index.idx, index.zip, index.entry)),
                                                          chunksize=8), total=len(index), desc="cropping"):
                for (k, _), c in zip(FOVS.items(), crops):
                    arrs[k][idx] = c
                short[idx] = s
        for a in arrs.values():
            a.flush()
        index.assign(short_side_px=short).groupby("dataset").short_side_px.describe().to_csv(
            REV / "scale_original_short_side_px.csv")
    for key in SCALE_ENCODERS:
        bb = None
        for k, p in paths.items():
            out = C.FEATURE_DIR / key / f"dev_{k}.npy"
            if out.exists():
                continue
            bb = bb or load_backbone(key, "cuda").eval()
            np.save(out, embed_array(bb, np.load(p, mmap_mode="r"), 64, "cuda", f"{key}/{k}"))
        del bb
        torch.cuda.empty_cache()


# ============================================================================== second tissues
def _decode256(args):
    i, entry = args
    im = Image.open(io.BytesIO(_zip(C.ZIP_DS2.name).read(entry))).convert("RGB")
    w, h = im.size
    sc = 256 / min(w, h)
    im = im.resize((max(256, round(w * sc)), max(256, round(h * sc))), Image.BICUBIC)
    left, top = (im.size[0] - 256) // 2, (im.size[1] - 256) // 2
    return i, np.asarray(im.crop((left, top, left + 256, top + 256)), dtype=np.uint8)


def run_tissues():
    rng = np.random.default_rng(C.SEED)
    names = zipfile.ZipFile(C.ZIP_DS2).namelist()
    for tissue, (neg, pos) in TISSUES.items():
        if (C.CACHE_DIR / f"{tissue}_adapt_x.npy").exists():
            continue
        rows = []
        for n in names:
            p = n.split("/")
            if len(p) == 3 and p[1] in (neg, pos) and p[2].lower().endswith(".jpg"):
                rows.append(dict(entry=n, split=p[0], label=int(p[1] == pos)))
        df = pd.DataFrame(rows)
        ev = df[df.split.isin(["val", "test"])]
        tr = df[df.split == "train"]
        ad = pd.concat([tr[tr.label == y].sample(1000, random_state=C.SEED) for y in (0, 1)])
        for pool_name, part in (("eval", ev), ("adapt", ad)):
            part = part.sort_values("entry").reset_index(drop=True)
            arr = np.empty((len(part), 256, 256, 3), np.uint8)
            with Pool(6) as pool:
                for i, im in tqdm(pool.imap_unordered(_decode256, list(enumerate(part.entry)), chunksize=16),
                                  total=len(part), desc=f"{tissue}/{pool_name}"):
                    arr[i] = im
            np.save(C.CACHE_DIR / f"{tissue}_{pool_name}_x.npy", arr)
            part.to_csv(C.CACHE_DIR / f"{tissue}_{pool_name}_index.csv", index=False)
            print(tissue, pool_name, len(part), "images,", int(part.label.sum()), "malignant")
    for key in C.DEFAULT_BACKBONES:
        bb = None
        for tissue in TISSUES:
            for pool_name in ("eval", "adapt"):
                out = C.FEATURE_DIR / key / f"{tissue}_{pool_name}.npy"
                if out.exists():
                    continue
                bb = bb or load_backbone(key, "cuda").eval()
                arr = np.load(C.CACHE_DIR / f"{tissue}_{pool_name}_x.npy", mmap_mode="r")
                np.save(out, embed_array(bb, arr, 64, "cuda", f"{key}/{tissue}_{pool_name}"))
        del bb
        torch.cuda.empty_cache()


# ============================================================================== PLIP zero-shot
PROMPTS = {
    "pcam": (["an H&E image of normal lymph node tissue", "an H&E image of benign lymphoid tissue",
              "a histopathology image of a lymph node without tumor"],
             ["an H&E image of lymph node metastasis", "an H&E image of metastatic carcinoma in a lymph node",
              "a histopathology image of a lymph node with tumor"]),
    "lung": (["an H&E image of benign lung tissue", "a histopathology image of normal lung"],
             ["an H&E image of lung squamous cell carcinoma", "a histopathology image of squamous cell carcinoma of the lung"]),
    "colon": (["an H&E image of benign colon tissue", "a histopathology image of normal colonic mucosa"],
              ["an H&E image of colon adenocarcinoma", "a histopathology image of colorectal adenocarcinoma"]),
    "oral": (["an H&E image of normal oral mucosa", "a histopathology image of normal oral squamous epithelium"],
             ["an H&E image of oral squamous cell carcinoma", "a histopathology image of squamous cell carcinoma of the oral cavity"]),
}


@torch.no_grad()
def run_plip():
    from transformers import CLIPModel, CLIPTokenizer
    name = "vinid/plip"
    model = CLIPModel.from_pretrained(name).cuda().eval()
    tok = CLIPTokenizer.from_pretrained(name)
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).cuda()
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).cuda()

    def text_embed(prompts):
        t = tok(prompts, padding=True, truncation=True, return_tensors="pt").to("cuda")
        e = model.get_text_features(**t)
        e = e / e.norm(dim=-1, keepdim=True)
        e = e.mean(0)
        return e / e.norm()

    def image_embed(arr):
        out = []
        for s in tqdm(range(0, len(arr), 128), leave=False):
            x = torch.from_numpy(np.ascontiguousarray(arr[s:s + 128])).cuda().permute(0, 3, 1, 2).float() / 255
            x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bicubic", align_corners=False,
                                                antialias=x.shape[-1] > 224).clamp(0, 1)
            e = model.get_image_features(pixel_values=(x - mean) / std)
            out.append((e / e.norm(dim=-1, keepdim=True)).float().cpu())
        return torch.cat(out)

    index = pd.read_csv(C.DEV_INDEX_CSV)
    oral_test = index[index.native_split == "test"].idx.to_numpy()
    sets = {"pcam": (np.load(C.CACHE_DIR / "pcam_test_x.npy", mmap_mode="r"),
                     pd.read_csv(C.CACHE_DIR / "pcam_test_index.csv"))}
    for t in TISSUES:
        if (C.CACHE_DIR / f"{t}_eval_x.npy").exists():
            sets[t] = (np.load(C.CACHE_DIR / f"{t}_eval_x.npy", mmap_mode="r"),
                       pd.read_csv(C.CACHE_DIR / f"{t}_eval_index.csv"))
    sets["oral"] = (np.load(C.DEV_CACHE_NPY, mmap_mode="r")[np.sort(oral_test)],
                    index.set_index("idx").loc[np.sort(oral_test)].reset_index())
    rows = []
    for name_, (arr, meta) in sets.items():
        neg, pos = PROMPTS[name_]
        img = image_embed(arr)
        d = (img @ text_embed(pos).cpu() - img @ text_embed(neg).cpu()).numpy()
        p = 1 / (1 + np.exp(-100 * d))
        y = meta.label.to_numpy()
        m = all_metrics(y, p, 0.5)
        groups = meta.wsi.to_numpy() if "wsi" in meta else None
        m.update(bootstrap_ci(y, p, 0.5, groups=groups, n_boot=1000))
        rows.append(dict(model="PLIP (zero-shot)", target=name_, **m))
        np.save(REV / f"plip_zero_shot_prob_{name_}.npy", p)
        print(f"PLIP zero-shot {name_:<6} acc {m['accuracy']:.3f} sens {m['sensitivity']:.3f} "
              f"spec {m['specificity']:.3f} auroc {m['auroc']:.3f} ({m['auroc_lo']:.3f}-{m['auroc_hi']:.3f})")
        if name_ == "pcam":                                  # image embeddings reused for few-shot
            np.save(REV / "plip_pcam_test_embed.npy", img.numpy())
            np.save(REV / "plip_pcam_adapt_embed.npy",
                    image_embed(np.load(C.CACHE_DIR / "pcam_adapt_x.npy", mmap_mode="r")).numpy())
    pd.DataFrame(rows).to_csv(REV / "plip_zero_shot.csv", index=False)


# ============================================================================== AdaBN
@torch.no_grad()
def run_adabn():
    adapt = np.load(C.CACHE_DIR / "pcam_adapt_x.npy", mmap_mode="r")
    test = np.load(C.CACHE_DIR / "pcam_test_x.npy", mmap_mode="r")
    meta = pd.read_csv(C.CACHE_DIR / "pcam_test_index.csv")
    y, wsi = meta.label.to_numpy(), meta.wsi.to_numpy()
    rows = []
    for ckpt in sorted(C.CKPT_DIR.glob("*.pt")):
        key = ckpt.stem.split("__")[0]
        model = Classifier(load_backbone(key, "cuda")).cuda()
        model.load_state_dict(torch.load(ckpt, map_location="cuda"))
        bns = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
        if not bns:
            continue

        def predict():
            model.eval()
            out = []
            for s in range(0, len(test), 128):
                x = preprocess(torch.from_numpy(np.ascontiguousarray(test[s:s + 128])), model.backbone)
                with torch.autocast("cuda", dtype=torch.float16):
                    out.append(torch.sigmoid(model(x).float()).cpu().numpy())
            return np.concatenate(out)

        before = predict()
        for m in bns:                                        # re-estimate statistics on the target
            m.reset_running_stats()
            m.momentum = None
        model.eval()
        for m in bns:
            m.train()
        for s in tqdm(range(0, len(adapt), 128), desc=f"AdaBN {ckpt.stem}", leave=False):
            x = preprocess(torch.from_numpy(np.ascontiguousarray(adapt[s:s + 128])), model.backbone)
            with torch.autocast("cuda", dtype=torch.float16):
                model(x)
        after = predict()
        for tag, p in (("source statistics", before), ("AdaBN (target statistics)", after)):
            m = all_metrics(y, p, 0.5)
            m.update(bootstrap_ci(y, p, 0.5, groups=wsi, n_boot=1000))
            rows.append(dict(run=ckpt.stem, batch_norm=tag, n_bn_layers=len(bns), **m))
            print(f"{ckpt.stem:<44}{tag:<28} acc {m['accuracy']:.3f} sens {m['sensitivity']:.3f} "
                  f"spec {m['specificity']:.3f} auroc {m['auroc']:.3f}")
        del model
        torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(REV / "adabn.csv", index=False)


if __name__ == "__main__":
    {"scale": run_scale, "tissues": run_tissues, "plip": run_plip, "adabn": run_adabn}[sys.argv[1]]()
