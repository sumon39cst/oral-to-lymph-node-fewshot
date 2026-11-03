"""Zero-shot vision-language baseline (PLIP): target classes defined by text, no image labels.

PLIP is distributed as a pickled checkpoint, which PyTorch older than 2.6 does not load
safely. This script therefore runs in its own environment (CPU is sufficient):

    python -m venv .venv_plip
    .venv_plip\\Scripts\\python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
    .venv_plip\\Scripts\\python -m pip install "transformers<5" numpy pandas scikit-learn pillow tqdm
    .venv_plip\\Scripts\\python plip_zero_shot.py

Outputs (result_prediction/revision/): plip_zero_shot.csv, plip_zero_shot_prompt_sets.csv,
per-patch probabilities and PLIP image embeddings of both PatchCamelyon pools.
"""
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, CLIPModel

import config as C
from metrics import all_metrics, bootstrap_ci

REV = C.RESULT_DIR / "revision"
NAME = "vinid/plip"
# Several wordings per task: zero-shot results depend on the prompt, so the spread is reported.
PROMPTS = {
    "pcam": [
        (["an H&E image of normal lymph node tissue"], ["an H&E image of lymph node metastasis"]),
        (["an H&E image of benign lymphoid tissue"], ["an H&E image of metastatic carcinoma in a lymph node"]),
        (["a histopathology image of a lymph node without tumor"], ["a histopathology image of a lymph node with tumor"]),
        (["normal lymph node"], ["metastatic breast carcinoma in a lymph node"]),
        (["lymphocytes"], ["tumor cells"]),
    ],
    "lung": [(["an H&E image of benign lung tissue"], ["an H&E image of lung squamous cell carcinoma"]),
             (["a histopathology image of normal lung"], ["a histopathology image of squamous cell carcinoma of the lung"])],
    "colon": [(["an H&E image of benign colon tissue"], ["an H&E image of colon adenocarcinoma"]),
              (["a histopathology image of normal colonic mucosa"], ["a histopathology image of colorectal adenocarcinoma"])],
    "oral": [(["an H&E image of normal oral mucosa"], ["an H&E image of oral squamous cell carcinoma"]),
             (["a histopathology image of normal oral squamous epithelium"],
              ["a histopathology image of squamous cell carcinoma of the oral cavity"])],
}


@torch.no_grad()
def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(NAME).to(dev).eval()
    tok = AutoTokenizer.from_pretrained(NAME)
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(dev)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(dev)

    def text_embed(prompts):
        e = model.get_text_features(**tok(prompts, padding=True, truncation=True, return_tensors="pt").to(dev))
        e = (e / e.norm(dim=-1, keepdim=True)).mean(0)
        return (e / e.norm()).cpu()

    def image_embed(arr, desc):
        out = []
        for s in tqdm(range(0, len(arr), 64), desc=desc):
            x = torch.from_numpy(np.ascontiguousarray(arr[s:s + 64])).to(dev).permute(0, 3, 1, 2).float() / 255
            x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bicubic", align_corners=False,
                                                antialias=x.shape[-1] > 224).clamp(0, 1)
            e = model.get_image_features(pixel_values=(x - mean) / std)
            out.append((e / e.norm(dim=-1, keepdim=True)).float().cpu())
        return torch.cat(out)

    index = pd.read_csv(C.DEV_INDEX_CSV)
    oral_test = np.sort(index[index.native_split == "test"].idx.to_numpy())
    sets = {"pcam": (np.load(C.CACHE_DIR / "pcam_test_x.npy", mmap_mode="r"),
                     pd.read_csv(C.CACHE_DIR / "pcam_test_index.csv"))}
    for t in ("lung", "colon"):
        if (C.CACHE_DIR / f"{t}_eval_x.npy").exists():
            sets[t] = (np.load(C.CACHE_DIR / f"{t}_eval_x.npy", mmap_mode="r"),
                       pd.read_csv(C.CACHE_DIR / f"{t}_eval_index.csv"))
    sets["oral"] = (np.load(C.DEV_CACHE_NPY, mmap_mode="r")[oral_test],
                    index.set_index("idx").loc[oral_test].reset_index())

    main_rows, prompt_rows = [], []
    for name, (arr, meta) in sets.items():
        img = image_embed(arr, name)
        y = meta.label.to_numpy()
        groups = meta.wsi.to_numpy() if "wsi" in meta else None
        diffs = []
        for i, (neg, pos) in enumerate(PROMPTS[name]):
            d = (img @ text_embed(pos) - img @ text_embed(neg)).numpy()
            diffs.append(d)
            m = all_metrics(y, 1 / (1 + np.exp(-100 * d)), 0.5)
            prompt_rows.append(dict(target=name, prompt_set=i, negative=neg[0], positive=pos[0],
                                    auroc=m["auroc"], sensitivity=m["sensitivity"], specificity=m["specificity"],
                                    positive_rate=m["positive_rate"]))
        d = np.mean(diffs, 0)                                    # prompt ensemble
        p = 1 / (1 + np.exp(-100 * d))
        m = all_metrics(y, p, 0.5)
        m.update(bootstrap_ci(y, p, 0.5, groups=groups, n_boot=1000))
        main_rows.append(dict(model="PLIP zero-shot, prompt ensemble", target=name, **m))
        np.save(REV / f"plip_zero_shot_prob_{name}.npy", p)
        print(f"PLIP zero-shot {name:<6} acc {m['accuracy']:.3f} sens {m['sensitivity']:.3f} spec {m['specificity']:.3f} "
              f"auroc {m['auroc']:.3f} ({m['auroc_lo']:.3f}-{m['auroc_hi']:.3f}) | single prompts: "
              + ", ".join(f"{r['auroc']:.3f}" for r in prompt_rows if r["target"] == name), flush=True)
        if name == "pcam":
            np.save(REV / "plip_pcam_test_embed.npy", img.numpy())
            np.save(REV / "plip_pcam_adapt_embed.npy",
                    image_embed(np.load(C.CACHE_DIR / "pcam_adapt_x.npy", mmap_mode="r"), "pcam adapt").numpy())
    pd.DataFrame(main_rows).to_csv(REV / "plip_zero_shot.csv", index=False)
    pd.DataFrame(prompt_rows).round(4).to_csv(REV / "plip_zero_shot_prompt_sets.csv", index=False)


if __name__ == "__main__":
    main()
