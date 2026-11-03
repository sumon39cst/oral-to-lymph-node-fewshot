"""Step 5 (Experiment C) - end-to-end fine-tuning: architecture study under domain shift.

Trains a backbone + linear head on the oral datasets (native, leakage-safe split) and
evaluates it on the oral test fold and on the PatchCamelyon evaluation pool. Used for the
CNN vs. hierarchical transformer vs. state-space comparison, each with and without H&E
stain augmentation as a domain-generalisation step.

    python finetune.py --model swinv2_tiny
    python finetune.py --model swinv2_tiny --stain_aug
    python finetune.py --model efficientnet_b0 --stain_aug --epochs 15

Outputs  checkpoints/<run>.pt, predictions/finetune/<run>__{internal,external}.csv,
         tables/finetune_metrics.csv, logs/finetune_<run>.csv (learning curves)
"""
import argparse
import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

import config as C
from metrics import all_metrics, bootstrap_ci, youden_threshold
from models import Classifier, load_backbone, preprocess

# Ruifrok & Johnston colour-deconvolution matrix (rows: haematoxylin, eosin, DAB).
_RGB_FROM_HED = torch.tensor([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11], [0.27, 0.57, 0.78]])
_HED_FROM_RGB = torch.linalg.inv(_RGB_FROM_HED)


def hed_stain_jitter(x, sigma=0.05, bias=0.02):
    """Random per-image perturbation of the H, E and residual stain channels.

    x: float [B, 3, H, W] in [0, 1]. Simulates laboratory / scanner stain variation
    (Tellez et al., 2018) so the network cannot rely on a dataset's colour signature.
    """
    b = x.shape[0]
    hed_from_rgb, rgb_from_hed = _HED_FROM_RGB.to(x), _RGB_FROM_HED.to(x)
    od = -torch.log(x.clamp_min(1e-6)).permute(0, 2, 3, 1)             # optical density
    hed = od @ hed_from_rgb
    alpha = 1 + (torch.rand(b, 1, 1, 3, device=x.device) * 2 - 1) * sigma
    beta = (torch.rand(b, 1, 1, 3, device=x.device) * 2 - 1) * bias
    od = (hed * alpha + beta) @ rgb_from_hed
    return torch.exp(-od).permute(0, 3, 1, 2).clamp(0, 1)


def augment(x_uint8, backbone, stain_aug):
    """GPU augmentation: random crop-resize, flips, 90-degree rotations, optional stain jitter."""
    x = x_uint8.to(backbone.mean.device).permute(0, 3, 1, 2).float().div_(255.0)
    b, _, h, w = x.shape
    s = backbone.img_size
    scale = torch.empty(1).uniform_(0.6, 1.0).item()
    ch = int(h * math.sqrt(scale))
    top = torch.randint(0, h - ch + 1, (1,)).item()
    left = torch.randint(0, w - ch + 1, (1,)).item()
    x = x[:, :, top:top + ch, left:left + ch]
    x = F.interpolate(x, size=(s, s), mode="bilinear", align_corners=False,
                      antialias=ch > s).clamp_(0, 1)
    if torch.rand(1).item() < 0.5:
        x = x.flip(-1)
    if torch.rand(1).item() < 0.5:
        x = x.flip(-2)
    x = torch.rot90(x, int(torch.randint(0, 4, (1,)).item()), dims=(-2, -1))
    if stain_aug:
        x = hed_stain_jitter(x)
    return (x - backbone.mean) / backbone.std


@torch.no_grad()
def predict(model, arr, idx, batch_size):
    model.eval()
    out = []
    for start in range(0, len(idx), batch_size):
        sel = np.sort(idx[start:start + batch_size])
        x = preprocess(torch.from_numpy(np.ascontiguousarray(arr[sel])), model.backbone)
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(torch.sigmoid(model(x).float()).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--stain_aug", action="store_true")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=48)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--head_lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--max_steps", type=int, default=0, help="debug: stop early")
    ap.add_argument("--mmap", action="store_true", help="do not load the image cache into RAM")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = "cuda"
    torch.backends.cudnn.benchmark = True
    run = f"{a.model}__{'stainaug' if a.stain_aug else 'noaug'}__seed{a.seed}"

    index = pd.read_csv(C.DEV_INDEX_CSV)
    ok = index[index.decoded]
    tr = ok[ok.native_split == "train"].idx.to_numpy()
    va = np.sort(ok[ok.native_split == "val"].idx.to_numpy())
    te = np.sort(ok[ok.native_split == "test"].idx.to_numpy())
    y_all = index.label.to_numpy()
    # Random mini-batches from a memory-mapped file are very slow on a hard disk, so the
    # 4.4 GB cache is read into RAM once (pass --mmap on machines with little memory).
    dev = np.load(C.DEV_CACHE_NPY, mmap_mode="r" if a.mmap else None)
    pcam_x = np.load(C.CACHE_DIR / "pcam_test_x.npy", mmap_mode="r")
    pcam = pd.read_csv(C.CACHE_DIR / "pcam_test_index.csv")

    model = Classifier(load_backbone(a.model, device)).to(device)
    pos_weight = torch.tensor((y_all[tr] == 0).sum() / (y_all[tr] == 1).sum(), device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW([{"params": model.backbone.parameters(), "lr": a.lr},
                             {"params": model.head.parameters(), "lr": a.head_lr}],
                            weight_decay=a.weight_decay)
    steps_per_epoch = len(tr) // a.batch_size
    total = steps_per_epoch * a.epochs
    warmup = steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    scaler = torch.amp.GradScaler("cuda")

    best_auc, best_epoch, bad, history, step = -1.0, -1, 0, [], 0
    ckpt = C.CKPT_DIR / f"{run}.pt"
    rng = np.random.default_rng(a.seed)
    for epoch in range(a.epochs):
        model.train()
        perm = rng.permutation(tr)
        t0, losses = time.time(), []
        for i in range(steps_per_epoch):
            sel = np.sort(perm[i * a.batch_size:(i + 1) * a.batch_size])
            x = augment(torch.from_numpy(np.ascontiguousarray(dev[sel])), model.backbone, a.stain_aug)
            y = torch.from_numpy(y_all[sel]).float().to(device)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = criterion(model(x).float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(loss.item())
            step += 1
            if a.max_steps and step >= a.max_steps:
                break
        p_val = predict(model, dev, va, a.batch_size * 2)
        auc = roc_auc_score(y_all[va], p_val)
        history.append(dict(epoch=epoch + 1, train_loss=float(np.mean(losses)), val_auroc=auc,
                            val_acc=float(((p_val >= 0.5) == y_all[va]).mean()),
                            seconds=round(time.time() - t0, 1)))
        print(history[-1], flush=True)
        if auc > best_auc:
            best_auc, best_epoch, bad = auc, epoch + 1, 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
            if bad >= a.patience:
                break
        if a.max_steps and step >= a.max_steps:
            break
    pd.DataFrame(history).to_csv(C.LOG_DIR / f"finetune_{run}.csv", index=False)

    model.load_state_dict(torch.load(ckpt, map_location=device))
    p_val = predict(model, dev, va, a.batch_size * 2)
    p_int = predict(model, dev, te, a.batch_size * 2)
    p_ext = predict(model, pcam_x, np.arange(len(pcam_x)), a.batch_size * 2)
    y_ext, wsi = pcam.label.to_numpy(), pcam.wsi.to_numpy()
    groups = wsi if len(np.unique(wsi)) > 1 else None

    out_dir = C.PRED_DIR / "finetune"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(dict(idx=te, dataset=index.dataset.to_numpy()[te], label=y_all[te],
                      prob=p_int)).to_csv(out_dir / f"{run}__internal.csv", index=False)
    pd.DataFrame(dict(pcam_index=pcam.pcam_index, wsi=wsi, label=y_ext,
                      prob=p_ext)).to_csv(out_dir / f"{run}__external.csv", index=False)

    rows = []
    thresholds = {"fixed_0.5": 0.5, "source_youden": youden_threshold(y_all[va], p_val)}
    for eval_set, y, p, g in (("internal", y_all[te], p_int, None), ("external", y_ext, p_ext, groups)):
        for rule, thr in thresholds.items():
            row = dict(run=run, backbone=a.model, family=C.BACKBONES[a.model]["family"],
                       stain_aug=a.stain_aug, seed=a.seed, best_epoch=best_epoch,
                       val_auroc=best_auc, eval_set=eval_set, threshold_rule=rule)
            row.update(all_metrics(y, p, thr))
            if not a.max_steps:
                row.update(bootstrap_ci(y, p, thr, groups=g, n_boot=1000))
            rows.append(row)
            print(f"{eval_set:<9}{rule:<14} acc {row['accuracy']:.3f} sens {row['sensitivity']:.3f} "
                  f"spec {row['specificity']:.3f} auroc {row['auroc']:.3f}")
    out_csv = C.TABLE_DIR / "finetune_metrics.csv"
    table = pd.read_csv(out_csv) if out_csv.exists() else pd.DataFrame()
    if len(table):
        table = table[table.run != run]
    pd.concat([table, pd.DataFrame(rows)], ignore_index=True).to_csv(out_csv, index=False)
    print("saved", out_csv)


if __name__ == "__main__":
    main()
