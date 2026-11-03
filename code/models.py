"""Backbone factory: one uniform interface over CNNs, hierarchical transformers,
state-space models, general self-supervised ViTs and pathology foundation models.

    backbone = load_backbone("phikon_v2")      # nn.Module, forward(x) -> [B, D] embedding
    x = preprocess(uint8_batch, backbone)      # resize + normalise on the GPU

Gated models (UNI, UNI2-h, H-optimus, Virchow2) need a Hugging Face account that has been
granted access and a prior `huggingface-cli login`; load_backbone raises BackboneUnavailable
with a readable message otherwise, and the calling scripts skip that backbone.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import config as C

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class BackboneUnavailable(RuntimeError):
    pass


class Backbone(nn.Module):
    """Wraps any encoder so that forward(x) returns one embedding vector per image."""

    def __init__(self, key, net, embed_fn, mean, std, img_size):
        super().__init__()
        self.key = key
        self.net = net
        self._embed = embed_fn
        self.img_size = img_size
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1), persistent=False)
        self.embed_dim = None

    def forward(self, x):
        return self._embed(self.net, x)

    @torch.no_grad()
    def infer_dim(self, device):
        was_training = self.training
        self.eval()
        x = torch.zeros(1, 3, self.img_size, self.img_size, device=device)
        self.embed_dim = int(self.forward(x).shape[-1])
        self.train(was_training)
        return self.embed_dim


def _timm_embed(net, x):
    return net(x)


def _hf_cls_embed(net, x):
    return net(pixel_values=x).last_hidden_state[:, 0]


def _hf_clsmean_embed(net, x):
    """[CLS token, mean of the remaining tokens], the classification embedding of the Midnight model card."""
    t = net(pixel_values=x).last_hidden_state
    return torch.cat([t[:, 0], t[:, 1:].mean(1)], dim=-1)


def _virchow_embed(net, x):
    tokens = net(x)                                   # [B, 1 + 4 + 256, 1280]
    return torch.cat([tokens[:, 0], tokens[:, 5:].mean(1)], dim=-1)


def _mambavision_embed(net, x):
    out_avg_pool, _ = net(x)
    return out_avg_pool


def load_backbone(key: str, device="cuda") -> Backbone:
    if key not in C.BACKBONES:
        raise KeyError(f"unknown backbone '{key}'. Known: {list(C.BACKBONES)}")
    spec = C.BACKBONES[key]
    kind, name, img_size = spec["kind"], spec["name"], spec["img_size"]
    mean, std = spec.get("mean"), spec.get("std")
    try:
        if kind in ("timm", "virchow"):
            import timm
            kwargs = dict(spec.get("kwargs", {}))
            if spec.get("uni2"):
                kwargs.update(img_size=224, patch_size=14, depth=24, num_heads=24,
                              init_values=1e-5, embed_dim=1536, mlp_ratio=2.66667 * 2,
                              no_embed_class=True, mlp_layer=timm.layers.SwiGLUPacked,
                              act_layer=torch.nn.SiLU, reg_tokens=8, dynamic_img_size=True)
            if kind == "virchow":
                kwargs.update(mlp_layer=timm.layers.SwiGLUPacked, act_layer=torch.nn.SiLU)
                net = timm.create_model(name, pretrained=True, **kwargs)
                embed = _virchow_embed
            else:
                net = timm.create_model(name, pretrained=True, num_classes=0, **kwargs)
                embed = _timm_embed
            if mean is None:
                cfg = timm.data.resolve_data_config({}, model=net)
                mean, std = cfg["mean"], cfg["std"]
        elif kind == "hf":
            from transformers import AutoModel
            net = AutoModel.from_pretrained(name)
            embed = _hf_cls_embed
            mean, std = mean or IMAGENET_MEAN, std or IMAGENET_STD
        elif kind == "hf_clsmean":
            from transformers import AutoModel
            net = AutoModel.from_pretrained(name)
            embed = _hf_clsmean_embed
            mean, std = mean or IMAGENET_MEAN, std or IMAGENET_STD
        elif kind == "mambavision":
            import importlib.util
            import sys
            if importlib.util.find_spec("mamba_ssm") is None:
                # No CUDA kernel available (e.g. Windows): use the pure-PyTorch scan.
                sys.path.insert(0, str(C.CODE_DIR / "compat"))
            from transformers import AutoModel
            net = AutoModel.from_pretrained(name, trust_remote_code=True)
            embed = _mambavision_embed
            mean, std = mean or IMAGENET_MEAN, std or IMAGENET_STD
        else:
            raise ValueError(kind)
    except Exception as exc:                           # gated repo, missing package, ...
        hint = ""
        if spec.get("gated"):
            hint = (" This model is gated: request access on its Hugging Face page, then run "
                    "`huggingface-cli login`.")
        if kind == "mambavision":
            hint = " MambaVision also needs the `einops` package."
        raise BackboneUnavailable(f"{key}: {type(exc).__name__}: {str(exc)[:300]}.{hint}") from exc

    bb = Backbone(key, net, embed, tuple(mean), tuple(std), img_size).to(device)
    bb.infer_dim(device)
    return bb


def preprocess(x_uint8: torch.Tensor, backbone: Backbone, size: int = None) -> torch.Tensor:
    """uint8 [B, H, W, 3] on any device -> normalised float [B, 3, S, S] on the model device.

    Down-scaling uses antialiasing; up-scaling (96 px lymph node patches) is bicubic.
    """
    size = size or backbone.img_size
    x = x_uint8.to(backbone.mean.device, non_blocking=True).permute(0, 3, 1, 2).float().div_(255.0)
    if x.shape[-1] != size or x.shape[-2] != size:
        x = F.interpolate(x, size=(size, size), mode="bicubic", align_corners=False,
                          antialias=x.shape[-1] > size).clamp_(0, 1)
    return (x - backbone.mean) / backbone.std


class Classifier(nn.Module):
    """Backbone + binary head, used for end-to-end fine-tuning."""

    def __init__(self, backbone: Backbone, dropout: float = 0.2):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(backbone.embed_dim, 1))

    def forward(self, x):
        return self.head(self.backbone(x).float()).squeeze(-1)
