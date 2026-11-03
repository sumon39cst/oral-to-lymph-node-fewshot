"""Central configuration: paths, seeds, dataset layout and the backbone registry.

Every script in this folder imports from here, so changing a path or adding a
backbone is done in exactly one place.
"""
from pathlib import Path

# --------------------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[1]   # repository root; place dataset/ here
DATA_DIR = ROOT / "dataset"
CODE_DIR = ROOT / "code"
RESULT_DIR = ROOT / "result_prediction"
ARCH_FIG_DIR = ROOT / "architecture_figure"

CACHE_DIR = DATA_DIR / "cache"            # decoded image arrays + indices
FEATURE_DIR = CACHE_DIR / "features"      # frozen embeddings, one folder per backbone
PCAM_DIR = DATA_DIR / "pcam"              # extracted PatchCamelyon files

PRED_DIR = RESULT_DIR / "predictions"     # per-patch probabilities (CSV)
TABLE_DIR = RESULT_DIR / "tables"         # metric tables (CSV)
FIG_DIR = RESULT_DIR / "figures"          # result figures (PNG + PDF)
CKPT_DIR = RESULT_DIR / "checkpoints"     # fine-tuned weights
LOG_DIR = RESULT_DIR / "logs"

for _d in (CACHE_DIR, FEATURE_DIR, PCAM_DIR, PRED_DIR, TABLE_DIR, FIG_DIR, CKPT_DIR,
           LOG_DIR, ARCH_FIG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- data
ZIP_DS1 = DATA_DIR / "dataset1_oral_cancer_OSCC.zip"
ZIP_DS2 = DATA_DIR / "dataset2_multicancer.zip"
ZIP_DS3 = DATA_DIR / "dataset3_OralNew.zip"
ZIP_PCAM = DATA_DIR / "dataset4_PatchCamelyon.zip"

PCAM_X_NAME = "camelyonpatch_level_2_split_train_x.h5"
PCAM_Y_NAME = "camelyonpatch_level_2_split_train_y.h5"
PCAM_META_NAME = "camelyonpatch_level_2_split_train_meta.csv"
# Official PatchCamelyon release (Veeling et al., 2018), Zenodo record 2546921.
PCAM_ZENODO = "https://zenodo.org/records/2546921/files/"

DEV_INDEX_CSV = CACHE_DIR / "dev_index.csv"
DEV_CACHE_NPY = CACHE_DIR / "dev_256.npy"       # uint8 [N, 256, 256, 3]
CACHE_SIZE = 256                                # short side resized to this, centre crop

PCAM_POOL_SIZE = 20_000                         # patches per pool (paper used 20,000)
PCAM_POOLS = ("pcam_test", "pcam_adapt")        # slide-disjoint: evaluation / adaptation

DS3_VAL_FRACTION = 0.15                         # dataset 3 ships without a validation fold
SEED = 2026
SEEDS = (2026, 2027, 2028, 2029, 2030)          # repeated random splits / few-shot draws

# --------------------------------------------------------------------------- backbones
# kind: "timm" | "hf" | "virchow" | "mambavision"
# gated=True -> needs `huggingface-cli login` with an account that was granted access.
# family is used for grouping in tables and figures.
BACKBONES = {
    # ---- baseline of the earlier study of Paun et al. (2026) (Keras EfficientNetB0 / ImageNet weights)
    "efficientnet_b0": dict(kind="timm", name="tf_efficientnet_b0.in1k", img_size=224,
                            family="CNN (ImageNet)", gated=False),
    # Same network at the earlier study's (Paun et al., 2026) 96 x 96 input size (closest replication).
    "efficientnet_b0_96px": dict(kind="timm", name="tf_efficientnet_b0.in1k", img_size=96,
                                 family="CNN (ImageNet)", gated=False),
    # ---- modern supervised CNN
    "convnextv2_tiny": dict(kind="timm", name="convnextv2_tiny.fcmae_ft_in22k_in1k",
                            img_size=224, family="CNN (ImageNet)", gated=False),
    # ---- hierarchical transformer
    "swinv2_tiny": dict(kind="timm", name="swinv2_tiny_window8_256.ms_in1k", img_size=256,
                        family="Hierarchical transformer (ImageNet)", gated=False),
    "swinv2_base": dict(kind="timm", name="swinv2_base_window8_256.ms_in1k", img_size=256,
                        family="Hierarchical transformer (ImageNet)", gated=False),
    # ---- state-space model
    "mambavision_tiny": dict(kind="mambavision", name="nvidia/MambaVision-T-1K",
                             img_size=224, family="State-space (ImageNet)", gated=False),
    # ---- general-purpose self-supervised ViT
    "dinov2_base": dict(kind="timm", name="vit_base_patch14_dinov2.lvd142m", img_size=224,
                        family="General SSL ViT", gated=False,
                        kwargs=dict(img_size=224)),
    # ---- pathology foundation models
    "phikon": dict(kind="hf", name="owkin/phikon", img_size=224,
                   family="Pathology foundation model", gated=False),
    "phikon_v2": dict(kind="hf", name="owkin/phikon-v2", img_size=224,
                      family="Pathology foundation model", gated=False),
    # added in the second review round (Reviewer 2): a newer, openly licensed pathology foundation model.
    # Midnight-12k (kaiko.ai, 2025), ViT-g/14, MIT licence; embedding = [CLS, mean of patch tokens] as on the model card.
    "midnight": dict(kind="hf_clsmean", name="kaiko-ai/midnight", img_size=224,
                     family="Pathology foundation model", gated=False,
                     mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    "uni": dict(kind="timm", name="hf-hub:MahmoodLab/uni", img_size=224,
                family="Pathology foundation model", gated=True,
                kwargs=dict(init_values=1e-5, dynamic_img_size=True)),
    "uni2_h": dict(kind="timm", name="hf-hub:MahmoodLab/UNI2-h", img_size=224,
                   family="Pathology foundation model", gated=True, uni2=True),
    "h_optimus_0": dict(kind="timm", name="hf-hub:bioptimus/H-optimus-0", img_size=224,
                        family="Pathology foundation model", gated=True,
                        kwargs=dict(init_values=1e-5, dynamic_img_size=False),
                        mean=(0.707223, 0.578729, 0.703617),
                        std=(0.211883, 0.230117, 0.177517)),
    "h_optimus_1": dict(kind="timm", name="hf-hub:bioptimus/H-optimus-1", img_size=224,
                        family="Pathology foundation model", gated=True,
                        kwargs=dict(init_values=1e-5, dynamic_img_size=False),
                        mean=(0.707223, 0.578729, 0.703617),
                        std=(0.211883, 0.230117, 0.177517)),
    "virchow2": dict(kind="virchow", name="hf-hub:paige-ai/Virchow2", img_size=224,
                     family="Pathology foundation model", gated=True),
}

# Backbones run by default (no gated access required).
DEFAULT_BACKBONES = ["efficientnet_b0_96px", "efficientnet_b0", "convnextv2_tiny", "swinv2_tiny", "swinv2_base",
                     "mambavision_tiny", "dinov2_base", "phikon", "phikon_v2"]
# Backbones fine-tuned end to end in the architecture study.
FINETUNE_BACKBONES = ["efficientnet_b0", "swinv2_tiny", "mambavision_tiny"]

# --------------------------------------------------------------------------- evaluation
N_BOOTSTRAP = 1000
FEWSHOT_K = (1, 2, 5, 10, 20, 50, 100)
FEWSHOT_REPEATS = 20
FEWSHOT_REPEATS_MIX = 5                         # repeats for the slower source+shots refit
LOGREG_C_GRID = (0.001, 0.01, 0.1, 1.0, 10.0)
