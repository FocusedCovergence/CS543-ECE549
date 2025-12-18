"""
Global configuration for the classification portion.
"""

from pathlib import Path
import argparse

from yacs.config import CfgNode as CN


_C = CN()

# -----------------------------------------------------------------------------
# File structure
# -----------------------------------------------------------------------------
# Paths for saving / loading models and other artifacts.
_C.PATHS = CN()
_C.PATHS.ROOT = "artifacts"  # Root folder for all artifacts
_C.PATHS.CHECKPOINTS = "classifier_checkpoints"  # Directory for Lightning checkpoints

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
# Model
_C.MODEL = CN()
_C.MODEL.DTYPE = "float32"
_C.MODEL.MODEL_NAME = "resnet18"  # {"resnet18", "vit_b_16"}
_C.MODEL.NUM_CLASSES = 2
_C.MODEL.FREEZE_BACKBONE = False
_C.MODEL.CLASSIFIER_DROPOUT = 0.0
_C.MODEL.RESNET18_WEIGHTS = None
_C.MODEL.VIT_B_16_WEIGHTS = "DEFAULT"
_C.MODEL.PRETRAINED = CN()
_C.MODEL.PRETRAINED.ENABLED = True
_C.MODEL.PRETRAINED.LOCAL_PATH = ""
_C.MODEL.PRETRAINED.HF_HUB_ID = "timm/resnet18.a1_in1k"
_C.MODEL.PRETRAINED.HF_FILENAME = "pytorch_model.bin"
_C.MODEL.PRETRAINED.STATE_DICT_KEY = ""
_C.MODEL.PRETRAINED.STATE_DICT_PREFIX = ""
_C.MODEL.PRETRAINED.STRICT = False
_C.MODEL.PRETRAINED.CACHE_DIR = "artifacts/hf_cache"

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
# Paths
_C.DATA = CN()
_C.DATA.ROOT = "data"
_C.DATA.FITZPATRICK_17K = CN()
_C.DATA.FITZPATRICK_17K.IMAGES = "Fitzpatrick17k_images"
_C.DATA.FITZPATRICK_17K.CSV = (Path("fitzpatrick17k") / "fitzpatrick17k.csv").as_posix()
_C.DATA.LABEL_COLUMN = "three_partition_label"
_C.DATA.DROP_NON_NEOPLASMS = True
_C.DATA.DDI = CN()
_C.DATA.DDI.DIR = "ddidiversedermatologyimages"
_C.DATA.DDI.CSV = "ddi_metadata.csv"
_C.DATA.DDI.IMG_COLUMN = "DDI_file"
_C.DATA.DDI.MALIGNANT_COLUMN = "malignant"
_C.DATA.DDI.SKIN_TONE_COLUMN = "skin_tone"
_C.DATA.DDI.DISEASE_COLUMN = "disease"
_C.DATA.DDI.MALIGNANT_LABEL = "malignant"
_C.DATA.DDI.BENIGN_LABEL = "benign"
# Hyperparameters
_C.DATA.IMG_SIZE = 512  # final square size
_C.DATA.BATCH_SIZE = 128
_C.DATA.NUM_WORKERS = 4
_C.DATA.VAL_SPLIT = 0.1  # fraction of data used for validation
_C.DATA.TEST_SPLIT = 0.1  # fraction of data used for test
_C.DATA.RANDOM_SEED = 42
# Augmentation
_C.DATA.AUG = CN()
_C.DATA.AUG.HORIZONTAL_FLIP = True
_C.DATA.AUG.RANDOM_ROTATION = 0.0  # degrees

# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.LEARNING_RATE = 1e-4
_C.TRAIN.ADAM_BETA1 = 0.9
_C.TRAIN.ADAM_BETA2 = 0.999
_C.TRAIN.ADAM_WEIGHT_DECAY = 1e-2
_C.TRAIN.ADAM_EPS = 1e-8
_C.TRAIN.EPOCHS = 30
_C.TRAIN.DEVICE = "cuda"
# int num devices, list of devices ids, or None if no accelerator
_C.TRAIN.DEVICES = [0, 2]
_C.TRAIN.CHECKPOINT_FILENAME = "classifier-{epoch:03d}-{step}"
_C.TRAIN.SAVE_LAST = True
_C.TRAIN.OPTIMIZER = "adamw"
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.LR_SCHEDULER = "cosine"  # {"cosine", "none"}
_C.TRAIN.COSINE_T_MAX = 30
_C.TRAIN.COSINE_MIN_LR = 0.0

# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------
_C.VAL = CN()
_C.VAL.SANITY_CHECK_STEPS = 0

# -----------------------------------------------------------------------------
# Loss
# -----------------------------------------------------------------------------
_C.LOSS = CN()
_C.LOSS.NAME = "cross_entropy"  # {"cross_entropy", "focal"}
_C.LOSS.LABEL_SMOOTHING = 0.0
_C.LOSS.FOCAL_ALPHA = [0.55, 0.45]
_C.LOSS.FOCAL_GAMMA = 3.0
_C.LOSS.REDUCTION = "mean"

# -----------------------------------------------------------------------------
# INFERENCE
# -----------------------------------------------------------------------------
_C.INFERENCE = CN()
_C.INFERENCE.DEVICE = "cuda"


def get_cfg_defaults() -> CN:
    return _C.clone()


def build_cfg_from_cli(args=None) -> CN:
    parser = argparse.ArgumentParser(description="Diffusion config builder")
    parser.add_argument(
        "--config-file",
        type=str,
        default="",
        help="Optional YAML config file to merge into defaults.",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help=(
            "Optional list of KEY=VALUE pairs to override config options, "
            "e.g. MODEL.DTYPE=float32 PATHS.ROOT=./outputs"
        ),
    )

    parsed = parser.parse_args(args=args)

    cfg = get_cfg_defaults()
    if parsed.config_file:
        cfg.merge_from_file(parsed.config_file)
    if parsed.opts:
        cfg.merge_from_list(parsed.opts)

    cfg.freeze()
    return cfg
