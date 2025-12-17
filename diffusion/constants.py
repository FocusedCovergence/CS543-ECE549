"""
Global configuration for the diffusion project.
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
_C.PATHS.SD21 = "sd21"  # Local copy of SD 2.1 weights
_C.PATHS.CONTROLNET = "controlnet"  # Local copy / init of ControlNet
_C.PATHS.CHECKPOINTS = "checkpoints"  # Directory for Lightning checkpoints

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
# Model
_C.MODEL = CN()
# Hugging Face repo for the Stable Diffusion model to use.
_C.MODEL.SD_REPO = "stabilityai/stable-diffusion-2-1-base"
# Maximum tokenized prompt length for the text encoder (77 for SD 2.1).
_C.MODEL.MAX_PROMPT_LENGTH = 77
# Data type for model weights when loading from disk / Hugging Face.
# One of {"float32", "float16", "bfloat16"}.
_C.MODEL.DTYPE = "float32"
_C.MODEL.INIT_CONTROLNET_FROM_UNET = True

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
# Paths
_C.DATA = CN()
_C.DATA.ROOT = "data"
_C.DATA.FITZPATRICK_17K = CN()
_C.DATA.FITZPATRICK_17K.IMAGES = "Fitzpatrick17k_images"
_C.DATA.FITZPATRICK_17K.CSV = (Path("fitzpatrick17k") / "fitzpatrick17k.csv").as_posix()
# Hyperparameters
_C.DATA.IMG_SIZE = 512  # final square size
_C.DATA.BATCH_SIZE = 8
_C.DATA.NUM_WORKERS = 4
_C.DATA.VAL_SPLIT = 0.1  # fraction of data used for validation
_C.DATA.TEST_SPLIT = 0.1  # fraction of data used for test
_C.DATA.RANDOM_SEED = 42
# Augmentation
_C.DATA.AUG = CN()
_C.DATA.AUG.HORIZONTAL_FLIP = True
_C.DATA.AUG.RANDOM_ROTATION = 0.0  # degrees
# Control signal configuration
_C.CONTROL = CN()
_C.CONTROL.TYPE = "grayscale"
_C.CONTROL.CANNY_LOW = 75
_C.CONTROL.CANNY_HIGH = 150
# CONTROL.TYPE may be one of {"canny", "grayscale"}.
# - "canny": use Canny edges as a single-channel control map (default)
# - "grayscale": use the image's grayscale intensities as the control map
# TODO: add segmentation image control
# TODO: give controlnet the caption

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
_C.TRAIN.N_DEVICES = 2  # or None if no accelerator
_C.TRAIN.CHECKPOINT_FILENAME = "controlnet-{epoch:03d}-{step}"
_C.TRAIN.SAVE_LAST = True
_C.TRAIN.OPTIMIZER = "adamw"
_C.TRAIN.MOMENTUM = 0.9

# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------
_C.VAL = CN()
_C.VAL.SANITY_CHECK_STEPS = 0
_C.VAL.INFERENCE_STEPS = 25
_C.VAL.GUIDANCE_SCALE = 7.5

# -----------------------------------------------------------------------------
# INFERENCE
# -----------------------------------------------------------------------------
_C.INFERENCE = CN()
_C.INFERENCE.DEVICE = "cuda"
_C.INFERENCE.STEPS = 25
_C.INFERENCE.GUIDANCE_SCALE = 7.5


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
