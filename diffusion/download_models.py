"""
Download and materialize Stable Diffusion 2.1 and ControlNet.
"""

import os
from typing import Tuple
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline, ControlNetModel

from .constants import build_cfg_from_cli


DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _resolve_dtype(dtype_str: str) -> torch.dtype:
    if dtype_str not in DTYPE_MAP:
        raise ValueError(
            f"Unsupported dtype '{dtype_str}'. Expected one of {list(DTYPE_MAP)}."
        )
    return DTYPE_MAP[dtype_str]


def download_sd21_and_controlnet(cfg) -> Tuple[str, str]:
    """
    Download SD 2.1 and ControlNet according to the global config.
    """
    root = Path(cfg.PATHS.ROOT)
    sd21_path = root / cfg.PATHS.SD21
    controlnet_path = root / cfg.PATHS.CONTROLNET
    os.makedirs(sd21_path, exist_ok=True)
    os.makedirs(controlnet_path, exist_ok=True)

    dtype = _resolve_dtype(cfg.MODEL.DTYPE)

    # ------------------------------------------------------------------
    # 1) Download / materialize Stable Diffusion 2.1
    # ------------------------------------------------------------------
    print(f"[download_models] Loading SD 2.1 from '{cfg.MODEL.SD21_REPO}' ...")
    pipe = StableDiffusionPipeline.from_pretrained(cfg.MODEL.SD21_REPO, dtype=dtype)

    print(f"[download_models] Saving SD 2.1 pipeline to '{sd21_path}' ...")
    pipe.save_pretrained(sd21_path)

    # ------------------------------------------------------------------
    # 2) Initialize or download ControlNet
    # ------------------------------------------------------------------
    if cfg.MODEL.INIT_CONTROLNET_FROM_UNET:
        print("[download_models] Initializing ControlNet from SD 2.1 UNet ...")
        controlnet = ControlNetModel.from_unet(pipe.unet, conditioning_channels=1)
    else:
        raise NotImplementedError()

    print(f"[download_models] Saving ControlNet to '{controlnet_path}' ...")
    controlnet.save_pretrained(controlnet_path)

    print("[download_models] Done.")
    return sd21_path, controlnet_path


def main(cli_args=None) -> None:
    cfg = build_cfg_from_cli(args=cli_args)
    download_sd21_and_controlnet(cfg)


if __name__ == "__main__":
    main()
