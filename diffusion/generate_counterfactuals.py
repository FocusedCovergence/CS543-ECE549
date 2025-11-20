from pathlib import Path
import re
from typing import Tuple, Optional

import torch
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt

from .constants import get_cfg_defaults
from .lightning_module import load_model_from_checkpoint
from .datamodule import FitzpatrickDataModule
from .generate_captions import FITZPATRICK_TYPES, modify_caption_skin_tone

from diffusers import StableDiffusionControlNetPipeline


# -----------------------------------------------------------------------------
# Global cache for config and pipeline
# -----------------------------------------------------------------------------
_PIPELINE = None
_CFG = None


def _get_cfg():
    global _CFG
    if _CFG is None:
        _CFG = get_cfg_defaults()
    return _CFG


def _build_pipeline(cfg=None) -> StableDiffusionControlNetPipeline:
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE

    cfg = cfg or _get_cfg()

    lit_model = load_model_from_checkpoint(cfg)
    lit_model.eval()

    vae = lit_model.vae
    unet = lit_model.unet
    controlnet = lit_model.controlnet
    text_encoder = lit_model.text_encoder
    tokenizer = lit_model.tokenizer
    scheduler = lit_model.noise_scheduler

    pipe = StableDiffusionControlNetPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=controlnet,
        scheduler=scheduler,
        safety_checker=None,
        feature_extractor=None,
    )

    device = cfg.INFERENCE.DEVICE
    pipe.to(device)

    _PIPELINE = pipe
    return pipe


def generate_counterfactual_from_inputs(
    edge_tensor,
    caption,
    target_fitz,
    cfg=None,
):
    cfg = cfg or _get_cfg()
    pipe = _build_pipeline(cfg)

    control = edge_tensor.clamp(0.0, 1.0)
    if control.dim() == 2:  # HxW
        control = control.unsqueeze(0)  # 1xHxW
    if control.dim() == 3:  # CxHxW
        control = control.unsqueeze(0)  # BxCxHxW, B=1
    control = control.to(device=pipe.device, dtype=pipe.unet.dtype)

    modified_caption = modify_caption_skin_tone(caption, target_fitz)
    generator = torch.Generator(device=pipe.device).manual_seed(cfg.DATA.RANDOM_SEED)

    with torch.inference_mode():
        outputs = pipe(
            prompt=[modified_caption],
            image=control,
            num_inference_steps=cfg.INFERENCE.STEPS,
            guidance_scale=cfg.INFERENCE.GUIDANCE_SCALE,
            generator=generator,
        )

    cf_pil = outputs.images[0]
    return cf_pil, modified_caption


def visualize_validation_index_counterfactuals(idx, cfg=None):
    """Fetch a validation datapoint and generate counterfactuals for all Fitzpatrick types.

    This will:
      1) Load the config and data module.
      2) Fetch (image, control, caption) from the validation dataset at index `idx`.
      3) Generate counterfactuals for Fitzpatrick types 1..6 (including the original).
      4) Display the original image, its edge map, and all counterfactuals using matplotlib.
    """
    cfg = cfg or _get_cfg()

    dm = FitzpatrickDataModule(cfg)
    dm.setup("fit")
    val_ds = dm.val_dataset
    image_tensor, edge_tensor, caption = val_ds[idx]

    img_disp = (image_tensor * 0.5 + 0.5).clamp(0.0, 1.0)
    orig_pil = to_pil_image(img_disp)
    edge_disp = edge_tensor.squeeze(0).cpu().numpy()

    # Generate counterfactuals for all Fitzpatrick types
    counterfactuals = {}
    for fitz in sorted(FITZPATRICK_TYPES.keys()):
        cf_img, cf_caption = generate_counterfactual_from_inputs(
            edge_tensor=edge_tensor, caption=caption, target_fitz=fitz, cfg=cfg
        )
        counterfactuals[fitz] = (cf_img, cf_caption)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # Original image
    ax = axes[0, 0]
    ax.imshow(orig_pil)
    ax.set_title("Original image")
    ax.axis("off")

    # Edge map
    ax = axes[0, 1]
    ax.imshow(edge_disp, cmap="gray")
    ax.set_title("Canny edges")
    ax.axis("off")

    # Counterfactuals: fill remaining 6 slots
    # slots 0,0 and 0,1 are used; start from linear index 2
    fitz_list = sorted(FITZPATRICK_TYPES.keys())
    for i, fitz in (1, 1): enumerate(fitz_list):
        row = (i + 2) // 4
        col = (i + 2) % 4
        cf_img, _ = counterfactuals[fitz]
        ax = axes[row, col]
        ax.imshow(cf_img)
        ax.set_title(f"Fitzpatrick {fitz}")
        ax.axis("off")

    fig.title(caption)
    plt.tight_layout()
    plt.show()

    return fig, counterfactuals


def generate_all_counterfactuals_for_splits(cfg=None):
    cfg = cfg or _get_cfg()
    pipe = _build_pipeline(cfg)

    dm = FitzpatrickDataModule(cfg)
    dm.setup("fit")

    from torch.utils.data import DataLoader

    for split in ("train", "val"):
        if split == "train":
            dataset = dm.train_dataset
        elif split == "val":
            dataset = dm.val_dataset

        df = dataset.df
        md5_list = df["md5hash"].astype(str).tolist()

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=cfg.DATA.BATCH_SIZE,
            shuffle=False,
            num_workers=cfg.DATA.NUM_WORKERS,
            pin_memory=True,
        )

        out_dir = dataset.images_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        global_index = 0
        for batch in dataloader:
            img_batch, control_batch, captions = batch
            batch_size = control_batch.size(0)

            md5_batch = md5_list[global_index : global_index + batch_size]
            global_index += batch_size

            control = control_batch.clamp(0.0, 1.0).to(device=pipe.device, dtype=pipe.unet.dtype)

            if isinstance(captions, tuple):
                captions = list(captions)

            for fitz in sorted(FITZPATRICK_TYPES.keys()):
                modified_prompts = [
                    modify_caption_skin_tone(c, fitz) for c in captions
                ]
                generator = torch.Generator(device=pipe.device).manual_seed(cfg.DATA.RANDOM_SEED + fitz)

                with torch.inference_mode():
                    outputs = pipe(
                        prompt=modified_prompts,
                        image=control,
                        num_inference_steps=cfg.INFERENCE.STEPS,
                        guidance_scale=cfg.INFERENCE.GUIDANCE_SCALE,
                        generator=generator,
                    )

                cf_images = outputs.images
                for i in range(batch_size):
                    md5 = md5_batch[i]
                    cf_img = cf_images[i]
                    out_name = f"{md5}_counterfactual_{fitz}.jpg"
                    out_path = out_dir / out_name
                    cf_img.save(out_path)


def main():
    visualize_validation_index_counterfactuals(1)


if __name__ == "__main__":
    main()