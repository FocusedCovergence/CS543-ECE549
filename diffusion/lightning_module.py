"""PyTorch Lightning module for training ControlNet on top of SD 2.1.

This module expects that:
- Stable Diffusion 2.1 has been downloaded to ``cfg.PATHS.SD21``.
- A ControlNet model has been materialized at ``cfg.PATHS.CONTROLNET``
"""

from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW

import lightning as L
from diffusers import StableDiffusionPipeline, ControlNetModel, DDPMScheduler

from .constants import get_cfg_defaults


DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _resolve_dtype(dtype_str):
    if dtype_str not in DTYPE_MAP:
        raise ValueError(
            f"Unsupported dtype '{dtype_str}'. Expected one of {list(DTYPE_MAP)}."
        )
    return DTYPE_MAP[dtype_str]


class ControlNetLightningModule(L.LightningModule):
    """
    Lightning module wrapping SD 2.1 + ControlNet.
    """

    def __init__(self, cfg) -> None:
        super().__init__()

        if cfg is None:
            cfg = get_cfg_defaults()
        self.cfg = cfg
        sd21_path = Path(cfg.PATHS.ROOT) / cfg.PATHS.SD21
        controlnet_path = Path(cfg.PATHS.ROOT) / cfg.PATHS.CONTROLNET

        # ------------------------------------------------------------------
        # Model components: load from local SD 2.1 and ControlNet directories
        # ------------------------------------------------------------------
        weight_dtype = _resolve_dtype(cfg.MODEL.DTYPE)
        self.weight_dtype = weight_dtype

        # Load SD 2.1 pipeline from disk.
        sd_pipe = StableDiffusionPipeline.from_pretrained(
            sd21_path, torch_dtype=weight_dtype, safety_checker=None
        )

        self.vae = sd_pipe.vae
        self.text_encoder = sd_pipe.text_encoder
        self.tokenizer = sd_pipe.tokenizer
        self.unet = sd_pipe.unet

        # Load ControlNet, which will be the only trainable component.
        self.controlnet = ControlNetModel.from_pretrained(
            controlnet_path, torch_dtype=weight_dtype
        )

        # Noise scheduler (DDPM, as used during SD 2.1 training).
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            sd21_path, subfolder="scheduler"
        )

        # Freeze SD 2.1 components; only ControlNet learns.
        self._freeze_module(self.vae)
        self._freeze_module(self.text_encoder)
        self._freeze_module(self.unet)

        # Save hyperparameters for Lightning (ignore large modules).
        self.save_hyperparameters({"cfg": cfg})

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    @staticmethod
    def _freeze_module(module: torch.nn.Module) -> None:
        for p in module.parameters():
            p.requires_grad = False

    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode images to latents using the frozen VAE."""

        pixel_values = pixel_values.to(self.device, dtype=self.weight_dtype)
        with torch.no_grad():
            latents_dist = self.vae.encode(pixel_values).latent_dist
            latents = latents_dist.sample()
            latents = latents * self.vae.config.scaling_factor
        return latents

    def encode_text(self, input_ids, attention_mask=None):
        """Encode text tokens into embeddings using the frozen text encoder."""

        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        with torch.no_grad():
            outputs = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            # CLIP text encoder returns (last_hidden_state, pooled_output, ...)
            text_embeddings = outputs[0]
        return text_embeddings

    def _prepare_batch(self, batch):
        """Normalize batch format from the DataModule into tensors.

        Supports either:
        - A tuple/list: (pixel_values, controlnet_cond, captions)
        - A dict with pre-tokenized fields:
          {"pixel_values", "controlnet_cond", "input_ids", "attention_mask"}
        """
        if isinstance(batch, dict):
            pixel_values = batch["pixel_values"]
            controlnet_cond = batch["controlnet_cond"]
            input_ids = batch["input_ids"]
            attention_mask = batch.get("attention_mask", None)
            return pixel_values, controlnet_cond, input_ids, attention_mask

        # Otherwise, assume (pixel_values, controlnet_cond, captions)
        pixel_values, controlnet_cond, captions = batch
        # Tokenize captions using the SD 2.1 tokenizer.
        encoding = self.tokenizer(
            list(captions),
            padding="max_length",
            max_length=self.cfg.MODEL.MAX_PROMPT_LENGTH,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding.input_ids
        attention_mask = encoding.attention_mask
        return pixel_values, controlnet_cond, input_ids, attention_mask

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------
    def forward(self, batch) -> torch.Tensor:
        pixel_values, controlnet_cond, input_ids, attention_mask = self._prepare_batch(
            batch
        )

        # Encode images to latents.
        latents = self.encode_images(pixel_values)

        # Sample noise and timesteps.
        noise = torch.randn_like(latents)
        bsz = latents.size(0)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.num_train_timesteps,
            (bsz,),
            device=self.device,
            dtype=torch.long,
        )

        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        encoder_hidden_states = self.encode_text(input_ids, attention_mask)
        controlnet_cond = controlnet_cond.to(self.device, dtype=self.weight_dtype)

        down_block_res_samples, mid_block_res_sample = self.controlnet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=controlnet_cond,
            return_dict=False,
        )
        unet_output = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
        )

        noise_pred = unet_output.sample
        return noise_pred, noise, latents, timesteps

    def training_step(self, batch, batch_idx):
        noise_pred, noise, latents, timesteps = self(batch)

        # Target depends on the prediction type (epsilon vs v_prediction).
        prediction_type = getattr(
            self.noise_scheduler.config, "prediction_type", "epsilon"
        )
        if prediction_type == "epsilon":
            target = noise
        elif prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            target = noise

        loss = F.mse_loss(noise_pred.float(), target.float())
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        """Configure AdamW optimizer over ControlNet parameters only."""

        lr = self.cfg.TRAIN.LEARNING_RATE
        beta1 = self.cfg.TRAIN.ADAM_BETA1
        beta2 = self.cfg.TRAIN.ADAM_BETA2
        weight_decay = self.cfg.TRAIN.ADAM_WEIGHT_DECAY
        eps = self.cfg.TRAIN.ADAM_EPS

        optimizer = AdamW(
            self.controlnet.parameters(),
            lr=lr,
            betas=(beta1, beta2),
            weight_decay=weight_decay,
            eps=eps,
        )

        return optimizer
