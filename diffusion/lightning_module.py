"""PyTorch Lightning module for training ControlNet on top of SD 2.1.

This module expects that:
- Stable Diffusion 2.1 has been downloaded to ``cfg.PATHS.SD21``.
- A ControlNet model has been materialized at ``cfg.PATHS.CONTROLNET``
"""

from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW, SGD
import lightning as L
from diffusers import StableDiffusionPipeline, ControlNetModel, DDPMScheduler
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance

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

        sd_pipe = StableDiffusionPipeline.from_pretrained(
            sd21_path, torch_dtype=weight_dtype, safety_checker=None
        )
        self.vae = sd_pipe.vae
        self.text_encoder = sd_pipe.text_encoder
        self.tokenizer = sd_pipe.tokenizer
        self.unet = sd_pipe.unet
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            sd21_path, subfolder="scheduler"
        )

        self.controlnet = ControlNetModel.from_pretrained(
            controlnet_path, torch_dtype=weight_dtype
        )

        # Latent downsampling factor of the VAE (e.g., 8 for 512x512 -> 64x64).
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        self.fid = FrechetInceptionDistance(normalize=True).to("cpu")
        self.kid = KernelInceptionDistance(normalize=True).to("cpu")
        self.fid.set_dtype(torch.float32)
        self.kid.set_dtype(torch.float32)

        self._freeze_module(self.vae)
        self._freeze_module(self.text_encoder)
        self._freeze_module(self.unet)

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
        """Normalize batch format from the DataModule into tensors."""
        if isinstance(batch, dict):
            pixel_values = batch["pixel_values"]
            controlnet_cond = batch["controlnet_cond"]
            input_ids = batch["input_ids"]
            attention_mask = batch.get("attention_mask", None)
            return pixel_values, controlnet_cond, input_ids, attention_mask

        pixel_values, controlnet_cond, captions = batch
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

    @torch.no_grad()
    def generate_images(
        self, pixel_values, controlnet_cond, input_ids, attention_mask=None
    ):
        batch_size, _, height, width = pixel_values.shape

        # Prepare conditioning
        encoder_hidden_states = self.encode_text(input_ids, attention_mask)
        controlnet_cond = controlnet_cond.to(self.device, dtype=self.weight_dtype)

        latents = torch.randn(
            batch_size,
            self.unet.in_channels,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
            device=self.device,
            dtype=self.weight_dtype,
        )

        self.noise_scheduler.set_timesteps(self.cfg.VAL.INFERENCE_STEPS)

        for t in self.noise_scheduler.timesteps:
            # ControlNet: get additional residuals
            down_block_res_samples, mid_block_res_sample = self.controlnet(
                latents,
                t,
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=controlnet_cond,
                return_dict=False,
            )

            # UNet denoising with ControlNet residuals
            unet_out = self.unet(
                latents,
                t,
                encoder_hidden_states=encoder_hidden_states,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
            )
            noise_pred = unet_out.sample

            latents = self.noise_scheduler.step(noise_pred, t, latents).prev_sample

        # Decode latents to images in [0, 1]
        latents = latents / self.vae.config.scaling_factor
        images = self.vae.decode(latents).sample
        images = (images / 2.0 + 0.5).clamp(0.0, 1.0)
        return images

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
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=self.cfg.DATA.BATCH_SIZE,
        )
        return loss

    def on_validation_epoch_start(self):
        self.fid.reset()
        self.fid.to("cpu")
        self.kid.reset()
        self.kid.to("cpu")

    def validation_step(self, batch, batch_idx):
        pixel_values, controlnet_cond, input_ids, attention_mask = self._prepare_batch(
            batch
        )

        pixel_values = pixel_values.to(self.device, dtype=self.weight_dtype)
        controlnet_cond = controlnet_cond.to(self.device, dtype=self.weight_dtype)
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        # Generate images with the current model.
        gen_images = self.generate_images(
            pixel_values=pixel_values,
            controlnet_cond=controlnet_cond,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        real_images = (pixel_values / 2.0 + 0.5).clamp(0.0, 1.0)
        real_images = real_images.detach().cpu().to(torch.float32)
        fake_images = gen_images.clamp(0.0, 1.0)
        fake_images = fake_images.detach().cpu().to(torch.float32)

        self.fid.update(real_images, real=True)
        self.fid.update(fake_images, real=False)
        self.kid.update(real_images, real=True)
        self.kid.update(fake_images, real=False)

        return {}

    def on_validation_epoch_end(self):
        """Compute and log FID / KID for the full validation epoch."""
        fid_score = self.fid.compute()
        kid_score = self.kid.compute()
        self.log("val_fid", fid_score, prog_bar=True, on_epoch=True, sync_dist=False)
        self.log("val_kid", kid_score, prog_bar=False, on_epoch=True, sync_dist=False)

    def configure_optimizers(self):
        lr = self.cfg.TRAIN.LEARNING_RATE
        beta1 = self.cfg.TRAIN.ADAM_BETA1
        beta2 = self.cfg.TRAIN.ADAM_BETA2
        weight_decay = self.cfg.TRAIN.ADAM_WEIGHT_DECAY
        eps = self.cfg.TRAIN.ADAM_EPS
        momentum = self.cfg.TRAIN.MOMENTUM

        match self.cfg.TRAIN.OPTIMIZER.lower():
            case "adamw":
                optimizer = AdamW(
                    self.controlnet.parameters(),
                    lr=lr,
                    betas=(beta1, beta2),
                    weight_decay=weight_decay,
                    eps=eps,
                )
            case "sgd":
                optimizer = SGD(self.controlnet.parameters(), lr=lr, momentum=momentum)

        return optimizer
