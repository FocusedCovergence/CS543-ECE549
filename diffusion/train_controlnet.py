import lightning as L
from pathlib import Path
from lightning.pytorch.callbacks import ModelCheckpoint

from .constants import build_cfg_from_cli
from .lightning_module import ControlNetLightningModule
from .datamodule import FitzpatrickDataModule


def train_from_scratch(cfg):
    dm = FitzpatrickDataModule(cfg)
    model = ControlNetLightningModule(cfg)
    model.controlnet.train()

    device = cfg.TRAIN.DEVICE.lower()
    if device in {"cuda", "gpu"}:
        accelerator = "gpu"
    elif device == "mps":
        accelerator = "mps"
    elif device == "cpu":
        accelerator = "cpu"
    else:
        accelerator = "auto"

    dtype = cfg.MODEL.DTYPE.lower()
    if dtype == "float16":
        precision = "16-mixed"
    elif dtype == "bfloat16":
        precision = "bf16-mixed"
    else:
        precision = "32-true"

    devices = 1 if accelerator in {"gpu", "mps"} else None

    checkpoint_dir = Path(cfg.PATHS.ROOT) / cfg.PATHS.CHECKPOINTS
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename=cfg.TRAIN.CHECKPOINT_FILENAME,
        save_top_k=1,
        monitor="train_loss",
        mode="min",
        save_last=cfg.TRAIN.SAVE_LAST,
    )

    trainer = L.Trainer(
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        default_root_dir=cfg.PATHS.ROOT,
        callbacks=[checkpoint_callback],
        num_sanity_val_steps=cfg.VAL.SANITY_CHECK_STEPS,
        max_epochs=cfg.TRAIN.EPOCHS,
    )

    trainer.fit(model, datamodule=dm)

    controlnet_save_dir = Path(cfg.PATHS.ROOT) / cfg.PATHS.CONTROLNET
    controlnet_save_dir.mkdir(parents=True, exist_ok=True)
    model.controlnet.save_pretrained(controlnet_save_dir)

    return trainer


def main(args=None):
    cfg = build_cfg_from_cli(args)
    train_from_scratch(cfg)


if __name__ == "__main__":
    main()
