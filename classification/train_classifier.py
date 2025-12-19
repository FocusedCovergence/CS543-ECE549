import lightning as L
from pathlib import Path
from lightning.pytorch.callbacks import ModelCheckpoint
import torch

from .constants import build_cfg_from_cli
from .lightning_module import FitzpatrickClassifier
from .datamodule import FitzpatrickDataModule


def train_from_scratch(cfg):
    dm = FitzpatrickDataModule(cfg)
    model = FitzpatrickClassifier(cfg)
    model.train()
    torch.set_float32_matmul_precision("medium")

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

    checkpoint_dir = Path(cfg.PATHS.ROOT) / cfg.PATHS.CHECKPOINTS
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    monitor_metric = "val/auc"
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename=cfg.TRAIN.CHECKPOINT_FILENAME,
        save_top_k=1,
        monitor=monitor_metric,
        mode="max",
        save_last=cfg.TRAIN.SAVE_LAST,
    )

    if getattr(cfg.MODEL.PRETRAINED, "ENABLED", False):
        source = cfg.MODEL.PRETRAINED.LOCAL_PATH or (
            f"hf://{cfg.MODEL.PRETRAINED.HF_HUB_ID}/{cfg.MODEL.PRETRAINED.HF_FILENAME}"
            if cfg.MODEL.PRETRAINED.HF_HUB_ID
            else "unspecified"
        )
        print(f"[train_classifier] Initializing classifier from pretrained weights: {source}")

    trainer = L.Trainer(
        accelerator=accelerator,
        devices=cfg.TRAIN.DEVICES,
        precision=precision,
        default_root_dir=cfg.PATHS.ROOT,
        callbacks=[checkpoint_callback],
        num_sanity_val_steps=cfg.VAL.SANITY_CHECK_STEPS,
        max_epochs=cfg.TRAIN.EPOCHS,
    )

    trainer.fit(model, datamodule=dm)
    return trainer


def main(args=None):
    cfg = build_cfg_from_cli(args)
    train_from_scratch(cfg)


if __name__ == "__main__":
    main()
