"""Lightning module for Fitzpatrick17k classification."""

from collections.abc import Sequence
from pathlib import Path
from typing import Optional, Tuple

import lightning as L
import torch
from torch import nn
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
from torchvision import models

from .constants import get_cfg_defaults


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


class FocalLossMulticlass(nn.Module):
    """Multi-class focal loss mirroring the notebook implementation."""

    def __init__(self, alpha: Sequence[float], gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        if len(alpha) == 0:
            raise ValueError("alpha must contain at least one class weight.")
        alpha_tensor = torch.tensor(alpha, dtype=torch.float32)
        self.register_buffer("alpha", alpha_tensor)
        self.gamma = gamma
        self.reduction = reduction
        self.ce_loss = nn.CrossEntropyLoss(reduction="none")

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce_loss(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha[targets] * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        if self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class FitzpatrickClassifier(L.LightningModule):
    """LightningModule for Fitzpatrick17k classification tasks."""

    def __init__(self, cfg=None) -> None:
        super().__init__()
        if cfg is None:
            cfg = get_cfg_defaults()
        self.cfg = cfg
        self.weight_dtype = _resolve_dtype(cfg.MODEL.DTYPE)

        self.model, self._head_module = self._build_model(cfg.MODEL.MODEL_NAME.lower())
        self._maybe_load_pretrained_weights()
        self.model = self.model.to(dtype=self.weight_dtype)
        self.criterion = self._build_loss()

        num_classes = cfg.MODEL.NUM_CLASSES
        self.train_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_acc = MulticlassAccuracy(num_classes=num_classes)
        self.test_acc = MulticlassAccuracy(num_classes=num_classes)
        self.train_f1 = MulticlassF1Score(num_classes=num_classes)
        self.val_f1 = MulticlassF1Score(num_classes=num_classes)
        self.test_f1 = MulticlassF1Score(num_classes=num_classes)

    # ------------------------------------------------------------------
    # Model / loss builders
    # ------------------------------------------------------------------
    def _build_model(self, name: str) -> Tuple[nn.Module, nn.Module]:
        if name == "resnet18":
            weights = self._resolve_weights(
                self.cfg.MODEL.RESNET18_WEIGHTS, models.ResNet18_Weights
            )
            model = models.resnet18(weights=weights)
            in_features = model.fc.in_features
            model.fc = self._build_classifier_head(in_features)
            head_module = model.fc
        elif name == "vit_b_16":
            weights = self._resolve_weights(
                self.cfg.MODEL.VIT_B_16_WEIGHTS, models.ViT_B_16_Weights
            )
            model = models.vit_b_16(weights=weights)
            in_features = self._infer_vit_hidden_dim(model)
            model.heads = self._build_classifier_head(in_features)
            head_module = model.heads
        else:
            raise ValueError(
                f"Unsupported architecture '{name}'. "
                "Expected one of {'resnet18', 'vit_b_16'}."
            )

        if self.cfg.MODEL.FREEZE_BACKBONE:
            self._freeze_backbone(model, head_module)
        return model, head_module

    def _build_classifier_head(self, in_features: int) -> nn.Module:
        layers: list[nn.Module] = []
        drop_prob = float(self.cfg.MODEL.CLASSIFIER_DROPOUT)
        if drop_prob > 0:
            layers.append(nn.Dropout(drop_prob))
        layers.append(nn.Linear(in_features, self.cfg.MODEL.NUM_CLASSES))
        if len(layers) == 1:
            return layers[0]
        return nn.Sequential(*layers)

    def _build_loss(self) -> nn.Module:
        name = self.cfg.LOSS.NAME.lower()
        if name == "cross_entropy":
            return nn.CrossEntropyLoss(
                label_smoothing=float(self.cfg.LOSS.LABEL_SMOOTHING)
            )
        if name == "focal":
            alpha = self.cfg.LOSS.FOCAL_ALPHA
            if len(alpha) != self.cfg.MODEL.NUM_CLASSES:
                raise ValueError(
                    "LOSS.FOCAL_ALPHA must contain one weight per class. "
                    f"Got {len(alpha)} for {self.cfg.MODEL.NUM_CLASSES} classes."
                )
            return FocalLossMulticlass(
                alpha=alpha,
                gamma=float(self.cfg.LOSS.FOCAL_GAMMA),
                reduction=self.cfg.LOSS.REDUCTION,
            )
        raise ValueError(f"Unsupported loss '{self.cfg.LOSS.NAME}'.")

    @staticmethod
    def _infer_vit_hidden_dim(model: nn.Module) -> int:
        heads = model.heads
        if hasattr(heads, "head") and hasattr(heads.head, "in_features"):
            return heads.head.in_features
        children = list(heads.children())
        if children and hasattr(children[-1], "in_features"):
            return children[-1].in_features
        raise ValueError("Unable to infer ViT head dimension.")

    @staticmethod
    def _resolve_weights(weights_name: Optional[str], enum_cls):
        if weights_name is None:
            return None
        weights_name = str(weights_name)
        if weights_name.lower() in {"", "none"}:
            return None
        if weights_name.upper() == "DEFAULT":
            return enum_cls.DEFAULT
        if not hasattr(enum_cls, weights_name):
            raise ValueError(
                f"Unknown weights '{weights_name}' for enum {enum_cls.__name__}."
            )
        return getattr(enum_cls, weights_name)

    @staticmethod
    def _freeze_backbone(model: nn.Module, head: nn.Module) -> None:
        for param in model.parameters():
            param.requires_grad = False
        for param in head.parameters():
            param.requires_grad = True

    def _maybe_load_pretrained_weights(self) -> None:
        cfg_pre = getattr(self.cfg.MODEL, "PRETRAINED", None)
        if cfg_pre is None or not cfg_pre.ENABLED:
            return

        state_dict = self._load_pretrained_state_dict(cfg_pre)
        if not cfg_pre.STRICT:
            state_dict = self._filter_mismatched_shapes(state_dict)

        missing, unexpected = self.model.load_state_dict(
            state_dict, strict=cfg_pre.STRICT
        )
        if cfg_pre.STRICT and (missing or unexpected):
            raise RuntimeError(
                "Strict loading of pretrained weights failed:\n"
                f"  Missing keys: {missing}\n"
                f"  Unexpected keys: {unexpected}"
            )
        if missing or unexpected:
            print(
                "Loaded pretrained weights with non-strict matching:\n"
                f"  Missing keys: {missing}\n"
                f"  Unexpected keys: {unexpected}"
            )

    def _load_pretrained_state_dict(self, cfg_pre):
        checkpoint_path = self._resolve_pretrained_checkpoint_path(cfg_pre)
        state_dict = torch.load(checkpoint_path, map_location="cpu")

        if isinstance(state_dict, dict):
            key = cfg_pre.STATE_DICT_KEY
            if key:
                if key not in state_dict:
                    raise KeyError(
                        f"STATE_DICT_KEY='{key}' not found in checkpoint keys {list(state_dict.keys())[:10]}"
                    )
                state_dict = state_dict[key]
            elif "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]

        prefix = cfg_pre.STATE_DICT_PREFIX
        if prefix and isinstance(state_dict, dict):
            stripped = {}
            for k, v in state_dict.items():
                if k.startswith(prefix):
                    stripped[k[len(prefix) :]] = v
            if not stripped:
                raise ValueError(
                    f"STATE_DICT_PREFIX='{prefix}' did not match any keys in the checkpoint."
                )
            state_dict = stripped

        return state_dict

    def _filter_mismatched_shapes(self, state_dict: dict) -> dict:
        """Drop checkpoint entries whose tensor shapes do not match the current model."""
        model_state = self.model.state_dict()
        filtered: dict[str, torch.Tensor] = {}
        dropped: list[str] = []
        for key, value in state_dict.items():
            if (
                key in model_state
                and isinstance(value, torch.Tensor)
                and model_state[key].shape != value.shape
            ):
                dropped.append(key)
                continue
            filtered[key] = value

        if dropped:
            print(
                "[FitzpatrickClassifier] Dropped mismatched pretrained tensors:",
                dropped,
            )
        return filtered

    @staticmethod
    def _resolve_pretrained_checkpoint_path(cfg_pre):
        if cfg_pre.LOCAL_PATH:
            checkpoint_path = Path(cfg_pre.LOCAL_PATH)
            if not checkpoint_path.is_file():
                raise FileNotFoundError(
                    f"Pretrained weights not found at LOCAL_PATH='{checkpoint_path}'."
                )
            return checkpoint_path

        if cfg_pre.HF_HUB_ID:
            try:
                from huggingface_hub import hf_hub_download
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ImportError(
                    "huggingface_hub is required to download weights from the Hub. "
                    "Install it via `pip install huggingface_hub`."
                ) from exc

            cache_dir = cfg_pre.CACHE_DIR or None
            filename = cfg_pre.HF_FILENAME or "pytorch_model.bin"
            downloaded = hf_hub_download(
                repo_id=cfg_pre.HF_HUB_ID,
                filename=filename,
                cache_dir=cache_dir,
            )
            return Path(downloaded)

        raise ValueError(
            "MODEL.PRETRAINED.ENABLED is True but neither LOCAL_PATH nor HF_HUB_ID was provided."
        )

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)

    def _shared_step(self, batch):
        images, labels = batch
        images = images.to(self.device, dtype=self.weight_dtype)
        labels = labels.to(self.device)
        logits = self(images)
        loss = self.criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, labels

    def training_step(self, batch, batch_idx):
        loss, preds, labels = self._shared_step(batch)
        self.train_acc(preds, labels)
        self.train_f1(preds, labels)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/f1", self.train_f1, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, labels = self._shared_step(batch)
        self.val_acc(preds, labels)
        self.val_f1(preds, labels)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/f1", self.val_f1, on_step=False, on_epoch=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        loss, preds, labels = self._shared_step(batch)
        self.test_acc(preds, labels)
        self.test_f1(preds, labels)
        self.log("test/loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, sync_dist=True)
        self.log("test/f1", self.test_f1, on_step=False, on_epoch=True, sync_dist=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        if isinstance(batch, (tuple, list)):
            images = batch[0]
        else:
            images = batch
        images = images.to(self.device, dtype=self.weight_dtype)
        logits = self(images)
        return torch.softmax(logits, dim=1)

    # ------------------------------------------------------------------
    # Optimizer / scheduler
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = self._build_optimizer(params)
        scheduler = self._build_scheduler(optimizer)

        if scheduler is None:
            return optimizer
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def _build_optimizer(self, params):
        opt_name = self.cfg.TRAIN.OPTIMIZER.lower()
        lr = self.cfg.TRAIN.LEARNING_RATE
        weight_decay = self.cfg.TRAIN.ADAM_WEIGHT_DECAY

        if opt_name == "adamw":
            return AdamW(
                params,
                lr=lr,
                betas=(self.cfg.TRAIN.ADAM_BETA1, self.cfg.TRAIN.ADAM_BETA2),
                eps=self.cfg.TRAIN.ADAM_EPS,
                weight_decay=weight_decay,
            )
        if opt_name == "sgd":
            return SGD(
                params,
                lr=lr,
                momentum=self.cfg.TRAIN.MOMENTUM,
                weight_decay=weight_decay,
                nesterov=True,
            )
        raise ValueError(
            f"Unsupported optimizer '{self.cfg.TRAIN.OPTIMIZER}'. Expected 'adamw' or 'sgd'."
        )

    def _build_scheduler(self, optimizer):
        name = self.cfg.TRAIN.LR_SCHEDULER.lower()
        if name == "none":
            return None
        if name == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.cfg.TRAIN.COSINE_T_MAX,
                eta_min=self.cfg.TRAIN.COSINE_MIN_LR,
            )
            return {"scheduler": scheduler, "interval": "epoch"}
        raise ValueError(
            f"Unsupported LR scheduler '{self.cfg.TRAIN.LR_SCHEDULER}'. Expected 'cosine' or 'none'."
        )
