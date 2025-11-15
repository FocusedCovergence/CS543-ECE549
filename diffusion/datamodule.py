from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import cv2

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

import lightning as L

from .constants import get_cfg_defaults
from .generate_captions import format_as_caption


class FitzpatrickDataset(Dataset):
    """
    Simple dataset for Fitzpatrick17k.

    Expects:
    - CSV at cfg.DATA.ROOT / cfg.DATA.FITZPATRICK_17K.CSV
    - Images at cfg.DATA.ROOT / cfg.DATA.FITZPATRICK_17K.IMAGES / {md5hash}.jpg

    Returns:
    - image: (3, H, W) tensor in [-1, 1]
    - control: (1, H, W) tensor with Canny edges
    - caption: str
    """

    def __init__(self, df: pd.DataFrame, cfg, train=True):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.train = train
        self.root = Path(cfg.DATA.ROOT)
        self.images_dir = self.root / cfg.DATA.FITZPATRICK_17K.IMAGES

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        # Image path: md5hash.jpg
        md5 = row["md5hash"]
        img_path = self.images_dir / f"{md5}.jpg"

        pil_img = Image.open(img_path).convert("RGB")

        # ------------------------------------------------------------------
        # Shared geometric augmentations for image and control
        # ------------------------------------------------------------------
        size = self.cfg.DATA.IMG_SIZE
        pil_img = pil_img.resize((size, size))

        # Random horizontal flip
        if self.train and self.cfg.DATA.AUG.HORIZONTAL_FLIP:
            if np.random.rand() < 0.5:
                pil_img = pil_img.transpose(Image.FLIP_LEFT_RIGHT)

        # Random rotation
        if (
            self.train
            and self.cfg.DATA.AUG.RANDOM_ROTATION
            and self.cfg.DATA.AUG.RANDOM_ROTATION > 0
        ):
            angle = np.random.uniform(
                -self.cfg.DATA.AUG.RANDOM_ROTATION,
                self.cfg.DATA.AUG.RANDOM_ROTATION,
            )
            pil_img = pil_img.rotate(angle)

        # ------------------------------------------------------------------
        # Image tensor in [-1, 1]
        # ------------------------------------------------------------------
        img_tensor = T.ToTensor()(pil_img)
        img_tensor = T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(img_tensor)

        # ------------------------------------------------------------------
        # Control tensor: Canny edge map from augmented image
        # ------------------------------------------------------------------
        np_img = np.array(pil_img)  # H, W, 3, uint8
        gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(
            gray,
            threshold1=self.cfg.CONTROL.CANNY_LOW,
            threshold2=self.cfg.CONTROL.CANNY_HIGH,
        )
        edges = edges.astype(np.float32) / 255.0  # [0, 1]
        control = torch.from_numpy(edges)[None, ...]  # [1, H, W]

        # Caption from your helper
        caption = format_as_caption(row)

        return img_tensor, control, caption


class FitzpatrickDataModule(L.LightningDataModule):
    """
    Lightning DataModule for Fitzpatrick17k.

    Usage:
        cfg = get_cfg_defaults()
        dm = FitzpatrickDataModule(cfg)
        dm.setup("fit")
        trainer.fit(model, datamodule=dm)
    """

    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg if cfg is not None else get_cfg_defaults()

        self.batch_size = self.cfg.DATA.BATCH_SIZE
        self.num_workers = self.cfg.DATA.NUM_WORKERS

        self._df = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        """
        Assumes CSV + images already exist on disk.
        """
        pass

    def setup(self, stage=None):
        """
        Create train/val/test splits and datasets.

        Splits are random, controlled by cfg.DATA.RANDOM_SEED.
        """
        if self._df is None:
            root = Path(self.cfg.DATA.ROOT)
            csv_path = root / self.cfg.DATA.FITZPATRICK_17K.CSV
            self._df = pd.read_csv(csv_path)

            # Filter to rows that actually have a downloaded image on disk.
            # and which have at least one color annotation
            images_dir = root / self.cfg.DATA.FITZPATRICK_17K.IMAGES
            df = self._df
            mask_valid_md5 = df["md5hash"].notna() & df["md5hash"].astype(
                str
            ).str.strip().ne("")
            df = df[mask_valid_md5].copy()

            def _has_image(md5: str) -> bool:
                return (images_dir / f"{md5}.jpg").is_file()

            existing_mask = df["md5hash"].astype(str).apply(_has_image)
            df = df[existing_mask].reset_index(drop=True)

            self._df = df[
                (df["fitzpatrick_scale"] > 0) | (df["fitzpatrick_centaur"] > 0)
            ]

            # Build splits once
            n = len(self._df)
            indices = np.arange(n)
            rng = np.random.default_rng(self.cfg.DATA.RANDOM_SEED)
            rng.shuffle(indices)

            n_test = int(n * self.cfg.DATA.TEST_SPLIT)
            n_val = int(n * self.cfg.DATA.VAL_SPLIT)
            n_train = n - n_val - n_test

            train_idx = indices[:n_train]
            val_idx = indices[n_train : n_train + n_val]
            test_idx = indices[n_train + n_val :]

            self._train_df = self._df.iloc[train_idx].reset_index(drop=True)
            self._val_df = self._df.iloc[val_idx].reset_index(drop=True)
            self._test_df = self._df.iloc[test_idx].reset_index(drop=True)

        if stage == "fit" or stage is None:
            self.train_dataset = FitzpatrickDataset(
                self._train_df, self.cfg, train=True
            )
            self.val_dataset = FitzpatrickDataset(self._val_df, self.cfg, train=False)

        if stage == "test" or stage is None:
            self.test_dataset = FitzpatrickDataset(self._test_df, self.cfg, train=False)

    # ------------------------------------------------------------------ #
    # Dataloaders
    # ------------------------------------------------------------------ #
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
