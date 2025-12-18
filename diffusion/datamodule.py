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


DDI_SKIN_TONE_DESCRIPTIONS = {
    12: "light (Fitzpatrick I-II)",
    34: "medium (Fitzpatrick III-IV)",
    56: "dark (Fitzpatrick V-VI)",
}


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
        # Control tensor: Canny edge map from augmented image or grayscale
        # ------------------------------------------------------------------
        np_img = np.array(pil_img)  # H, W, 3, uint8
        control_map = _compute_control_map(np_img, self.cfg)

        control = torch.from_numpy(control_map)[None, ...]  # [1, H, W]
        caption = format_as_caption(row)
        return img_tensor, control, caption


class DDIDataset(Dataset):
    """DDI evaluation dataset with ControlNet conditioning."""

    def __init__(self, df: pd.DataFrame, cfg, train=False):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.train = train
        root = Path(cfg.DATA.ROOT)
        self.images_dir = root / cfg.DATA.DDI.DIR
        self.img_col = cfg.DATA.DDI.IMG_COLUMN
        self.skin_tone_col = cfg.DATA.DDI.SKIN_TONE_COLUMN
        self.malignant_col = cfg.DATA.DDI.MALIGNANT_COLUMN
        self.disease_col = cfg.DATA.DDI.DISEASE_COLUMN

    def __len__(self):
        return len(self.df)

    @staticmethod
    def _as_bool(value) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes"}
        return bool(value)

    def _format_caption(self, row):
        tone_value = row.get(self.skin_tone_col)
        tone_value = int(tone_value) if pd.notna(tone_value) else None
        tone_desc = (
            DDI_SKIN_TONE_DESCRIPTIONS.get(tone_value, "an unspecified skin tone")
            if tone_value is not None
            else "an unspecified skin tone"
        )
        disease = row.get(self.disease_col, "a skin condition")
        disease = (
            disease if isinstance(disease, str) and disease else "a skin condition"
        )
        malignancy = "malignant" if self._as_bool(row[self.malignant_col]) else "benign"
        if tone_value is not None:
            tone_phrase = f" (group {tone_value})"
        else:
            tone_phrase = ""
        return (
            f"A dermatology image from the DDI dataset showing {disease} ({malignancy}) "
            f"on a patient with {tone_desc}{tone_phrase}."
        )

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_name = row[self.img_col]
        img_path = self.images_dir / img_name
        pil_img = Image.open(img_path).convert("RGB")

        size = self.cfg.DATA.IMG_SIZE
        pil_img = pil_img.resize((size, size))

        if self.train and self.cfg.DATA.AUG.HORIZONTAL_FLIP:
            if np.random.rand() < 0.5:
                pil_img = pil_img.transpose(Image.FLIP_LEFT_RIGHT)

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

        img_tensor = T.ToTensor()(pil_img)
        img_tensor = T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(img_tensor)

        np_img = np.array(pil_img)
        control_map = _compute_control_map(np_img, self.cfg)

        control = torch.from_numpy(control_map)[None, ...]
        caption = self._format_caption(row)
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
        self._ddi_df = None

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
            if self._ddi_df is None:
                self._ddi_df = self._load_ddi_dataframe()
            self.test_dataset = DDIDataset(self._ddi_df, self.cfg, train=False)

    def _load_ddi_dataframe(self) -> pd.DataFrame:
        root = Path(self.cfg.DATA.ROOT)
        ddi_dir = root / self.cfg.DATA.DDI.DIR
        csv_path = ddi_dir / self.cfg.DATA.DDI.CSV
        df = pd.read_csv(csv_path)
        img_col = self.cfg.DATA.DDI.IMG_COLUMN

        df = df[df[img_col].notna()].copy()
        df[img_col] = df[img_col].astype(str).str.strip()
        df = df[df[img_col].ne("")].copy()

        def _has_image(name: str) -> bool:
            return (ddi_dir / name).is_file()

        exists_mask = df[img_col].apply(_has_image)
        return df[exists_mask].reset_index(drop=True)

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
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )


def _compute_control_map(np_img: np.ndarray, cfg) -> np.ndarray:
    """Return a single-channel control map per CONTROL.TYPE."""
    control_type = cfg.CONTROL.TYPE.lower()

    if control_type == "canny":
        gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(
            gray,
            threshold1=cfg.CONTROL.CANNY_LOW,
            threshold2=cfg.CONTROL.CANNY_HIGH,
        )
        control_map = edges.astype(np.float32) / 255.0
    elif control_type == "grayscale":
        gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        control_map = gray
    elif control_type == "sobel":
        gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        ksize = cfg.CONTROL.SOBEL_KERNEL
        if ksize % 2 == 0 or ksize <= 0:
            raise ValueError(f"SOBEL_KERNEL must be a positive odd number, got {ksize}")
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=ksize)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=ksize)
        grad_mag = cv2.magnitude(grad_x, grad_y)
        if cfg.CONTROL.SOBEL_NORMALIZE:
            max_val = grad_mag.max()
            if max_val > 0:
                grad_mag = grad_mag / max_val
        control_map = grad_mag
    else:
        raise ValueError(f"Unsupported control type: {cfg.CONTROL.TYPE}")

    return control_map.astype(np.float32)
