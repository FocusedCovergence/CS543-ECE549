from pathlib import Path

import numpy as np
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import lightning as L

from .constants import get_cfg_defaults


# ESTABLISH CLASS LABELS

_C = get_cfg_defaults()
_csv = pd.read_csv(Path(_C.DATA.ROOT) / _C.DATA.FITZPATRICK_17K.CSV)
ALL_LABELS = sorted(_csv["label"].dropna().unique().tolist())
THREE_PARTITION_LABELS = sorted(
    _csv["three_partition_label"].dropna().unique().tolist()
)
NINE_PARTITION_LABELS = sorted(
    _csv["nine_partition_label"].dropna().unique().tolist()
)
del _csv, _C


def _get_label_vocab(label_column: str):
    label_column = label_column.lower()
    if label_column == "label":
        return ALL_LABELS
    if label_column == "nine_partition_label":
        return NINE_PARTITION_LABELS
    if label_column == "three_partition_label":
        return THREE_PARTITION_LABELS
    raise ValueError(f"Unknown label source {label_column}")


# DATASET CLASS
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

    def __init__(self, df: pd.DataFrame, cfg, train=True, filter_df=True):
        self.cfg = cfg
        self.train = train
        self.root = Path(cfg.DATA.ROOT)
        self.images_dir = self.root / cfg.DATA.FITZPATRICK_17K.IMAGES
        self.label_col = cfg.DATA.LABEL_COLUMN

        if filter_df:
            df = self.filter_dataframe(df, cfg, images_dir=self.images_dir)

        if cfg.DATA.DROP_NON_NEOPLASMS:
            self.df = df[
                (df["three_partition_label"] != "non-neoplastic")
            ].reset_index(drop=True)
        else:
            self.df = df.reset_index(drop=True)

        self.labels = _get_label_vocab(self.label_col)

    @classmethod
    def filter_dataframe(cls, df: pd.DataFrame, cfg, images_dir=None):
        """Filter rows to those with valid metadata and downloaded images."""
        if images_dir is None:
            root = Path(cfg.DATA.ROOT)
            images_dir = root / cfg.DATA.FITZPATRICK_17K.IMAGES

        df = df.copy()

        md5_series = df["md5hash"]
        valid_md5_mask = md5_series.notna()
        df = df[valid_md5_mask].copy()

        df["md5hash"] = df["md5hash"].astype(str).str.strip()
        df = df[df["md5hash"].ne("")].copy()

        existing_mask = df["md5hash"].apply(
            lambda md5: (images_dir / f"{md5}.jpg").is_file()
        )
        df = df[existing_mask].reset_index(drop=True)

        color_mask = (df["fitzpatrick_scale"] > 0) | (df["fitzpatrick_centaur"] > 0)
        return df[color_mask].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        # Image path: md5hash.jpg
        md5 = row["md5hash"]
        img_path = self.images_dir / f"{md5}.jpg"
        pil_img = Image.open(img_path).convert("RGB")

        # ------------------------------------------------------------------
        # Geometric augmentations for image
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
        # Label
        # ------------------------------------------------------------------
        label = row[self.label_col]
        label = self.labels.index(label)

        return img_tensor, label


class DDIDataset(Dataset):
    """Dataset exposing the DDI evaluation set."""

    def __init__(self, df: pd.DataFrame, cfg):
        self.cfg = cfg
        self.df = df.reset_index(drop=True)
        root = Path(cfg.DATA.ROOT)
        self.images_dir = root / cfg.DATA.DDI.DIR
        self.label_col = cfg.DATA.LABEL_COLUMN
        if self.label_col.lower() != "three_partition_label":
            raise ValueError(
                "DDI testing currently supports LABEL_COLUMN='three_partition_label' only."
            )
        self.labels = _get_label_vocab(self.label_col)
        self.img_col = cfg.DATA.DDI.IMG_COLUMN
        self.malignant_col = cfg.DATA.DDI.MALIGNANT_COLUMN
        self.malignant_label = cfg.DATA.DDI.MALIGNANT_LABEL
        self.benign_label = cfg.DATA.DDI.BENIGN_LABEL

    def __len__(self):
        return len(self.df)

    @staticmethod
    def _as_bool(value) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes"}
        return bool(value)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_name = row[self.img_col]
        img_path = self.images_dir / img_name
        pil_img = Image.open(img_path).convert("RGB")

        size = self.cfg.DATA.IMG_SIZE
        pil_img = pil_img.resize((size, size))

        img_tensor = T.ToTensor()(pil_img)
        img_tensor = T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(img_tensor)

        is_malignant = self._as_bool(row[self.malignant_col])
        label_str = self.malignant_label if is_malignant else self.benign_label
        if label_str not in self.labels:
            raise ValueError(
                f"Label '{label_str}' not found in label vocabulary {self.labels}"
            )
        label = self.labels.index(label_str)

        return img_tensor, label


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
            df = pd.read_csv(csv_path)
            images_dir = root / self.cfg.DATA.FITZPATRICK_17K.IMAGES
            self._df = FitzpatrickDataset.filter_dataframe(
                df, self.cfg, images_dir=images_dir
            )

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
                self._train_df, self.cfg, train=True, filter_df=False
            )
            self.val_dataset = FitzpatrickDataset(
                self._val_df, self.cfg, train=False, filter_df=False
            )

        if stage == "test" or stage is None:
            if self._ddi_df is None:
                self._ddi_df = self._load_ddi_dataframe()
            self.test_dataset = DDIDataset(self._ddi_df, self.cfg)

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
