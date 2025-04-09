import lightning as L

from pathlib import Path

import numpy as np
from torchvision import transforms
from torchvision.datasets import ImageFolder, Subset
from torch.utils.data import DataLoader, ConcatDataset

from sklearn.model_selection import train_test_split


def get_train_transform(
    size: int, degrees: int, translate: float, scale: float, mean: np.ndarray, std: np.ndarray
):

    tsfrm = transforms.Compose(
        [
            transforms.RandomAffine(
                degrees=degrees,
                translate=translate,
                scale=scale,
            ),
            transforms.ToTensor(),
            transforms.Grayscale(),
            transforms.Resize((size, size)),
            transforms.Normalize(mean, std),
        ]
    )

    return tsfrm


def get_val_transform(size: int, mean: np.ndarray, std: np.ndarray):

    tsfrm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Grayscale(),
            transforms.Resize((size, size)),
            transforms.Normalize(mean, std),
        ]
    )

    return tsfrm


class DigitsDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_transform,
        val_transform,
        train_path: Path,
        test_path: Path,
        batch_size: int,
    ):
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.train_path = train_path
        self.test_path = test_path
        self.batch_size = batch_size

    def prepare_data(self):

        if not (self.train_path.exists() and self.train_path.is_dir()):
            raise ValueError(f"{self.train_path} does not exist or is not a directory")

        if not (self.test_path.exists() and self.test_path.is_dir()):
            raise ValueError(f"{self.test_path} does not exist or is not a directory")

    def setup(self, stage=None):
        self.num_classes = 0
        self.dims = 0

        self.data = ImageFolder(self.train_path, self.train_transform)
        train_idx, test_idx = train_test_split(
            np.arange(len(self.data)), stratify=self.data.targets
        )
        self.train_dataset = Subset(self.data, train_idx)
        self.val_dataset = Subset(self.data, test_idx)

        self.test_dataset = ImageFolder(self.test_path, self.val_transform)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
