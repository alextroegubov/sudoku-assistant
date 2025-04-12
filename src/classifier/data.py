"""Module to work with data during training process"""

from pathlib import Path
import cv2 as cv
import numpy as np

from sklearn.model_selection import train_test_split

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset

import lightning as L


def calculate_mean_and_std(folders: list[str], suffixes={".jpg", ".jpeg", ".png"}, norm=True):
    """Calculate mean and std for images in folders. Converts images to grayscale."""
    mean = 0
    std = 0
    total_files = 0

    for folder in folders:
        files = [file for file in Path(folder).rglob("*") if file.suffix in suffixes]

        for file in files:
            img = cv.imread(file, cv.IMREAD_GRAYSCALE)

            mean += img.mean()
            std += img.std()

        total_files += len(files)

    mean /= total_files
    std /= total_files

    if norm:
        mean = mean / 255
        std = std / 255

    return round(mean, 4), round(std, 4)


def get_train_transform(
    mean: tuple[float],
    std: tuple[float],
    size: int,
    degrees: int,
    translate: float,
    scale: float,
):
    """Train torchvision tranform"""

    return transforms.Compose(
        [
            transforms.RandomAffine(
                degrees=degrees,
                translate=(translate, translate),
                scale=(1 - scale, 1 + scale),
            ),
            transforms.ToTensor(),
            transforms.Grayscale(),
            transforms.Resize((size, size)),
            transforms.Normalize(mean, std),
        ]
    )


def get_val_transform(mean: tuple[float], std: tuple[float], size: int):
    """Validation torchvision tranform"""
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Grayscale(),
            transforms.Resize((size, size)),
            transforms.Normalize(mean, std),
        ]
    )


class DigitsDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_transform,
        val_transform,
        train_path: Path,
        test_path: Path,
        batch_size: int,
        split_seed: int,
        valid_size: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.train_path = train_path
        self.test_path = test_path
        self.batch_size = batch_size
        self.split_seed = split_seed
        self.valid_size = valid_size

    def prepare_data(self):

        if not (self.train_path.exists() and self.train_path.is_dir()):
            raise ValueError(f"{self.train_path} does not exist or is not a directory")

        if not (self.test_path.exists() and self.test_path.is_dir()):
            raise ValueError(f"{self.test_path} does not exist or is not a directory")

    def setup(self, stage=None):
        self.num_classes = 0
        self.dims = 0

        if stage == "fit" or stage is None:

            self.data = ImageFolder(self.train_path, self.train_transform)
            train_idx, test_idx = train_test_split(
                np.arange(len(self.data)),
                test_size=self.valid_size,
                stratify=self.data.targets,
                random_state=self.split_seed,
            )
            self.train_dataset = Subset(self.data, train_idx)
            self.val_dataset = Subset(self.data, test_idx)

            self.num_classes = len(self.train_dataset.dataset.class_to_idx.keys())
            self.dims = self.train_dataset[0][0].size

        if stage == "test" or stage is None:
            self.test_dataset = ImageFolder(self.test_path, self.val_transform)

            self.num_classes = len(self.test_dataset.class_to_idx.keys())
            self.dims = self.test_dataset[0][0].size

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
