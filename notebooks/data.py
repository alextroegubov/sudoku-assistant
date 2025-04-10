import lightning as L

from pathlib import Path


import cv2 as cv
import numpy as np
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset

from sklearn.model_selection import train_test_split


def calculate_mean_and_std(folders: list[str], suffixes={".jpg", ".jpeg", ".png"}, norm=True):
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

    return (mean,), (std,)


def get_train_transform(
    size: int, degrees: int, translate: float, scale: float, mean: tuple[float], std: tuple[float]
):

    tsfrm = transforms.Compose(
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

    return tsfrm


def get_val_transform(size: int, mean: tuple[float], std: tuple[float]):

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
        split_seed: int,
        valid_size: float = 0.1,
    ):
        super().__init__()
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

        self.data = ImageFolder(self.train_path, self.train_transform)
        train_idx, test_idx = train_test_split(
            np.arange(len(self.data)),
            test_size=self.valid_size,
            stratify=self.data.targets,
            random_state=self.split_seed,
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


if __name__ == "__main__":
    mean, std = calculate_mean_and_std(["data_digits/hand/train"])

    train_transform = get_train_transform(
        50, degrees=15, scale=0.1, translate=0.1, mean=mean, std=std
    )
    val_transform = get_val_transform(50, mean=mean, std=std)

    datamodule = DigitsDataModule(
        train_transform,
        val_transform,
        train_path="data_digits/hand/train",
        test_path="data_digits/hand/test",
        batch_size=32,
        split_seed=42,
    )
