"""Class to train CNN network"""

import numpy as np
from pathlib import Path
import cv2 as cv
import yaml
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, ConcatDataset

from timm import create_model
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class ClassifierTrainer:
    num_classes: 9

    @staticmethod
    def calculate_mean_and_std(folders: list[str], suffixes={".jpg", ".jpeg", ".png"}):
        mean = 0
        std = 0
        total_files = 0

        for folder in folders:
            files = [file for file in Path(folder).rglob("*") if file.suffix in suffixes]

            for file in files:
                img = cv.imread(file, cv.IMREAD_GRAYSCALE)

                mean += img.mean()
                std += img.std()
                total_files += 1

        mean /= total_files
        std /= total_files

        return mean / 255, std / 255

    def __init__(self, params_file):
        with open(params_file, "r", encoding="utf-8") as f:
            self.params = yaml.load(f, Loader=yaml.SafeLoader)

        self.train_folders = self.params["train_folders"]
        self.test_folders = self.params["test_folders"]

        self.digit_size = self.params["digit_size"]
        self.mean, self.std = ClassifierTrainer.get_mean_and_std(self.train_folders)

        self.batch_size = self.params["batch_size"]
        self.epochs = self.params["epochs"]

        self.model_name = self.params["model_name"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self._best_val_metric = np.inf

    def get_dataset_loader(self, train=True):

        aug_transforms = transforms.Compose(
            [
                transforms.RandomAffine(
                    degrees=self.params["aug"]["degrees"],
                    translate=self.params["aug"]["translate"],
                    scale=self.params["aug"]["scale"],
                )
            ]
        )
        val_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Grayscale(),
                transforms.Resize((self.params["digit_size"], self.params["digit_size"])),
                transforms.Normalize((self.mean), (self.std)),
            ]
        )
        train_transforms = transforms.Compose(
            [
                val_transforms,
                aug_transforms,
            ]
        )

        if train:
            dataset = ConcatDataset(
                ImageFolder(root=f, transform=train_transforms) for f in self.train_folders
            )
        else:
            dataset = ConcatDataset(
                [ImageFolder(root=f, transform=val_transforms) for f in self.test_folders]
            )

        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=train)
        return loader

    def train_epoch(self, model, loader, loss_fn, optimizer):
        model.to(self.device).train()
        total = 0
        total_loss = 0
        total_correct = 0

        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)

            optimizer.zero_grad()
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            total += labels.size(0)
            total_loss += loss.item()
            total_correct += (preds == labels).sum().item()

        return total_loss / total, total_correct / total

    def validate(self, model, loader, loss_fn):
        model.to(self.device).eval()
        total = 0
        total_loss = 0
        total_correct = 0

        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                _, preds = torch.max(outputs, 1)

                total += labels.size(0)
                total_loss += loss.item()
                total_correct += (preds == labels).sum().item()

        return total_loss / total, total_correct / total

    def train(self):
        train_loader = self.get_dataset_loader(train=True)
        test_loader = self.get_dataset_loader(train=False)

        model = create_model(
            self.model_name, pretrained=True, num_classes=self.num_classes, in_chans=1
        )

        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=self.params["lr"])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, factor=self.params["factor"], patience=self.params["patience"]
        )

        for epoch in range(self.epochs):
            train_loss, train_acc = self.train_epoch(model, train_loader, loss_fn, optimizer)
            val_loss, val_acc = self.validate(model, test_loader, loss_fn)

            print(
                f"Epoch [{epoch+1:3}/{self.epochs:3}]:\n"
                f"\tTrain: Loss = {train_loss:.4f}, Acc = {train_acc:.3f}"
                f"\tVal: Loss = {val_loss:.4f}, Acc = {val_acc:.3f}"
            )
            self.save_checkpoint(epoch, val_acc, model)
            scheduler.step(val_loss)

    def save_checkpoint(self, epoch: int, metric: float, model: nn.Module):
        if metric > self._best_val_metric:
            torch.save(model.state_dict(), f"model_{epoch}.pt")
