"""Lightning Model"""

from pathlib import Path

import torch
import torch.nn.functional as F

from torchmetrics.classification import Accuracy, Recall, Precision

from timm import create_model
from timm import scheduler as timm_schedulers

import lightning as L


class LightningClassifier(L.LightningModule):
    def __init__(self, model_name: str, num_classes: int, optim_params: dict, input_size: int):
        super().__init__()
        self.save_hyperparameters()
        self.optim_params = optim_params

        self.model_name = model_name
        self.num_classes = num_classes
        self.input_size = input_size

        self.model = create_model(
            self.model_name, pretrained=True, num_classes=self.num_classes, in_chans=1
        )
        self.example_input_array = torch.randn((32, 1, self.input_size, self.input_size))

        # metrics
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_recall = Recall(task="multiclass", num_classes=num_classes, average="macro")

        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_recall = Recall(task="multiclass", num_classes=num_classes, average="macro")
        self.test_precision = Precision(task="multiclass", num_classes=num_classes, average="macro")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self.model(images)
        loss = F.cross_entropy(outputs, labels)

        self.log("train_loss", loss, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch

        outputs = self.model(images)
        loss = F.cross_entropy(outputs, labels)

        # _, preds = torch.max(outputs, 1)
        # acc = (preds == labels).sum().item() / labels.size(0)

        self.val_acc.update(outputs, labels)
        self.val_recall.update(outputs, labels)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        val_acc = self.val_acc.compute()
        val_recall = self.val_recall.compute()

        self.val_acc.reset()
        self.val_recall.reset()

        self.log_dict({"val_acc": val_acc, "val_recall": val_recall}, prog_bar=True)

    def test_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self.model(images)
        loss = F.cross_entropy(outputs, labels)

        # _, preds = torch.max(outputs, 1)
        # acc = (preds == labels).sum().item() / labels.size(0)

        self.test_acc.update(outputs, labels)
        self.test_recall.update(outputs, labels)
        self.test_precision.update(outputs, labels)

        return loss

    def on_test_epoch_end(self):
        test_acc = self.test_acc.compute()
        test_recall = self.test_recall.compute()
        test_precision = self.test_precision.compute()

        self.test_acc.reset()
        self.test_recall.reset()
        self.test_precision.reset()

        self.log_dict(
            {"test_acc": test_acc, "test_recall": test_recall, "test_precision": test_precision},
            prog_bar=True,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.optim_params["lr"],
            weight_decay=self.optim_params["weight_decay"],
        )
        scheduler = timm_schedulers.CosineLRScheduler(
            optimizer,
            t_initial=self.optim_params["t_initial"],
            warmup_lr_init=self.optim_params["warmup_lr_init"],
            warmup_t=self.optim_params["warmup_t"],
            lr_min=self.optim_params["lr_min"],
            cycle_decay=self.optim_params["cycle_decay"],
            cycle_limit=self.optim_params["cycle_limit"],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "val_loss",
                "frequency": 1,
                "strict": True,
            },
        }

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(self.current_epoch)
