from pathlib import Path


from timm import create_model
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms


import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import DeviceStatsMonitor
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks import ModelCheckpoint

from data import DigitsDataModule, get_train_transform, get_val_transform, calculate_mean_and_std


class LightningClassifier(L.LightningModule):
    def __init__(self, model_name: str, num_classes: int):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.num_classes = num_classes

        self.model = create_model(
            self.model_name, pretrained=True, num_classes=self.num_classes, in_chans=1
        )

        self.example_input_array = torch.randn((32, 1, 50, 50))

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

        _, preds = torch.max(outputs, 1)

        acc = (preds == labels).sum().item() / labels.size(0)

        self.log_dict({"val_loss": loss, "val_acc": acc}, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self.model(images)
        loss = F.cross_entropy(outputs, labels)

        _, preds = torch.max(outputs, 1)

        acc = (preds == labels).sum().item() / labels.size(0)

        self.log_dict({"test_loss": loss, "test_acc": acc})

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=0.3,
            patience=5,
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


if __name__ == "__main__":

    mean, std = calculate_mean_and_std(["data_digits/hand/train"])

    train_transform = get_train_transform(
        50, degrees=15, scale=0.1, translate=0.1, mean=mean, std=std
    )
    val_transform = get_val_transform(50, mean=mean, std=std)

    datamodule = DigitsDataModule(
        train_transform,
        val_transform,
        train_path=Path("data_digits/hand/train"),
        test_path=Path("data_digits/hand/test"),
        batch_size=32,
        split_seed=42,
    )

    save_path = "checkpoints"
    model_name = "mobilenetv2_100.ra_in1k"
    num_classes = 9
    model = LightningClassifier(model_name, num_classes)

    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=0.00, patience=10, verbose=False, mode="min"
    )

    trainer = L.Trainer(
        default_root_dir=save_path,
        callbacks=[
            early_stop_callback,
            DeviceStatsMonitor(),
            LearningRateMonitor("epoch"),
            ModelCheckpoint(
                filename='{epoch}-{val_loss:.2f}',
                save_last=True,
                monitor='val_loss',
                save_top_k=3,
                save_weights_only=True,
            ),
        ],
        # fast_dev_run=True,
        num_sanity_val_steps=2,
        # profiler="simple",
        max_epochs=30,
        accelerator='gpu',
        enable_progress_bar=True,

    )
    trainer.fit(model=model, datamodule=datamodule)

    trainer.test(model, datamodule=datamodule)

# model = LightningClassifier.load_from_checkpoint(PATH)
# model.freeze()

# x = some_images_from_cifar10()
# predictions = model(x)


# checkpoint = ...
# checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)
# print(checkpoint["hyper_parameters"])
