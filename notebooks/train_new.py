from pathlib import Path


from timm import create_model
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms


import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping


class LightningClassifier(L.LightningModule):
    def __init__(self, model_name: str, num_classes: int):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model_name = model_name
        self.num_classes = num_classes

        self.model = create_model(
            self.model_name, pretrained=True, num_classes=self.num_classes, in_chans=1
        )

    def forward(self, x):
        return model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self.model(images)
        loss = F.cross_entropy(outputs, labels)

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch

        outputs = self.model(images)
        loss = F.cross_entropy(outputs, labels)

        _, preds = torch.max(outputs, 1)

        acc = (preds == labels).sum().item() / labels.size(0)

        self.log("val_loss", loss)
        self.log("val_acc", acc)

    def test_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self.model(images)
        loss = F.cross_entropy(outputs, labels)

        _, preds = torch.max(outputs, 1)

        acc = (preds == labels).sum().item() / labels.size(0)

        self.log("test_loss", loss)
        self.log("test_acc", acc)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer


dataset = ...
train_loader = ...
model_name = ...
num_classes = ...

test_loader = ...
val_loader = ...

save_path = ...


model = LightningClassifier(model_name, num_classes)

early_stop_callback = EarlyStopping(
    monitor="val_loss", min_delta=0.00, patience=5, verbose=False, mode="min"
)

trainer = L.Trainer(default_root_dir=save_path, callbacks=[early_stop_callback])
trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

trainer.test(model, dataloaders=test_loader)

model = LightningClassifier.load_from_checkpoint(PATH)
model.freeze()

x = some_images_from_cifar10()
predictions = model(x)


checkpoint = ...
checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)
print(checkpoint["hyper_parameters"])
