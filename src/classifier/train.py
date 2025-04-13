from pathlib import Path
import shutil
import yaml

from src.classifier.data import (
    calculate_mean_and_std,
    get_train_transform,
    get_val_transform,
    DigitsDataModule,
)
from src.classifier.model import LightningClassifier

import lightning as L
from lightning.pytorch.loggers import CometLogger
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    EarlyStopping,
)


def show_dataset_info(roots: list[str], splits: list[str], classes: list[str]):
    for root in roots:
        print(f"\nRoot: {Path(root).name}")
        for split in splits:
            print(f"    Split: {split}")
            total = 0
            for cls in classes:
                path = Path(root) / split / cls
                if path.exists():
                    count = len(list(path.glob("*")))
                else:
                    count = 0
                total += count
                print(f"        {cls}: {count} images")
            print(f"        Total: {total} images")


def merge_roots(dst: Path, roots: list[str], splits: list[str], classes: list[str]):

    for split in splits:
        for cls in classes:
            (dst / split / cls).mkdir(parents=True, exist_ok=True)

    for root in roots:
        root = Path(root)
        for split in splits:
            for cls in classes:
                src_dir = root / split / cls
                dst_dir = dst / split / cls
                dst_dir.mkdir(parents=True, exist_ok=True)

                for file in src_dir.iterdir():
                    if file.is_file():
                        target_file = dst_dir / file.name
                        # Avoid overwriting if files with same name exist
                        if target_file.exists():
                            target_file = dst_dir / f"{root.name}_{file.name}"
                        shutil.copy2(file, target_file)


def train(config_file: Path):

    # show_dataset_info(
    #     roots=[
    #         "/home/user/Documents/data/sudoku-assistant/data_digits/hand",
    #         "/home/user/Documents/data/sudoku-assistant/data_digits/print",
    #     ],
    #     splits=["train", "test"],
    #     classes=["1", "2", "3", "4", "5", "6", "7", "8", "9"],
    # )

    # merge_roots(
    #     dst=Path("/home/user/Documents/data/sudoku-assistant/data_digits/merged"),
    #     roots=[
    #         "/home/user/Documents/data/sudoku-assistant/data_digits/hand",
    #         "/home/user/Documents/data/sudoku-assistant/data_digits/print",
    #     ],
    #     splits=["train", "test"],
    #     classes=["1", "2", "3", "4", "5", "6", "7", "8", "9"],
    # )

    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    mean, std = calculate_mean_and_std([config["data"]["train_path"]])

    augs = config["data"]["augmentations"]

    train_transform = get_train_transform(
        size=config["data"]["in_size"],
        degrees=augs["degrees"],
        scale=augs["scale"],
        translate=augs["translate"],
        mean=(mean,),
        std=(std,),
    )
    val_transform = get_val_transform(size=config["data"]["in_size"], mean=(mean,), std=(std,))

    datamodule = DigitsDataModule(
        train_transform,
        val_transform,
        train_path=Path(config["data"]["train_path"]),
        test_path=Path(config["data"]["test_path"]),
        batch_size=config["train"]["batch_size"],
        split_seed=config["train"]["split_seed"],
    )

    model_name = config["model_name"]
    num_classes = config["num_classes"]
    optim_config = config["train"]["optim"]
    model = LightningClassifier(
        model_name, num_classes, optim_config, input_size=config["data"]["in_size"]
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=0.00, patience=10, verbose=False, mode="min"
    )

    comet_logger = CometLogger(project="sudoku-assistant", name=model_name + "_50")

    trainer = L.Trainer(
        callbacks=[
            early_stop_callback,
            LearningRateMonitor("epoch"),
            ModelCheckpoint(
                filename="{epoch}-{val_acc:.4f}",
                save_last=True,
                monitor="val_acc",
                mode="max",
                save_top_k=3,
                save_weights_only=True,
            ),
        ],
        # fast_dev_run=True,
        num_sanity_val_steps=2,
        # profiler="simple",
        max_epochs=config["train"]["max_epochs"],
        accelerator="gpu",
        enable_progress_bar=True,
        logger=comet_logger,
    )
    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)
