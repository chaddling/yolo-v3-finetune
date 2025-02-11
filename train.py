import yaml

import lightning as L

from data.dataloader import create_dataloader
from yolo.model import PretrainedYOLOModel, ModelWrapper

from lightning.pytorch.loggers import TensorBoardLogger


if __name__ == "__main__":
    with open("./config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    dataloader_config = config.pop("dataloader")
    train_loader = create_dataloader(dataloader_config, split="train")
    val_loader = create_dataloader(dataloader_config, split="val")

    train_config = config.pop("train")
    model = PretrainedYOLOModel(
        weights_dir="./weights",
        num_classes=train_config["num_classes"],
        anchors=train_config["anchors"],
        strides=train_config["strides"],
    )
    wrapped_model = ModelWrapper(model, input_image_size=train_config["input_image_size"])

    trainer = L.Trainer(
        accelerator="gpu",
        limit_train_batches=1000,
        limit_val_batches=10,
        max_epochs=10,
        log_every_n_steps=100,
        logger=TensorBoardLogger(save_dir="tensorboard")
    )
    trainer.fit(wrapped_model, train_loader, val_loader)