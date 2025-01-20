import yaml

import lightning as L

from data.dataloader import create_dataloader
from yolo.model import PretrainedYOLOModel, ModelWrapper


if __name__ == "__main__":
    with open("./config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    dataloader_config = config.pop("dataloader")
    train_loader = create_dataloader(dataloader_config, split="train")
    #val_loader = create_dataloader(dataloader_config, split="val")

    train_config = config.pop("train")
    model = PretrainedYOLOModel(
        weights_dir="./weights",
        num_classes=1203,
        anchors=train_config["anchors"],
        strides=train_config["strides"],
    )
    wrapped_model = ModelWrapper(model, input_image_size=train_config["input_image_size"])

    trainer = L.Trainer(
        accelerator="gpu",
        limit_train_batches=1,
        max_epochs=5,
    )
    trainer.fit(wrapped_model, train_loader)