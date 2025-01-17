import argparse

import lightning as L

from data.dataloader import create_dataloader
from yolo.model import PretrainedYOLOModel, ModelWrapper


if __name__ == "__main__":
    trainer = L.Trainer()
    trainer.fit()