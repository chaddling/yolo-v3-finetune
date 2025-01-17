import os
import torch

import lightning as L

from yolo.layers import Detect
from yolo.loss import YOLOLoss

from typing import List, Tuple

# TODO anchors need to be scaled in loss calculation
ANCHORS = [
    (10, 13, 16, 30, 33, 23), # stride 8
    (30, 61, 62, 45, 59, 119), # stride 16
    (116, 90, 156, 198, 373, 326) # stride 32
]
STRIDE = (8, 16, 32)
CONNECT_FROM = (27, 22, 15)

class PretrainedYOLOModel(torch.nn.Module):
    def __init__(self, 
        weights_dir: str, 
        num_classes: int, # num_classes = # of classes in LVIS
        anchors: List[Tuple[int]] = ANCHORS, 
        training: bool = True
    ):
        super().__init__()

        # Load custom weights, downloaded to `weights_dir`
        # TODO add function to get the weights
        self._model = torch.hub.load(
            "ultralytics/yolov3", 
            "custom", 
            os.path.join(weights_dir, "yolov3.pt"), 
            **{"autoshape": False if training else True} 
        )
        # Removes the top
        self._model.model.model = self._model.model.model[:-1]

        # NOTE we could also probably package this with Lightning module instead
        detect_layer = Detect(
            nc=num_classes, 
            anchors=anchors,
            ch=[len(anchors)],
        )
        # These attributes are needed in the BaseModel's _forward_once() instructions
        detect_layer.stride = STRIDE
        detect_layer.i = len(self._model.model.model)
        detect_layer.type = "yolo.models.Detect"
        detect_layer.f = CONNECT_FROM
        detect_layer.np = sum(x.numel() for x in detect_layer.parameters())

        self._model.append(detect_layer)

    def forward(self, x):
        x = self._model(x)
        return x
    

class ModelWrapper(L.LightningModule):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

        self.num_classes = model.num_classes

        self.strides = (8, 16, 32)

        self.yolo_loss = [
            YOLOLoss(self.num_classes, ANCHORS, i, s) 
            for i, s in enumerate(self.strides)
        ]

    def training_step(self, batch, batch_idx):
        inputs, label = batch
        preds = self.model(inputs)

        total_loss = 0
        for pred, loss in zip(preds, self.yolo_loss):
            total_loss += loss(pred, label)

        return total_loss
    
    def validation_step(self, batch, batch_idx):
        return
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer