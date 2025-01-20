import os
import torch

import lightning as L

from yolo.layers import Detect
from yolo.loss import YOLOLoss

from typing import List, Tuple

# TODO anchors need to be scaled in loss calculation
CONNECT_FROM = (27, 22, 15)

class PretrainedYOLOModel(torch.nn.Module):
    def __init__(self, 
        weights_dir: str, 
        num_classes: int, # num_classes = # of classes in LVIS
        anchors: List[List[int]], # TODO anchors need to be scaled in loss calculation
        strides: List[int],
        training: bool = True
    ):
        super().__init__()

        # Load custom weights, downloaded to `weights_dir`
        # TODO add function to get the weights
        # NOTE it seems like by default these loaded parameters are frozen
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
            ch=[256, 512, 1024], # NOTE These are hardcoded to input size = 512
        )
        # These attributes are needed in the BaseModel's _forward_once() instructions
        detect_layer.stride = strides
        detect_layer.i = len(self._model.model.model)
        detect_layer.type = "yolo.models.Detect"
        detect_layer.f = CONNECT_FROM
        detect_layer.np = sum(x.numel() for x in detect_layer.parameters())

        self._model.model.model.append(detect_layer)

        self.num_classes = num_classes
        self.anchors = anchors
        self.strides = strides

    def forward(self, x):
        x = self._model(x)
        return x
    

class ModelWrapper(L.LightningModule):
    def __init__(self, model: torch.nn.Module, input_image_size: int):
        super().__init__()
        self.model = model
        self.yolo_loss = [
            YOLOLoss(
                num_classes=model.num_classes,
                all_anchors=model.anchors, 
                anchors_idx=i, 
                stride=s,
                input_image_size=input_image_size,
            ) for i, s in enumerate(model.strides)
        ]

    def training_step(self, batch, batch_idx):
        inputs, label = batch
        preds = self.model(inputs.float())

        total_loss = 0
        for pred, loss in zip(preds, self.yolo_loss):
            total_loss += loss(pred, label.float())

        return total_loss
    
    # def validation_step(self, batch, batch_idx):
    #     return
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        return optimizer