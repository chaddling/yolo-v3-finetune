import os
import torch

import lightning as L

from yolo.layers import Detect

from typing import List, Tuple

# TODO anchors need to be scaled in data preprocessing/loss calculation
_anchors = [
    (10, 13, 16, 30, 33, 23), # P3/8
    (30, 61, 62, 45, 59, 119), # P4/16
    (116, 90, 156, 198, 373, 326) # P5/32
]
_connect_from = (27, 22, 15)

class PretrainedYOLOModel(torch.nn.Module):
    def __init__(self, 
        weights_dir: str, 
        num_classes: int, # num_classes = # of classes in LVIS
        anchors: List[Tuple[int]] = _anchors, 
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
            ch=[3], # TODO check if this is right
        )
        # These 4 attributes are needed in the BaseModel's _forward_once() instructions
        detect_layer.i = len(self._model.model.model)
        detect_layer.type = "yolo.models.Detect"
        detect_layer.f = _connect_from
        detect_layer.np = sum(x.numel() for x in detect_layer.parameters())

        self._model.append(detect_layer)

    def forward(self, x):
        x = self._model(x)
        return x
    

class ModelWrapper(L.LightningModule):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def training_step(self, batch, batch_idx):
        return
    
    def validation_step(self, batch, batch_idx):
        return
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer