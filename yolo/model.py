import os
import torch

import lightning as L

from yolo.layers import Detect
from yolo.loss import YOLOLoss
from yolo.loss_utils import get_box_coordinates, make_grid_idx

from torchmetrics.detection import MeanAveragePrecision
from torchvision.ops import batched_nms, box_convert
from typing import List

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

        self.mean_avg_precision = MeanAveragePrecision(box_format="cxcywh")


    def training_step(self, batch, batch_idx):
        inputs, label = batch
        preds = self.model(inputs.float())

        total_loss = 0
        for pred, loss in zip(preds, self.yolo_loss):
            total_loss += loss(pred, label.float())

        self.log("train_loss", total_loss, logger=True)
        return total_loss
    

    def validation_step(self, batch, batch_idx):
        inputs, label = batch
        outputs = self.model(inputs.float())

        bs = inputs.shape[0]
        batch_idx, label = torch.split(label, [1, 5], dim=-1)
        batch_idx = batch_idx.int()
        label_box = label[:, 1:]
        label_class = label[:, [0]]

        targets = [
            {
                "boxes": label_box[torch.where(batch_idx.squeeze() == i)],
                "labels": label_class[torch.where(batch_idx.squeeze() == i)].int().squeeze(),
            }
            for i in torch.arange(bs)
        ]

        preds_box = []
        preds_obj = []
        preds_class = []
        for pred, loss in zip(outputs, self.yolo_loss):
            grid_idx = make_grid_idx(label, loss.stride)
            x_idx, y_idx = grid_idx.int().T
            pred_box, pred_obj, pred_class = torch.split(pred, [4, 1, self.model.num_classes], dim=-1)
            pred_box = get_box_coordinates(pred_box, loss.anchors, loss.stride)
            
            # Localize predictions to batch element + cells
            # NOTE is this cell indexing not done correctly?
            pred_box = pred_box[batch_idx.squeeze(), :, x_idx, y_idx, :]
            pred_obj = pred_obj[batch_idx.squeeze(), :, x_idx, y_idx, :].sigmoid()
            pred_class = pred_class[batch_idx.squeeze(), :, x_idx, y_idx, :].sigmoid()

            preds_box.append(pred_box)
            preds_obj.append(pred_obj)
            preds_class.append(pred_class)

        # predictions per batch element per (anchor, cell) index
        pred_box = torch.cat(preds_box, dim=1)
        pred_obj = torch.cat(preds_obj, dim=1)
        pred_class = torch.cat(preds_class, dim=1)
        pred_conf = pred_class * pred_obj

        preds = []
        for i in range(bs):
            batch_idx_mask = batch_idx.squeeze() == i

            p = pred_box[batch_idx_mask]
            pc = pred_conf[batch_idx_mask]
            # n_obj in this batch element/image
            n_obj, n_anchors, _ = p.shape

            conf_mask = pc.view(-1, self.model.num_classes) > 0.7
            conf_idx, cl = conf_mask.nonzero().T # conf_idx.max() < n_obj * n_anchors, cl is the "label"

            p = p.view(-1, 4)[conf_idx, :]
            pc = pc.view(-1)[conf_idx]
            nms_flat_idx = batched_nms(
                boxes=box_convert(p, "cxcywh", "xyxy"), 
                scores=pc, 
                idxs=cl,
                iou_threshold=0.5
            )
            preds.append(
                {
                    "boxes": p[nms_flat_idx, :],
                    "scores": pc[nms_flat_idx],
                    "labels": cl[nms_flat_idx].int(),
                }
            )

        self.mean_avg_precision.update(preds, targets)


    def on_validation_epoch_end(self):
        self.log_dict(self.mean_avg_precision.compute())
        self.mean_avg_precision.reset()


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        return optimizer