import logging
import torch

from typing import List, Tuple
from yolo.loss_utils import (
    get_box_coordinates, 
    obj_anchors_ious, 
    make_grid_idx, 
    make_one_hot_label, 
    standardize_label_box,
    compute_prediction_mask
)


class YOLOLoss(object):
    def __init__(
        self, 
        num_classes: int, 
        all_anchors: List[Tuple[int]],
        anchors_idx: int,
        stride: int,
        input_image_size: int,
        ignore_threshold: float = 0.7,
    ):
        """
        Loss at a single scale/stride. 3 anchors per scale
        """
        self.bce = torch.nn.BCEWithLogitsLoss(reduction="none")
        self.mse = torch.nn.MSELoss(reduction="none")

        self.num_classes = num_classes
        self.stride = stride
        self.n_cells = input_image_size / stride

        anchors = all_anchors[anchors_idx]
        na = len(anchors)
        self.all_anchors = torch.tensor(all_anchors).view(-1, 2)

        self.anchors = torch.tensor(anchors).view(-1, 2)
        self.anchor_masks = [anchors_idx + (i+1) * na for i in range(na // 2)]

        self.ignore_threshold = ignore_threshold        


    def __call__(self, preds: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        bs, na, nx, ny, _ = preds.shape
        stride = torch.tensor(self.stride).to("cuda")
        anchors = self.anchors.to("cuda")

        with torch.no_grad():
            pred_box, pred_obj, pred_class = torch.split(preds, split_size_or_sections=[4, 1, self.num_classes], dim=-1)
            pred_box = get_box_coordinates(pred_box, anchors, stride)

            batch_idx, label = torch.split(label, split_size_or_sections=[1, 5], dim=-1)
            batch_idx = batch_idx.view(-1, 1)

            label_box = label[:, 1:] # (?, 4)
            pred_mask = compute_prediction_mask(pred_box, label_box, self.ignore_threshold)

            ious = obj_anchors_ious(label_box, self.all_anchors)
            best_anchor_idx = torch.argmax(ious, dim=-1).view(-1, 1)
            obj_mask = (best_anchor_idx >= self.anchor_masks[0]) * (best_anchor_idx <= self.anchor_masks[-1])
            obj_mask = obj_mask.squeeze()

            grid_idx = make_grid_idx(label, stride)
            label_class = make_one_hot_label(label, self.num_classes)
            
            batch_idx = batch_idx[obj_mask]
            grid_idx = grid_idx[obj_mask]
            best_anchor_idx = best_anchor_idx[obj_mask]
            best_anchor_idx = torch.fmod(best_anchor_idx, len(self.anchor_masks))

            # indexing into (bs, na, nx, ny)
            # used to create label_obj and used to filter all prediction tensors
            merged_idx = torch.cat([batch_idx, best_anchor_idx, grid_idx], dim=-1).to("cuda")
            label_box = label_box[obj_mask]
            label_class = label_class[obj_mask]

            label_obj = torch.sparse_coo_tensor(
                indices=merged_idx.T,
                values=torch.ones(size=(len(merged_idx), )).to("cuda"),
                size=(bs, na, nx, ny)
            ).to_dense()

            bs_idx, a_idx, x_idx, y_idx = merged_idx.int().T

        # shapes: (bs, na, nx, ny)
        # pred__mask should be set to True, where label_obj == 1?
        # e.g. https://github.com/w86763777/pytorch-simple-yolov3/blob/fb08b7c83493bdad3c0d60f8446a2018827b53a1/yolov3/models/layers.py#L237
        # NOTE maybe fix this, it doesn't handle batch size=1 otherwise.
        object_loss = pred_mask * self.bce(pred_obj.squeeze(), label_obj.squeeze())
        object_loss = object_loss.sum() # sum or mean?

        class_loss = self.bce(pred_class[bs_idx, a_idx, x_idx, y_idx, :], label_class).sum()
        
        box_loss = self.mse(
            pred_box[bs_idx, a_idx, x_idx, y_idx, :], 
            standardize_label_box(label_box, best_anchor_idx, anchors, stride)
        ).sum()

        return object_loss + class_loss + box_loss
    