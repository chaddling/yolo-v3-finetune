import torch

from typing import List, Tuple
from yolo.loss_utils import (
    get_box_coordinates, 
    obj_anchors_ious, 
    make_grid_idx, 
    make_one_hot_label, 
    standardize_label_box,
    compute_threshold_mask
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


    def __call__(self, preds: torch.Tensor, label: torch.Tensor):
        box_loss = torch.tensor(0.0)
        class_loss = torch.tensor(0.0)
        object_loss = torch.tensor(0.0)

        bs, na, nx, ny, _ = preds.shape

        pred_box, pred_obj, pred_class = torch.split(preds, split_size_or_sections=[4, 1, self.num_classes], dim=-1)
        batch_idx, label = torch.split(label, split_size_or_sections=[1, 5], dim=-1)
        batch_idx = batch_idx.view(-1, 1)

        pred_box = get_box_coordinates(pred_box, self.anchors, self.stride)
        label_box = label[:, 1:] # (?, 4)
        threshold_mask = compute_threshold_mask(pred_box, label_box, self.ignore_threshold)

        ious = obj_anchors_ious(label_box, self.all_anchors)
        best_anchor_idx = torch.argmax(ious, dim=-1).view(-1, 1)
        obj_mask = (best_anchor_idx >= self.anchor_masks[0]) * (best_anchor_idx <= self.anchor_masks[-1])
        obj_mask = obj_mask.squeeze()

        grid_idx = make_grid_idx(label, self.stride)
        label_class = make_one_hot_label(label, self.num_classes)
        
        batch_idx = batch_idx[obj_mask]
        grid_idx = grid_idx[obj_mask]
        best_anchor_idx = best_anchor_idx[obj_mask]
        best_anchor_idx = torch.fmod(best_anchor_idx, len(self.anchor_masks))

        # indexing into (bs, na, nx, ny)
        # used to create label_obj and used to filter all prediction tensors
        merged_idx = torch.cat([batch_idx, best_anchor_idx, grid_idx], dim=-1)
        
        label_box = label_box[obj_mask]
        label_class = label_class[obj_mask]

        label_obj = torch.sparse_coo_tensor(
            indices=merged_idx.T,
            values=torch.ones(size=(len(merged_idx), )),
            size=(bs, na, nx, ny)
        ).to_dense()
        mask = label_obj.bool()

        # NOTE wtf, label_obj[mask] are suppsoed to be all 1s? that can't be right?
        object_loss = threshold_mask[mask] * self.bce(pred_obj[mask].squeeze(), label_obj[mask])
        object_loss = object_loss.sum() # sum or mean?
        class_loss = self.bce(pred_class[mask], label_class).sum()
        box_loss = self.mse(
            pred_box[mask], 
            standardize_label_box(label_box, best_anchor_idx, self.anchors, self.stride)
        ).sum()

        return box_loss + class_loss + object_loss
    