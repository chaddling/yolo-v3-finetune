import torch
import torch.nn.functional as F

from torchvision.ops import box_iou, box_convert
from typing import List, Tuple

from data.utils import make_classification_label, make_objectness_label, make_grid_idx, make_one_hot_label, obj_anchors_ious


# TODO anchors need to be scaled in data preprocessing/loss calculation
all_anchors = [
    (10, 13, 16, 30, 33, 23), # P3/8
    (30, 61, 62, 45, 59, 119), # P4/16
    (116, 90, 156, 198, 373, 326) # P5/32
]


def obj_anchors_ious(    
    label_box: torch.Tensor, 
    all_anchors: torch.Tensor,
):
    xy = label_box[:, 0:2]
    label_rect = F.pad(xy, (2, 0))
    anchor_rect = F.pad(all_anchors.view(-1, 2), (2,0))

    ious = box_iou(
        boxes1=box_convert(label_rect, "cxcywh", "xyxy"),
        boxes2=box_convert(anchor_rect, "cxcywh", "xyxy"),
    )
    return ious


def make_one_hot_label(label: torch.Tensor, num_classes: int):
    c = label[:, 0].squeeze()
    one_hot = torch.eye(n=num_classes)
    return one_hot[c.int()]


def make_grid_idx(label: torch.Tensor, stride: int):
    stride = torch.tensor(stride).float()
    xy = label[:, 1:3]
    xy = torch.div(xy - torch.fmod(xy, stride), stride)
    return xy


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

        Loss terms:
        1. MSE for bounding box (size = 1)
        2. BCE for confidence score / IoU (size = 1)
        3. BCE for classification for class (size = # of classes)
        """
        self.bce = torch.nn.BCEWithLogitsLoss(reduction="none")
        self.mse = torch.nn.MSELoss(reduction="none")

        self.num_classes = num_classes
        self.stride = stride
        self.n_cells = input_image_size / stride

        anchors = all_anchors[anchors_idx]
        na = len(anchors)
        self.all_anchors = torch.tensor(all_anchors).view(-1, 2)
        self.anchors = torch.cat([torch.tensor(x) for x in anchors]).view(-1, 2)
        self.anchor_masks = (anchors_idx + (i+1) * na for i in range(na // 2))

        self.ignore_threshold = ignore_threshold        


    def get_box(self, pred_box):
        """
        pred_box: (bs, na, nx, ny, 4)
        output: (bs, na, nx, ny, 4)
        """
        box = torch.zeros_like(pred_box)
        bs, na, nx, ny = pred_box.shape[0:4]

        grid = torch.zeros((nx, ny))
        idx = torch.cat(torch.where(grid == 0)).view(2, -1).T.view(1, 1, nx, ny, 2) # could probably be cached
        offset = idx.repeat(bs, na, 1, 1, 1)

        anchors = self.anchors.view(1, na, 1, 1, 2)
        anchors = anchors.repeat(bs, 1, nx, ny, 1)

        box[..., 0:2] = pred_box[..., 0:2].sigmoid() + offset * self.stride
        box[..., 2:4] = anchors * torch.exp(pred_box[..., 2:4])
 
        return box


    def compute_threshold_mask(
        self, 
        pred_box: torch.Tensor, 
        label_box: torch.Tensor, 
    ) -> torch.Tensor:
        """
        Returns a boolean mask computed by overlapping the predictions in each cell + anchor
        with the ground truth bounding box.

        For each object + anchor/cell pair, the best IoUs below a threshold are kept.

        pred_box: (bs, na, nx, ny, 4)
        label_box: (?, 4), ? is the # of objects in the batch

        :return: Boolean tensor of shape (bs, na, nx, ny)
        """

        bs, na, nx, ny, _ = pred_box.shape

        ious = box_iou(
            boxes1=box_convert(pred_box.view(-1, 4), "cxcywh", "xyxy"),
            boxes2=box_convert(label_box.view(-1, 4), "cxcywh", "xyxy"),
        ) # (bs * na * nx * ny, ?)
        best_ious = ious.amax(dim=-1).view(bs, na, nx, ny)
        mask = best_ious < self.ignore_threshold

        return mask
    

    def standardize_label_box(self, label_box: torch.Tensor, best_idx: torch.Tensor) -> torch.Tensor:
        """
        Standardizes box coordinates. (x, y) is measured as the offset to
        nearest cell + normalized by stride. (w, h) are exponents.
        """
        x, y, w, h = label_box.T # shapes = (?, 1)
        shape = label_box.T.shape
        s = torch.tensor(self.stride).float()

        x = torch.div(torch.fmod(x, s), s)
        y = torch.div(torch.fmod(y, s), s)
        w = torch.log(w / self.anchors[best_idx, 0] + 1e-16)
        h = torch.log(h / self.anchors[best_idx, 1] + 1e-16)
        return torch.cat([x, y, w, h]).view(shape).T


    def __call__(self, preds: torch.Tensor, label: torch.Tensor):
        box_loss = torch.tensor(0.0)
        class_loss = torch.tensor(0.0)
        object_loss = torch.tensor(0.0)

        bs, na, nx, ny, _ = preds.shape

        pred_box, pred_obj, pred_class = torch.split(preds, split_size_or_sections=[4, 1, self.num_class], dim=-1)
        batch_idx, label = torch.split(label, split_size_or_sections=[1, 5], dim=-1)
        batch_idx = batch_idx.view(-1, 1)

        pred_box = self.get_box(pred_box)
        label_box = label[:, 2:] # (?, 4)
        threshold_mask = self.compute_threshold_mask(pred_box, label_box)

        ious = obj_anchors_ious(label_box, self.all_anchors)
        best_anchor_idx = torch.argmax(ious, dim=-1).view(-1, 1)
        # Used to filter out the ? objects in the batch
        obj_mask = (best_anchor_idx >= self.anchor_masks[0]) * (best_anchor_idx <= self.anchor_masks[-1])

        grid_idx = make_grid_idx(label, self.stride)
        label_class = make_one_hot_label(label, batch_idx, self.num_classes)
        
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

        object_loss = threshold_mask[mask] * self.bce(pred_obj[mask], label_obj)        
        class_loss = self.bce(pred_class[mask], label_class)
        box_loss = self.mse(pred_box[mask], self.standardize_label_box(label_box, best_anchor_idx))

        return box_loss + class_loss + object_loss
