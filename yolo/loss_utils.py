import torch
import torch.nn.functional as F

from torchvision.ops import box_iou, box_convert


def get_box_coordinates(pred_box: torch.Tensor, anchors: torch.Tensor, stride: int):
    """
    Computes box coordinates at each (batch, anchor, cell) index given predictions [tx ty tw th].
    """
    box = torch.zeros_like(pred_box)
    bs, na, nx, ny = pred_box.shape[0:4]

    grid = torch.zeros((nx, ny))
    idx = torch.cat(torch.where(grid == 0)).view(2, -1).T.view(1, 1, nx, ny, 2) # could probably be cached
    offset = idx.repeat(bs, na, 1, 1, 1)
    offset = offset.to("cuda")

    anchors = anchors.view(1, na, 1, 1, 2)
    anchors = anchors.repeat(bs, 1, nx, ny, 1)
    anchors = anchors.to("cuda")

    box[..., 0:2] = pred_box[..., 0:2].sigmoid() + offset * stride
    box[..., 2:4] = anchors * torch.exp(pred_box[..., 2:4])

    return box


def standardize_label_box(label_box: torch.Tensor, best_idx: torch.Tensor, anchors: torch.Tensor, stride: torch.Tensor) -> torch.Tensor:
    """
    Standardizes box coordinates. (x, y) is measured as the offset to the 
    nearest cell and is normalized by stride, while (w, h) are arguments to the exponents.
    """
    x, y, w, h = label_box.T # shapes = (?, 1)
    shape = label_box.T.shape

    x = torch.div(torch.fmod(x, stride), stride)
    y = torch.div(torch.fmod(y, stride), stride)
    w = torch.log(w / anchors[best_idx.squeeze(), 0] + 1e-16)
    h = torch.log(h / anchors[best_idx.squeeze(), 1] + 1e-16)

    return torch.cat([x, y, w, h]).view(shape).T


def compute_prediction_mask(pred_box: torch.Tensor, label_box: torch.Tensor, ignore_threshold: float) -> torch.Tensor:
    """
    Returns a boolean mask computed by overlapping the predictions in each cell
    with the ground truth bounding box. Each cell is responsible for `na` predictions equal
    to the number of anchor boxes.

    For each object + anchor/cell pair, the best IoUs below the specified threshold are kept.
    """
    bs, na, nx, ny, _ = pred_box.shape
    ious = box_iou(
        boxes1=box_convert(pred_box.view(-1, 4), "cxcywh", "xyxy"),
        boxes2=box_convert(label_box.view(-1, 4), "cxcywh", "xyxy"),
    ) # (bs * na * nx * ny, ?)
    best_ious = ious.amax(dim=-1).view(bs, na, nx, ny)
    mask = best_ious < ignore_threshold

    return mask


def obj_anchors_ious(label_box: torch.Tensor, all_anchors: torch.Tensor):
    """
    Computes the IoUs between all objects in a batch vs all anchor boxes
    (across all prediction scales).
    """
    xy = label_box[:, 0:2]
    label_rect = F.pad(xy, (2, 0))
    anchor_rect = F.pad(all_anchors.view(-1, 2), (2,0)).to("cuda")

    ious = box_iou(
        boxes1=box_convert(label_rect, "cxcywh", "xyxy"),
        boxes2=box_convert(anchor_rect, "cxcywh", "xyxy"),
    )
    return ious


def make_one_hot_label(label: torch.Tensor, num_classes: int):
    c = label[:, 0].squeeze()
    one_hot = torch.eye(n=num_classes).to("cuda")
    return one_hot[c.int()]


def make_grid_idx(label: torch.Tensor, stride: int):
    stride = torch.tensor(stride).float()
    xy = label[:, 1:3]
    xy = torch.div(xy - torch.fmod(xy, stride), stride)
    return xy