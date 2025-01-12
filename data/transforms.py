import torch

import torchvision.transforms.v2 as tv2

from torchvision.transforms.functional import resize, pad, hflip, vflip

from typing import List, Union, Tuple

# NOTE do these transforms take into account batch shape?
# Check that this part of the code doesn't receive gradients (it shouldn't)
class LabelPreprocessor(torch.nn.Module):
    def __init__(self, num_classes: int, max_boxes: int, strides: Tuple[int]):
        super().__init__()
        self.num_classes = num_classes
        self.max_boxes = max_boxes
        self.strides = strides

        self.objectness_iou_threshold = 0.5

    def standardize_bbox_label(self, label: torch.Tensor, stride: int, size: int) -> torch.Tensor:
        """
        Standardizes box coordinates: (x, y) are normalized by the box stride. (w, h) are
        normalized by the image size.
        """
        stride = torch.tensor(stride)
        size = torch.tensor(size)

        label_box = label[:, 1:]
        label_box[:, 0] = torch.div(torch.fmod(label_box[:, 0], stride), stride)
        label_box[:, 1] = torch.div(torch.fmod(label_box[:, 1], stride), stride)
        label_box[:, 2] = torch.div(label_box[:, 2], size)
        label_box[:, 3] = torch.div(label_box[:, 4], size)

        return label_box

    def make_bbox_label(self, label: torch.Tensor) -> torch.Tensor:
        """
        Maps input shape (?, 5) -> (max_boxes, 4)
        """
        label_box = torch.zeros(size=(self.max_boxes, 4))
        values = label[:, 1:]
        label_box[0:len(values), :] = values

        return label_box

    def make_objectness_label(self, label: torch.Tensor, n_cells: int, stride: int) -> torch.Tensor:
        """
        Maps input shape (?, 5) -> (n_cells, n_cells)
        """
        stride = torch.tensor(stride)
        # (x, y)
        indices = label[:, 1:3]
        indices = torch.div(indices - torch.fmod(indices, stride), stride)
        indices = indices.int()

        label_obj = torch.sparse_coo_tensor(
            indices=indices,
            values=torch.ones(size=(len(indices),)),
            size=(n_cells, n_cells),
        )
        label_obj = label_obj.to_dense()

        return label_obj

    def make_classification_label(self, label: torch.Tensor, n_cells: int, stride: int) -> torch.Tensor:
        """
        Maps input shape (?, 5) -> (n_cells, n_cells, n_classes)
        """
        stride = torch.tensor(stride)

        # (c, x, y)
        indices = label[:, 0:3]
        indices[:, 1:] = torch.div(indices[:, 1:] - torch.fmod(indices[:, 1:], stride), stride)
        indices = indices.int()
        indices = torch.cat([indices[:, 1:], indices[:, 0:1]], axis=-1) # (x, y, c)

        label_class = torch.sparse_coo_tensor(
            indices=indices.T,
            values=torch.ones(size=(len(indices),)),
            size=(n_cells, n_cells, self.num_classes),
        )
        label_class = label_class.to_dense()

        return label_class

    def forward(self, image: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor]]]:
        labels = []
        size = image.size()[-1]
        for stride in self.strides:
            label_box = self.make_bbox_label(label)

            n_cells = int(size / stride)
            label_obj = self.make_objectness_label(label, n_cells, stride)
            label_class = self.make_classification_label(label, n_cells, stride)
        
            labels.append((label_box, label_obj, label_class))

        return image, labels

class Resize(torch.nn.Module):
    """
    Resizes and pads an input image up to the given size. The bounding box
    is transformed according to the computed rescaling and padding.
    """
    def __init__(
        self, 
        size: Union[int, Tuple[int]], 
    ):
        super().__init__()
        if isinstance(size, tuple):
            assert len(size) == 2, f"Resize() operation must be given a square image shape, but len(size) = {len(size)}."
            assert size[0] == size[1], f"Resize() operation must be given a square image shape, but {size[0]} != {size[1]}"
            size = size[0]

        self.size = size

    def forward(self, image: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        w and h are normalized
        dim(label) = (m, 5) for m objects in the image
        """
        # This could be retrieved from the metadata
        # should use metadata for inference
        w, h = image.shape[-2:]
        r = min(self.size / w, self.size / h)

        new_w = int(w * r)
        new_h = int(h * r)

        # size: height then width
        image = resize(image, size=(new_h, new_w))

        pad_w = int(0.5 * (self.size - new_w))
        pad_h = int(0.5 * (self.size - new_h))

        # padding: left, top, right, bottom
        image = pad(image, padding=(pad_w, pad_h, pad_w, pad_h))

        scale_w = new_w / w
        scale_h = new_h / h

        label[:, 1] = label[:, 1] * scale_w
        label[:, 2] = label[:, 2] * scale_h
        label[:, 3] = label[:, 3] * scale_w
        label[:, 4] = label[:, 4] * scale_h

        return image, label
    

class RandomHorizontalFlip(torch.nn.Module):
    def __init__(self, p: float):
        super().__init__()
        self.p = p

    def forward(self, image: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor]:
        if torch.rand(1) < self.p:
            image = hflip(image)
            label[:, 2] = 1 - label[:, 2]
        return image, label
    

class RandomVerticalFlip(torch.nn.Module):
    def __init__(self, p: float):
        super().__init__()
        self.p = p

    def forward(self, image: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor]:
        if torch.rand(1) < self.p:
            image = vflip(image)
            label[:, 1] = 1 - label[:, 1]
        return image, label


class ColorJitter(torch.nn.Module):
    def __init__(
        self, 
        brightness: float = 0.0, 
        contrast: float = 0.0, 
        hue: float = 0.0, 
        saturation:float = 0.0, 
    ):
        super().__init__()
        self.transform = tv2.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            hue=hue,
            saturation=saturation,
        )

    def forward(self, image: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor]:
        return self.transform(image), label