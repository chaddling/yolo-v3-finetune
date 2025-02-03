import torch

import torchvision.transforms.v2 as tv2

from math import ceil, floor
from torchvision.transforms.functional import resize, pad, hflip, vflip

from typing import List, Union, Tuple


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
        h, w = image.shape[-2:]
        r = min(self.size / w, self.size / h)

        new_w = int(w * r)
        new_h = int(h * r)

        # size: height then width
        image = resize(image, size=(new_h, new_w))
        pad_w = 0.5 * (self.size - new_w)
        pad_h = 0.5 * (self.size - new_h)

        if (self.size - new_w) % 2 == 0:
            pad_l = int(pad_w)
            pad_r = int(pad_w)
        else:
            pad_l = floor(pad_w)
            pad_r = ceil(pad_w)

        if (self.size - new_h) % 2 == 0:
            pad_t = int(pad_h)
            pad_b = int(pad_h)
        else:
            pad_t = floor(pad_h)
            pad_b = ceil(pad_h)
        
        # padding: left, top, right, bottom
        image = pad(image, padding=(pad_l, pad_t, pad_r, pad_b))

        label[:, 1] = label[:, 1] * r + pad_l
        label[:, 2] = label[:, 2] * r + pad_t
        label[:, 3] = label[:, 3] * r
        label[:, 4] = label[:, 4] * r

        return image, label
    

class RandomHorizontalFlip(torch.nn.Module):
    def __init__(self, p: float):
        super().__init__()
        self.p = p

    def forward(self, image: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor]:
        if torch.rand(1) < self.p:
            image = hflip(image)
            label[:, 1] = image.shape[-1] - label[:, 1]
        return image, label
    

class RandomVerticalFlip(torch.nn.Module):
    def __init__(self, p: float):
        super().__init__()
        self.p = p

    def forward(self, image: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor]:
        if torch.rand(1) < self.p:
            image = vflip(image)
            label[:, 2] = image.shape[-1] - label[:, 2]
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