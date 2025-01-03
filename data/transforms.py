import torch

import torchvision.transforms.v2 as tv2

from typing import Optional, Union, Tuple


class Resize(torch.nn.Module):
    def __init__(
        self, 
        size: Union[int, Tuple[int]], 
        interpolation: Union[tv2.InterpolationMode, int] = tv2.InterpolationMode.BILINEAR,
        max_size: Optional[int] = None
    ):
        super().__init__(self)
        self.transform = tv2.Resize(
            size=size,
            interpolation=interpolation,
            max_size=max_size,
        )

    def forward(self, image: torch.Tensor, label: torch.Tensor):
        # TODO need to resize the bounding box in `label`
        return self.transform(image), label
    

class RandomHorizontalFlip(torch.nn.Module):
    def __init__(self, p: float):
        super().__init__(self)
        self.transform = tv2.RandomHorizontalFlip(p=p)

    def forward(self, image: torch.Tensor, label: torch.Tensor):
        label[:, 2] = 1 - label[:, 2]
        return self.transform(image), label
    

class RandomVerticalFlip(torch.nn.Module):
    def __init__(self, p: float):
        super().__init__(self)
        self.transform = tv2.RandomVerticalFlip(p=p)

    def forward(self, image: torch.Tensor, label: torch.Tensor):
        label[:, 1] = 1 - label[:, 1]
        return self.transform(image), label


class ColorJitter(torch.nn.Module):
    def __init__(
        self, 
        brightness: float = 0.0, 
        contrast: float = 0.0, 
        hue: float = 0.0, 
        saturation:float = 0.0, 
    ):
        super().__init__(self)
        self.transform = tv2.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            hue=hue,
            saturation=saturation,
        )

    def forward(self, image: torch.Tensor, label: torch.Tensor):
        return self.transform(image), label