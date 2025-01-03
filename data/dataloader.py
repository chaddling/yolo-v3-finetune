import os
import glob
import logging
import pandas as pd
import torch
import torchvision.transforms.v2 as tv2

from data.transforms import Resize, ColorJitter, RandomHorizontalFlip, RandomVerticalFlip

from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
from typing import List, Optional, Tuple, Union


def create_dataloader() -> DataLoader:
    pass


class BaseDataset(Dataset):
    def __init__(
        self, 
        image_dir: str, 
        label_dir: str,
        transforms: Optional[List[Union[torch.nn.Module, tv2.Transform]]] = [
            Resize(size=640),
            ColorJitter(brightness=0.1, contrast=0.1, hue=0.1, saturation=0.1), # set in config
            RandomHorizontalFlip(p=0.1),
            RandomVerticalFlip(p=0.1),
        ],
    ):
        image_files = glob.glob(os.path.join(image_dir, "*.jpg"))
        self.image_files = list(sorted(image_files))
        label_files = glob.glob(os.path.join(label_dir, "*.txt"))
        self.label_files = list(sorted(label_files))

        self.transforms = None
        if transforms:
            self.transforms = tv2.Compose(transforms)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        pass


class LVISDataset(BaseDataset):
    def __init__(self, image_dir: str, label_dir: str, transforms):
        super().__init__(
            image_dir=image_dir, 
            label_dir=label_dir, 
            transforms=transforms
        )

        self._validate_data_files()

    
    def _validate_data_files(self) -> None:
        """ Ensure alignment between image and label files. """
        pass

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO image sizes are not uniform, should be the default transform, I think yolov3 uses 640?
        # the transforms also need to be applied to the boxes!
        # TODO width/height can be obtained from metadata
        image = read_image(self.image_files[idx])
        
        # TODO labels' xywh dimensions are not normalized
        # TODO
        label = torch.from_numpy(
            pd.read_csv(self.label_files[idx], header=None, sep=" ").values
        )
        if self.transforms:
            image = self.transforms(image)

        return image, label