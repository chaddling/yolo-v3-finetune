import os
import glob
import pandas as pd
import torch
import torchvision.transforms.v2 as tv2

import data.transforms

from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
from typing import Dict, List, Optional, Tuple


def create_dataloader(config: Dict) -> DataLoader:
    batch_size = config["dataloader"]["batch_size"]
    dataset_params = config["dataloader"]["dataset"]

    cls = DetectionDataset # TODO configure

    dataset_params.pop("cls")
    dataset = cls(**dataset_params)
    return DataLoader(dataset, batch_size=batch_size), dataset


class BaseDataset(Dataset):
    def __init__(
        self, 
        image_dir: str, 
        label_dir: str,
        transform_params: Optional[Dict[str, Dict]]
    ):
        image_files = glob.glob(os.path.join(image_dir, "*.jpg"))
        self.image_files = list(sorted(image_files))
        label_files = glob.glob(os.path.join(label_dir, "*.txt"))
        self.label_files = list(sorted(label_files))

        self.transforms = None
        if transform_params:
            transforms = self.get_transforms(transform_params)
            self.transforms = tv2.Compose(transforms)

    def get_transforms(self, params: Dict[str, Dict]) -> List[torch.nn.Module]:
        transforms_list = []
        for cls, arg in params.items():
            transform_cls = getattr(data.transforms, cls)
            transforms_list.append(transform_cls(**arg))

        return transforms_list

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int):
        pass


class DetectionDataset(BaseDataset):
    def __init__(
        self, 
        image_dir: str, 
        label_dir: str, 
        transform_params: Optional[Dict[str, Dict]],
    ):
        super().__init__(
            image_dir=image_dir, 
            label_dir=label_dir, 
            transform_params=transform_params,
        )
        self.validate_data_files()
    
    def validate_data_files(self) -> None:
        """ Ensure alignment between image and label files. """
        pass

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor]]]:
        # NOTE width/height can be obtained from metadata, would be needed for inference
        image = read_image(self.image_files[idx])
        
        label = torch.from_numpy(
            pd.read_csv(self.label_files[idx], header=None, sep=" ").values
        )
        if self.transforms:
            image, labels = self.transforms(image, label)

        return image, labels