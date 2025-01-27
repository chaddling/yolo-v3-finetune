import os
import glob
import logging
import pandas as pd
import torch
import torchvision.transforms.v2 as tv2
import torch.nn.functional as F

import data.transforms

from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image, ImageReadMode
from typing import Dict, List, Optional, Tuple

logging.getLogger().setLevel(logging.INFO)


def create_dataloader(dataloader_config: Dict, split: str) -> DataLoader:
    batch_size = dataloader_config["batch_size"]
    dataset_params = dataloader_config["dataset"]

    cls = DetectionDataset # TODO configure

    dataset_params = {k: v for k, v in dataset_params.items() if k != "cls"}
    dataset = cls(**dataset_params, split=split)
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        collate_fn=DetectionDataset.collate
    )


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
        split: str,
        transform_params: Optional[Dict[str, Dict]],
        dataset: str = "lvis",
    ):
        root = os.getenv("HOME")
        image_dir = image_dir.format(
            root=root, dataset=dataset, split=split
        )
        label_dir = label_dir.format(
            root=root, dataset=dataset, split=split
        )

        if split == "val":
            transform_params = {k: v for k, v in transform_params.items() if k == "Resize"}

        super().__init__(
            image_dir=image_dir, 
            label_dir=label_dir, 
            transform_params=transform_params,
        )

        if split in ("train", "val"):
            self.validate_data_files()
    
    def validate_data_files(self) -> None:
        """ Ensure alignment between image and label files. """
        num_image_files = len(self.image_files)
        num_label_files = len(self.label_files)
        logging.info(f"Retrieved {num_image_files} images files and {num_label_files} label files from local directories.")

        image_files = {f.split("/")[-1].rstrip(".jpg"): f for f in self.image_files}
        label_files = {f.split("/")[-1].rstrip(".txt"): f for f in self.label_files}

        reference = label_files
        other = image_files
        if num_image_files < num_label_files:
            reference = image_files
            other = label_files
            logging.info("Using the list of image files/names as reference for indexing.")
        else:
            logging.info("Using the list of label files/names as reference for indexing.")

        keys = [k for k in reference.keys() if k in other.keys()]
        logging.info(f"{len(keys)} matching file names found.")

        self.image_files = [image_files[k] for k in keys]
        self.label_files = [label_files[k] for k in keys]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor]]]:
        # NOTE width/height can be obtained from metadata, would be needed for inference
        image = read_image(self.image_files[idx], mode=ImageReadMode.RGB)
        
        label = torch.from_numpy(
            pd.read_csv(self.label_files[idx], header=None, sep=" ").values
        )
        if self.transforms:
            image, labels = self.transforms(image, label)

        return image, labels
    
    @staticmethod
    def collate(batch):
        """
        returns label in shape (?, 6). ? is the number of objects in this batch
        """
        images, labels = zip(*batch)
        labels = list(labels)
        for i, label in enumerate(labels):
            labels[i] = F.pad(label, (1, 0), value=i)

        return (torch.stack(images, dim=0), torch.cat(labels, dim=0))