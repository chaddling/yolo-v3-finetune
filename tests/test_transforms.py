import pytest
import torch

from data.transforms import Resize, RandomHorizontalFlip, RandomVerticalFlip

def test_resize(dummy_image, dummy_label):
    pass

def test_random_horizontal_flip(dummy_image, dummy_label):
    transform = RandomHorizontalFlip(p=1.0)
    _, transformed_label = transform(dummy_image, dummy_label)

    assert torch.equal(
        transformed_label,
        torch.tensor([1.0, 0.2, 0.7, 0.2, 0.2]).reshape(1, 5)
    )

def test_random_vertical_flip(dummy_image, dummy_label):
    transform = RandomVerticalFlip(p=1.0)
    _, transformed_label = transform(dummy_image, dummy_label)

    assert torch.equal(
        transformed_label,
        torch.tensor([1.0, 0.8, 0.3, 0.2, 0.2]).reshape(1, 5)
    )