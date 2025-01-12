import torch

from data.transforms import Resize, RandomHorizontalFlip, RandomVerticalFlip

def test_resize(dummy_image, dummy_label):
    size = 512
    transformed_image, transformed_label = Resize(size)(dummy_image, dummy_label)
    assert transformed_image.size() == (3, size, size)

    bounding_box = transformed_label[:, 1:]
    assert bounding_box[:, 0] < size, f"Scaled x is out of bounds (> {size}) after transform."
    assert bounding_box[:, 1] < size, f"Scaled y is out of bounds (> {size}) after transform."
    assert bounding_box[:, 2] < 1, "Normalized w is out of bounds (> 1) after transform."
    assert bounding_box[:, 3] < 1, "Normalized h is out of bounds (> 1) after transform."


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

def test_label_preprocessor():
    pass