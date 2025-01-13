import pytest
import torch

from data.transforms import Resize, RandomHorizontalFlip, RandomVerticalFlip, LabelPreprocessor


@pytest.mark.parametrize(
    "dummy_image", [
        torch.rand(3, 249, 361),
        torch.rand(3, 361, 249),
        torch.rand(3, 412, 249),
        torch.rand(3, 249, 412),
        torch.rand(3, 248, 612),
    ]
)
def test_resize(dummy_image, dummy_label):
    size = 512
    transformed_image, transformed_label = Resize(size)(dummy_image, dummy_label)
    assert transformed_image.size() == (3, size, size)

    bounding_box = transformed_label[:, 1:]
    assert bounding_box[:, 0] < size, f"Scaled x is out of bounds (> {size}) after transform."
    assert bounding_box[:, 1] < size, f"Scaled y is out of bounds (> {size}) after transform."
    assert bounding_box[:, 2] < size, f"Scaled w is out of bounds (> {size}) after transform."
    assert bounding_box[:, 3] < size, f"Scaled h is out of bounds (> {size}) after transform."


@pytest.mark.parametrize(
    "dummy_image,expected", [
        (torch.rand(3, 512, 512), torch.tensor([1.0, 50.0, 512.0 - 50.0, 100.0, 100.0]).reshape(1, 5))
    ]
)
def test_random_horizontal_flip(dummy_image, dummy_label, expected):
    transform = RandomHorizontalFlip(p=1.0)
    _, transformed_label = transform(dummy_image, dummy_label)

    assert torch.equal(transformed_label, expected)


@pytest.mark.parametrize(
    "dummy_image,expected", [
        (torch.rand(3, 512, 512), torch.tensor([1.0, 512.0 - 50.0, 50.0, 100.0, 100.0]).reshape(1, 5))
    ]
)
def test_random_vertical_flip(dummy_image, dummy_label, expected):
    transform = RandomVerticalFlip(p=1.0)
    _, transformed_label = transform(dummy_image, dummy_label)

    assert torch.equal(
        transformed_label,
        expected,
    )


@pytest.mark.parametrize(
    "input,n_obj,n_class", [
        (torch.tensor(
            [
                [2.0, 50.0, 50.0, 100.0, 100.0],
                [10.0, 90.0, 90.0, 100.0, 100.0],
                [7.0, 210.0, 210.0, 100.0, 100.0],
            ]
        ), 3, 3),
        (torch.tensor(
            [
                [2.0, 50.0, 50.0, 100.0, 100.0],
                [2.0, 90.0, 90.0, 100.0, 100.0],
                [7.0, 210.0, 210.0, 100.0, 100.0],
                [0.0, 110.0, 110.0, 100.0, 100.0],
            ]
        ), 4, 4),
        (torch.tensor(
            [
                [2.0, 90.0, 90.0, 100.0, 100.0],
                [2.0, 90.0, 90.0, 100.0, 100.0],
                [0.0, 110.0, 110.0, 100.0, 100.0],
                [0.0, 110.0, 110.0, 100.0, 100.0],
                [7.0, 210.0, 210.0, 100.0, 100.0],
            ]
        ), 3, 3)
    ]
)
def test_label_preprocessor(label_preprocessor, input, n_obj, n_class):
    label_box = label_preprocessor.make_bbox_label(input)
    assert label_box.shape == (label_preprocessor.max_boxes, 4)

    stride = 32
    n_cells = int(512 / stride)
    label_obj = label_preprocessor.make_objectness_label(input, n_cells, stride)
    assert label_obj.shape == (n_cells, n_cells)
    assert label_obj.sum() == n_obj

    label_class = label_preprocessor.make_classification_label(input, n_cells, stride)
    assert label_class.shape == (n_cells, n_cells, label_preprocessor.num_classes)
    assert label_class.sum() == n_class