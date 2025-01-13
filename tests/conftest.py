import pytest
import torch

from data.transforms import LabelPreprocessor


@pytest.fixture
def dummy_label() -> torch.Tensor:
    return torch.tensor([1.0, 50.0, 50.0, 100.0, 100.0]).reshape((1, 5))


@pytest.fixture
def label_preprocessor() -> LabelPreprocessor:
    return LabelPreprocessor(
        num_classes=42,
        max_boxes=50,
        strides=(8, 16, 32)
    )