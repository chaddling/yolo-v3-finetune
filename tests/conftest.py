import pytest
import torch

from typing import Tuple

@pytest.fixture
def dummy_label() -> torch.Tensor:
    return torch.tensor([1.0, 50.0, 50.0, 100.0, 100.0]).reshape((1, 5))

@pytest.fixture
def image_size() -> int:
    return 512

@pytest.fixture
def stride() -> int:
    return 8

@pytest.fixture
def anchors() -> torch.Tensor:
    return torch.tensor([[10.0, 20.0], [30.0, 40.0], [40.0, 50.0]])

@pytest.fixture
def num_classes() -> int:
    return 420

@pytest.fixture
def num_objects() -> int:
    return 69

@pytest.fixture
def ignore_threshold() -> float:
    return 0.5

@pytest.fixture
def pred_index_shape(image_size, stride) -> Tuple[int]:
    """
    dimensions: (batch_size, num_anchors, cell_x, cell_y)
    """
    n_cells = image_size // stride
    return (8, 3, n_cells, n_cells)