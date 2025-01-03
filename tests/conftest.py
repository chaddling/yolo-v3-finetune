import pytest
import torch

@pytest.fixture
def dummy_image() -> torch.Tensor:
    return torch.rand(3, 256, 256)

@pytest.fixture
def dummy_label() -> torch.Tensor:
    return torch.tensor([1.0, 0.2, 0.3, 0.2, 0.2]).reshape((1, 5))