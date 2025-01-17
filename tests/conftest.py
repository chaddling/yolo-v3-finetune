import pytest
import torch


@pytest.fixture
def dummy_label() -> torch.Tensor:
    return torch.tensor([1.0, 50.0, 50.0, 100.0, 100.0]).reshape((1, 5))
