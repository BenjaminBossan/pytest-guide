import numpy as np
import pytest
import torch


@pytest.fixture
def set_random_seed():
    torch.manual_seed(0)
    np.random.seed(0)


def test_forward():
    model = torch.nn.Linear(3, 5)
    x = torch.randn(10, 3)
    y = model(x)
    expected = torch.randn(10, 5)
    # this will fail and report the magnitude of the error
    torch.testing.assert_allclose(y, expected)


def test_other_device():
    x = torch.zeros(10, 3)
    y = torch.zeros(10, 3).to("cuda")
    # this will fail because the tensors are on different devices
    torch.testing.assert_allclose(x, y)


@pytest.mark.usefixtures("set_random_seed")
def test_rand():
    # this test works because the fixture sets the random seed
    x = torch.rand(2, 2)
    expected = torch.tensor([[0.4963, 0.7682], [0.0885, 0.1320]])
    torch.testing.assert_allclose(x, expected, rtol=1e-4, atol=1e-4)
