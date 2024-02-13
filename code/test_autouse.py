import pytest
import torch

class TestWithAutouse:
    # BAD: This example uses autouse, which can be confusing.
    @pytest.fixture(autouse=True)
    def no_grad_context(self):
        # this fixture makes it so that if a function uses it, it is automatically called
        # using the torch.no_grad() context
        with torch.no_grad():
            yield

    def test_forward(self):
        model = torch.nn.Linear(3, 5)
        x = torch.randn(10, 3)
        model(x)

class TestFixtureArgument:
    # BETTER: This example does not use autouse, but the no_grad_context fixture
    # is passed as an argument but unused, which can be confusing as well.
    @pytest.fixture
    def no_grad_context(self):
        with torch.no_grad():
            yield

    def test_forward(self, no_grad_context):
        model = torch.nn.Linear(3, 5)
        x = torch.randn(10, 3)
        model(x)


class TestWithUsefixture:
    # BEST: This example uses usefixtures, which makes it clear that the fixture
    # is used for its side effect and not for its return value.
    @pytest.fixture
    def no_grad_context(self):
        with torch.no_grad():
            yield

    @pytest.mark.usefixtures("no_grad_context")
    def test_forward(self):
        model = torch.nn.Linear(3, 5)
        x = torch.randn(10, 3)
        model(x)
