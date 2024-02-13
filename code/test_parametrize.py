import pytest
import torch
from transformers import AutoModelForCausalLM


class TestModel:
    @pytest.mark.parametrize("model_name", ["bigscience/bloomz-560m", "gpt2"])
    def test_forward(self, model_name):
        # this test runs once for twice, once for each model
        model = AutoModelForCausalLM.from_pretrained(model_name)
        x = torch.zeros(1, 10, dtype=torch.long)
        model(x)

    @pytest.mark.parametrize("model_name, num_layers", [
        ("bigscience/bloomz-560m", 24),
        ("gpt2", 12),
    ])
    def test_layers(self, model_name, num_layers):
        model = AutoModelForCausalLM.from_pretrained(model_name)
        assert len(model.transformer.h) == num_layers

    @pytest.mark.parametrize("model_name", ["bigscience/bloomz-560m", "gpt2"])
    @pytest.mark.parametrize("method", ["forward", "generate"])
    def test_has_method(self, model_name, method):
        # this test runse four times, once for each model and method
        model = AutoModelForCausalLM.from_pretrained(model_name)
        assert hasattr(model, method)


class TestModel2:
    # here we need to use the "request" fixture, which is a builtin fixture from
    # pytest
    @pytest.fixture(scope="class")
    def model(self, request):
        return AutoModelForCausalLM.from_pretrained(request.param)

    # here we use the "model" fixture and parametrize it with the indirect=True
    # argument
    @pytest.mark.parametrize("method", ["forward", "generate"])
    @pytest.mark.parametrize("model", ["bigscience/bloomz-560m", "gpt2"], indirect=True)
    def test_with_parametrized_fixture(self, model, method):
        assert hasattr(model, method)
