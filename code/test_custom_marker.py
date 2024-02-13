import pytest
from transformers import AutoModelForCausalLM


@pytest.fixture(scope="module")
def model():
    return AutoModelForCausalLM.from_pretrained("bigscience/bloomz-560m")


@pytest.mark.bloomz
def test_config(model):
    assert model.config.architectures == ["BloomForCausalLM"]


@pytest.mark.bloomz
class TestBloomz:
    def test_num_layers(self, model):
        assert len(model.transformer.h) == 24

    def test_in_features(self, model):
        assert model.transformer.h[0].self_attention.query_key_value.in_features == 1024


# this test is slow and skipped by default
@pytest.mark.slow
def test_train_model():
    model = AutoModelForCausalLM.from_pretrained("bigscience/bloomz-560m")
    for _ in range(1000):
        # training code
        pass
