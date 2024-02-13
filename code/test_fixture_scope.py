import pytest
from transformers import AutoModelForCausalLM


class TestFunctionScope:
    @pytest.fixture
    def lst(self):
        # by default, fixtures are function scoped
        return [1, 2, 3]

    def test_append(self, lst):
        lst.append(4)
        assert lst == [1, 2, 3, 4]

    def test_remove(self, lst):
        # Note: This test would fail if the lst fixture were not re-created for
        # each test.
        lst.remove(2)
        assert lst == [1, 3]

class TestClassScope:
    @pytest.fixture(scope="class")
    def model(self):
        # This fixture is created only once for the entire class. This is good
        # because loading the model is expensive.
        return AutoModelForCausalLM.from_pretrained("bigscience/bloomz-560m")

    def test_architecture(self, model):
        assert "BloomForCausalLM" in model.config.architectures

    def test_num_layers(self, model):
        assert len(model.transformer.h) == 24

    def test_mutate_model(self, model):
        # DO NOT DO THIS, as it affects other tests that use the model fixture
        def fine_tune(model):
            # model training code here
            pass

        fine_tune(model)
