import gc
import unittest

import pytest
import torch
from transformers import AutoModelForCausalLM


class TestBloomzUnittest(unittest.TestCase):
    def setUp(self):
        self.model = AutoModelForCausalLM.from_pretrained("bigscience/bloomz-560m")
        self.config = self.model.config

    def tearDown(self):
        del self.model
        gc.collect()
        torch.cuda.empty_cache()

    def test_architecture(self):
        self.assertIn("BloomForCausalLM", self.config.architectures)

    def test_num_layers(self):
        self.assertEqual(len(self.model.transformer.h), 24)


class TestBloomzPytest:
    @pytest.fixture
    def model(self):
        # everything until the yield statement is the setup
        yield AutoModelForCausalLM.from_pretrained("bigscience/bloomz-560m")

        # everything after the yield statement is the teardown
        gc.collect()
        torch.cuda.empty_cache()

    @pytest.fixture
    def config(self, model):
        return model.config

    def test_architecture(self, config):
        # fixture is passed as an argument because the name "config" matches the fixture name
        assert "BloomForCausalLM" in config.architectures

    def test_num_layers(self, model):
        # fixture is passed as an argument because the name "model" matches the fixture name
        assert len(model.transformer.h) == 24
