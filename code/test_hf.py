import gc

import pytest
import torch

from transformers import AutoModelForCausalLM


def test_forward(bloomz_model):
    x = torch.zeros(1, 10, dtype=torch.long)
    bloomz_model(x)


@pytest.fixture
def big_model():
    # This model is loaded into memory and then deleted again for each test that
    # uses this fixture
    model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
    print("Loaded model into memory")
    yield model
    # clean up after the fixture
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("Deleted model from memory")


def test_opt_forward(big_model):
    x = torch.zeros(1, 10, dtype=torch.long)
    big_model(x)


def test_opt_generate(big_model):
    big_model.generate(
        torch.zeros(1, 10, dtype=torch.long),
        do_sample=True,
        max_length=20,
        num_return_sequences=1,
    )
