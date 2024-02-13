import os
import sys

import pytest
from transformers import AutoModelForCausalLM


@pytest.mark.xfail(reason="Flying is not implemented yet", strict=True)
def test_model_can_fly():
    model = AutoModelForCausalLM.from_pretrained("bigscience/bloomz-560m")
    assert model.can_fly()  # this method does not exist yet


@pytest.mark.xfail(reason="This test fails on Windows", condition=sys.platform == "win32")
def test_fail_on_windows():
    assert os.path.exists("/tmp")


@pytest.mark.xfail(reason="This test fails on Unix", condition=sys.platform in ["linux", "darwin"])
def test_fail_on_unix():
    assert os.path.exists("C:\\Windows")
