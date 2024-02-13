import os
import sys

import pytest


def is_peft_installed():
    try:
        import peft
        return True
    except ImportError:
        return False

@pytest.mark.skip(reason="peft is not installed")
def test_peft():
    from peft import get_peft_model

    assert callable(get_peft_model)

def test_peft_2():
    if not is_peft_installed():
        pytest.skip("peft is not installed")

    from peft import get_peft_model
    assert callable(get_peft_model)


@pytest.mark.skipif(sys.platform == "win32", reason="This test fails on Windows")
def test_fail_on_windows():
    assert os.path.exists("/tmp")


def test_spacy():
    spacy = pytest.importorskip("spacy")
    nlp = spacy.load("en_core_web_sm")
    assert nlp("This is a test").ents
