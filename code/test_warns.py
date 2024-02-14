import warnings

import pytest


def test_warning():
    def warn():
        warnings.warn("This is a warning")

    with pytest.warns(UserWarning):
        warn()

    def another_warn():
        warnings.warn("This is another warning", FutureWarning)

    with pytest.warns(FutureWarning, match="This is another warning"):
        another_warn()
