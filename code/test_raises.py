import pytest


def test_zero_division():
    with pytest.raises(ZeroDivisionError):
        1 / 0

def test_with_message():
    def check_value(value):
        if value <= 10:
            raise ValueError(f"value must be greater than 10, got {value} instead")

    check_value(11)  # works
    with pytest.raises(ValueError, match="value must be greater than 10, got 5 instead"):
        check_value(5)  # raises
