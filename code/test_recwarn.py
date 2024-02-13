import warnings


def test_warning(recwarn):
    warnings.warn("This is a warning")
    assert len(recwarn) == 1
    assert recwarn[0].message.args[0] == "This is a warning"

    warnings.warn("This is another warning", FutureWarning)
    assert len(recwarn) == 2
    assert recwarn[1].category == FutureWarning
