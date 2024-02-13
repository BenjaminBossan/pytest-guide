import sys


def test_print(capsys):
    print("This is printed to stdout")
    print("This is printed to stderr", file=sys.stderr)
    captured = capsys.readouterr()
    assert captured.out == "This is printed to stdout\n"
    assert captured.err == "This is printed to stderr\n"
