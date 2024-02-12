import unittest


# to invoke this test unittst-style, run `python -m unittest test_unittest_assert.py`
class TestUnittestAssert(unittest.TestCase):
    def test_upper(self):
        assert "foo".upper() == "BAR"


class TestUnittestAssertMethod(unittest.TestCase):
    def test_upper(self):
        self.assertEqual("foo".upper(), "BAR")


class TestPytestAssert:
    def test_upper(self):
        assert "foo".upper() == "BAR"


if __name__ == "__main__":
    unittest.main()
