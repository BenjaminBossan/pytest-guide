import unittest


class TestLists(unittest.TestCase):
    def test_contains(self):
        self.assertIn(2, [1, 2, 3])

    def test_append(self):
        lst = [1, 2, 3]
        lst.append(4)
        self.assertEqual(lst, [1, 2, 3, 4])


# pytest style using standalone functions
def test_contains():
    assert 2 in [1, 2, 3]


def test_append():
    lst = [1, 2, 3]
    lst.append(4)
    assert lst == [1, 2, 3, 4]
