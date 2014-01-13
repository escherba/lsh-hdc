__author__ = 'escherba'

import unittest
from lsh.utils import sort_by_length


class MyTestCase(unittest.TestCase):
    def test_sort_by_length(self):
        """Should be able to sort a list of lists by length of sublists"""
        test_case = ["abraca", "abracadabra", "a", "aba"]
        result = sort_by_length(test_case)
        self.assertEqual(list(result), ["abracadabra", "abraca", "aba", "a"])


if __name__ == '__main__':
    unittest.main()
