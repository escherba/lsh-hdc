__author__ = 'escherba'

import sys
import unittest
from lsh.utils import sort_by_length
from lsh import hamming_idist, hamming_ndist, bitlist, create_sig_selectors
from random import randint


class TestUtils(unittest.TestCase):
    def test_sort_by_length(self):
        """Should be able to sort a list of lists by length of sublists"""
        test_case = ["abraca", "abracadabra", "a", "aba"]
        result = sort_by_length(test_case)
        self.assertEqual(list(result), ["abracadabra", "abraca", "aba", "a"])

    def test_hamming(self):
        """Correctly calculate Hamming distances between numbers"""
        for i in range(100):
            x = randint(0, sys.maxint)
            y = randint(0, sys.maxint)
            self.assertEqual(hamming_ndist(x, y),
                             hamming_idist(bitlist(x), bitlist(y)))

    def test_create_sig_selectors1(self):
        """Test non-overlapping selectors"""
        sig = range(9)
        sig_res = create_sig_selectors(9, 3, "a0")
        selectors = sig_res
        selected = []
        for index, selector in selectors:
            selected.append(selector(sig))
        self.assertListEqual(selected, [(0, 1, 2),
                                        (3, 4, 5),
                                        (6, 7, 8)])

    def test_create_sig_selectors2(self):
        """Test overlapping selectors"""
        sig = range(4)
        sig_res = create_sig_selectors(4, 2, "a1")
        selectors = sig_res
        selected = []
        for index, selector in selectors:
            selected.append(selector(sig))
        self.assertListEqual(selected, [(0, 1),
                                        (0, 3),
                                        (1, 2),
                                        (2, 3)])

    def test_create_sig_selectors3(self):
        """Number of selectors must match n choose k"""
        selectors = create_sig_selectors(8, 3, "a3")
        self.assertEqual(len(selectors), 56)


if __name__ == '__main__':
    unittest.main()
