__author__ = 'escherba'

import unittest

from lsh.stats import UncertaintySummarizer
from collections import Counter


class MyTestCase(unittest.TestCase):
    def test_uindex(self):
        """
        Example from Manning et al. Introduction to Information Retrieval.
        CUP. 2009, p. 357.
        """
        uindex = UncertaintySummarizer()
        uindex.add_object(Counter({'x': 5, 'o': 1}), 6)
        uindex.add_object(Counter({'x': 1, 'o': 4, 'v': 1}), 6)
        uindex.add_object(Counter({'x': 2, 'v': 3}), 5)
        self.assertAlmostEqual(uindex.get_summary(), 0.370949657022)


if __name__ == '__main__':
    unittest.main()
