__author__ = 'escherba'

import unittest

from lsh.stats import UncertaintySummarizer, VarianceSummarizer
from collections import Counter


class MyTestCase(unittest.TestCase):
    def test_uncertainty_index(self):
        """
        Example from Manning et al. Introduction to Information Retrieval.
        CUP. 2009, p. 357.
        """
        summ = UncertaintySummarizer()
        summ.add_object(Counter({'x': 5, 'o': 1}), 6)
        summ.add_object(Counter({'x': 1, 'o': 4, 'v': 1}), 6)
        summ.add_object(Counter({'x': 2, 'v': 3}), 5)
        self.assertAlmostEqual(summ.get_summary(), 0.370949657022)

    def test_explained_variance(self):
        """
        Example from Manning et al. Introduction to Information Retrieval.
        CUP. 2009, p. 357.
        """
        summ1 = VarianceSummarizer()
        summ1.add_object([1, 2, 3])
        summ1.add_object([12, 13, 14])
        summ1.add_object([23, 24, 25])
        var1 = summ1.get_summary()

        summ2 = VarianceSummarizer()
        summ2.add_object([1, 2, 3])
        summ2.add_object([2, 3, 4])
        summ2.add_object([3, 4, 5])

        var2 = summ2.get_summary()
        self.assertLess(var2, var1)

if __name__ == '__main__':
    unittest.main()
