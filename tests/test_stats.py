__author__ = 'escherba'

import unittest

from lsh.stats import UncertaintySummarizer, ExplainedVarianceSummarizer
from collections import Counter


class TestStats(unittest.TestCase):
    def test_uncertainty_index(self):
        """Expect higher index with better clustering
        """

        summ1 = UncertaintySummarizer()
        summ1.add_object(Counter({'x': 5, 'o': 1}), 6)
        summ1.add_object(Counter({'x': 1, 'o': 4, 'v': 1}), 6)
        summ1.add_object(Counter({'x': 2, 'v': 3}), 5)
        index1 = summ1.get_summary()

        # Example from Manning et al. Introduction to Information Retrieval.
        # CUP. 2009, p. 357.
        self.assertAlmostEqual(index1, 0.370949657022)

        summ2 = UncertaintySummarizer()
        summ2.add_object(Counter({'x': 6, 'o': 0, 'v': 0}), 6)
        summ2.add_object(Counter({'x': 0, 'o': 6, 'v': 0}), 6)
        summ2.add_object(Counter({'x': 0, 'o': 0, 'v': 5}), 5)
        index2 = summ2.get_summary()

        # perfect clustering
        self.assertAlmostEqual(index2, 1.0)

        summ3 = UncertaintySummarizer()
        summ3.add_object(Counter({'x': 2, 'o': 2, 'v': 2}), 6)
        summ3.add_object(Counter({'x': 7, 'o': 7, 'v': 7}), 21)
        summ3.add_object(Counter({'x': 3, 'o': 3, 'v': 3}), 9)
        index3 = summ3.get_summary()

        # worst possible clustering
        self.assertAlmostEqual(index3, 0.0)

    def test_explained_variance(self):
        """Expect higher index with better clustering
        """

        summ1 = ExplainedVarianceSummarizer()
        summ1.add_object([1, 2, 3])
        summ1.add_object([2, 3, 4])
        summ1.add_object([3, 4, 5])
        var1 = summ1.get_summary()

        # different means but same intra-cluster variance
        self.assertAlmostEqual(var1, 0.5)

        summ2 = ExplainedVarianceSummarizer()
        summ2.add_object([1, 2, 3])
        summ2.add_object([0, 2, 4])
        summ2.add_object([-1, 2, 5])
        var2 = summ2.get_summary()

        # same mean but different intra-cluster variance
        # (worst possible clustering)
        self.assertAlmostEqual(var2, 0.0)

        summ3 = ExplainedVarianceSummarizer()
        summ3.add_object([10, 10, 10])
        summ3.add_object([20, 20, 20])
        summ3.add_object([30, 30, 30])
        var3 = summ3.get_summary()

        # no intra-cluster variance (perfect clustering)
        self.assertAlmostEqual(var3, 1.0)

if __name__ == '__main__':
    unittest.main()
