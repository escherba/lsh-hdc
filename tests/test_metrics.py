import unittest
from lsh_hdc.metrics import clustering_metrics


class TestMetrics(unittest.TestCase):

    def test_homogeneous_but_not_complete_labeling(self):
        """homogeneous but not complete clustering
        """
        h, c, v = clustering_metrics(
            [0, 0, 0, 1, 1, 1],
            [0, 0, 0, 1, 2, 2])
        self.assertAlmostEqual(h, 1.00, 2)
        self.assertAlmostEqual(c, 0.69, 2)
        self.assertAlmostEqual(v, 0.81, 2)

    def test_complete_but_not_homogeneous_labeling(self):
        """complete but not homogeneous clustering
        """
        h, c, v = clustering_metrics(
            [0, 0, 1, 1, 2, 2],
            [0, 0, 1, 1, 1, 1])
        self.assertAlmostEqual(h, 0.58, 2)
        self.assertAlmostEqual(c, 1.00, 2)
        self.assertAlmostEqual(v, 0.73, 2)

    def test_not_complete_and_not_homogeneous_labeling(self):
        """neither complete nor homogeneous but not so bad either
        """
        h, c, v = clustering_metrics(
            [0, 0, 0, 1, 1, 1],
            [0, 1, 0, 1, 2, 2])
        self.assertAlmostEqual(h, 0.67, 2)
        self.assertAlmostEqual(c, 0.42, 2)
        self.assertAlmostEqual(v, 0.52, 2)

    def test_non_consicutive_labels(self):
        """regression tests for labels with gaps
        """
        h, c, v = clustering_metrics(
            [0, 0, 0, 2, 2, 2],
            [0, 1, 0, 1, 2, 2])
        self.assertAlmostEqual(h, 0.67, 2)
        self.assertAlmostEqual(c, 0.42, 2)
        self.assertAlmostEqual(v, 0.52, 2)

        h, c, v = clustering_metrics(
            [0, 0, 0, 1, 1, 1],
            [0, 4, 0, 4, 2, 2])
        self.assertAlmostEqual(h, 0.67, 2)
        self.assertAlmostEqual(c, 0.42, 2)
        self.assertAlmostEqual(v, 0.52, 2)
