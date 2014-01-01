import unittest
from utils import randset
from lsh import Cluster, jaccard_sim, get_bandwidth, get_uncertainty_index


class TestCluster(unittest.TestCase):

    def test_uindex(self):
        """
        This example is taken from:
        Manning C D, Raghavan P, Schutze H. Introduction to Information
        Retrieval, CUP 2009, p. 357.
        :return: None
        """
        cluster_sets = [{"1.1", "1.2", "1.3", "1.4", "1.5", "1.6"},
                        {"2.1", "2.2", "2.3", "2.4", "2.5", "2.6"},
                        {"3.1", "3.2", "3.3", "3.4", "3.5"}]
        items_to_shingles = {
            "1.1": {("x",)},
            "1.2": {("x",)},
            "1.3": {("x",)},
            "1.4": {("x",)},
            "1.5": {("x",)},
            "1.6": {("o",)},

            "2.1": {("x",)},
            "2.2": {("o",)},
            "2.3": {("o",)},
            "2.4": {("o",)},
            "2.5": {("v",)},
            "2.6": {("o",)},

            "3.1": {("x",)},
            "3.2": {("x",)},
            "3.3": {("v",)},
            "3.4": {("v",)},
            "3.5": {("v",)}
        }
        uindex = get_uncertainty_index(cluster_sets, items_to_shingles)
        self.assertAlmostEqual(uindex, 0.370949657)

    def test_empty(self):
        """Should place the two empty sets into a separate cluster"""
        cluster = Cluster(bands=5, bandwidth=2)
        cluster.add_set("abcdefg")
        cluster.add_set("abcdefghi")
        cluster.add_set("")
        cluster.add_set("")
        self.assertEqual(len(cluster.get_clusters()), 2)

    def test_same_set(self):
        """A set should be clustered with itself"""
        s = randset()
        cluster = Cluster(bands=5, bandwidth=2)
        cluster.add_set(s)
        cluster.add_set(s)
        self.assertEqual(len(cluster.get_clusters()), 1)

    def test_similar_sets(self):
        """Two similar sets should be clustered"""
        cluster = Cluster(bands=5, bandwidth=2)
        cluster.add_set("abcdefg")
        cluster.add_set("abcdefghi")
        self.assertEqual(len(cluster.get_clusters()), 1)

    def test_dissimilar_sets(self):
        """Two non-similar sets should not be clustered"""
        cluster = Cluster(bands=5, bandwidth=2)
        cluster.add_set("12yu5abcdef")
        cluster.add_set("1234567890z")
        print cluster.get_clusters()
        num_clusters = len(cluster.get_clusters())
        self.assertEqual(num_clusters, 2,
                         "Expected 2 clusters, got {}".format(num_clusters))

    def test_cluster_threshold(self):
        """Expected error for threshold to similarity should be reasonable"""
        n_tests = 50
        dim = 15
        expected_error = 0.20

        tot_err = 0
        for test in range(n_tests):
            # Get some sets and their similarities
            sets = (randset(), randset())
            jsim = jaccard_sim(*sets)

            # Find the threshold at which they cluster together
            for threshold in range(1, 100, 5):
                threshold = float(threshold) / 100
                bandwidth = get_bandwidth(dim, threshold)
                num_bands = int(dim / bandwidth)
                cluster = Cluster(bands=num_bands,
                                  bandwidth=bandwidth)
                cluster.add_set(sets[0])
                cluster.add_set(sets[1])
                if len(cluster.get_clusters()) == 2:
                    tot_err += abs(jsim - threshold)
                    break
        avg_err = float(tot_err) / n_tests
        self.assertLessEqual(avg_err, expected_error, "Error too large")


if __name__ == '__main__':
    unittest.main()
