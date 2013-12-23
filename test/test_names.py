import unittest
from lsh import Cluster, shingle


class TestNames(unittest.TestCase):
    def test_names(self):
        """
        Should return 352 clusters of names.
        """
        names = open('data/perrys.csv', 'r').readlines()
        cluster = Cluster(threshold=0.75)
        for name in set(names):
            cluster.add_set(shingle(name, 3), name)
        self.assertEqual(len(cluster.get_sets()), 352, "len check failed")