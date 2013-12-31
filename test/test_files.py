import unittest
from lsh import Cluster, WordShingler, SimpleShingler


class TestFiles(unittest.TestCase):
    def test_names(self):
        """
        Should return 352 clusters of names.
        """
        names = open('data/perrys.csv', 'r').readlines()
        cluster = Cluster(bands=2, bandwidth=4)
        shingler = SimpleShingler(3)
        for name in set(names):
            cluster.add_set(shingler.get_shingles(name), name)
        num_clusters = len(cluster.get_sets())
        self.assertEqual(num_clusters, 352,
                         "expected 352 clusters, got {}".format(num_clusters))

    def test_bills(self):
        """
        Should return 97 clusters of bills.
        """
        lines = open('data/bills100.txt', 'r').readlines()
        cluster = Cluster(bands=2, bandwidth=4)
        shingler = WordShingler(4)
        for line in lines[0:100]:
            label, text = line.split("|")
            s = shingler.get_shingles(text)
            if len(s) > 0:
                cluster.add_set(s, label)
        num_clusters = len(cluster.get_sets())
        self.assertEqual(num_clusters, 97,
                         "expected 97 clusters, got {}".format(num_clusters))

if __name__ == '__main__':
    unittest.main()