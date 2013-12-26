import unittest
from lsh import Cluster, shingle, word_shingle


class TestFiles(unittest.TestCase):
    def test_names(self):
        """
        Should return 352 clusters of names.
        """
        names = open('data/perrys.csv', 'r').readlines()
        cluster = Cluster(threshold=0.75)
        for name in set(names):
            cluster.add_set(shingle(name, 3), name)
        self.assertEqual(len(cluster.get_sets()), 352, "len check failed")

    def test_bills(self):
        lines = open('data/bills.txt', 'r').readlines()
        cluster = Cluster(threshold=0.25)
        for line in lines[0:100]:
            label, text = line.split("|")
            s = word_shingle(text, 4)
            if len(s) > 0:
                cluster.add_set(s, label)
        sets = cluster.get_sets()
        self.assertEqual(len(sets), 88, "failed at similarity level 25%")

if __name__ == '__main__':
    unittest.main()