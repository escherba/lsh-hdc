import unittest
from lsh import Cluster, shingle, Shingler


class TestFiles(unittest.TestCase):
    def test_names(self):
        """
        Should return 352 clusters of names.
        """
        names = open('data/perrys.csv', 'r').readlines()
        cluster = Cluster(10, 4)
        for name in set(names):
            cluster.add_set(shingle(name, 3), name)
        self.assertEqual(len(cluster.get_sets()), 352, "len check failed")

    def test_bills(self):
        lines = open('data/bills100.txt', 'r').readlines()
        cluster = Cluster(10, 4)
        shingler = Shingler(4)
        for line in lines[0:100]:
            label, text = line.split("|")
            s = shingler.get_shingles(text)
            if len(s) > 0:
                cluster.add_set(s, label)
        sets = cluster.get_sets()
        #self.assertEqual(len(sets), 89, "expected 89 sets")
        self.assertEqual(len(sets), 97, "expected 97 sets")

if __name__ == '__main__':
    unittest.main()