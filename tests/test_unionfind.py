__author__ = 'escherba'

import unittest
from lsh.unionfind import UnionFind


class TestUnionFind(unittest.TestCase):
    def test_simple_cluster(self):
        uf = UnionFind()
        uf.union(0, 1)
        uf.union(2, 3)
        uf.union(3, 0)
        self.assertEqual(uf.sets(), [[0, 1, 2, 3]])


if __name__ == '__main__':
    unittest.main()
