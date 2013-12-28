__author__ = 'escherba'

import unittest
from utils import uniq_rev_index


class MyTestCase(unittest.TestCase):
    def test_rev_index_lengths(self):
        d = {"x": [1, 2, 3], "y": [4, 5, 6, 9]}
        e = uniq_rev_index(d)
        self.assertEqual(len(e), sum(map(len, d.values())))


if __name__ == '__main__':
    unittest.main()
