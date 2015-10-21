import unittest
from functools import reduce
from lsh_hdc.ext import PHashCombiner, \
    hash_combine_murmur as hash_combine_1, \
    hash_combine_boost as hash_combine_2, \
    binom as binomial


class TestCombiners(unittest.TestCase):

    def test_hash_combiner_0(self):
        """PHashCombiner should work on long inputs"""
        vec = [hash(str(x)) for x in range(8)]
        comb = PHashCombiner(8)
        self.assertEqual(15556957597623732740L, comb.combine(vec))

    def test_hash_combiner_1(self):
        """PHashCombiner should work on long inputs"""
        vec = [hash(str(x)) for x in range(100)]
        comb = PHashCombiner(18)
        self.assertNotEqual(comb.combine(vec[:17]), comb.combine(vec[:18]))
        self.assertEqual(comb.combine(vec[:18]), comb.combine(vec[:19]))

    def test_hash_combiner_2(self):
        """PHashCombiner should return 0 on empty inputs"""
        comb = PHashCombiner(8)
        self.assertEqual(0L, comb.combine([]))

    def _check_combiner(self, func):
        VEC_SIZE = 8
        vec = [hash(str(x)) for x in range(VEC_SIZE)]
        result1 = reduce(func, vec, 0L)
        for val in vec:
            self.assertNotEqual(result1, val)
        vec[VEC_SIZE // 2] = hash("test")
        result2 = reduce(func, vec, 0L)
        for val in vec:
            self.assertNotEqual(result2, val)
        self.assertNotEqual(result1, result2)

    def test_hash_combine_1(self):
        """hash_combine_1 should work"""
        self._check_combiner(hash_combine_1)

    def test_hash_combine_2(self):
        """hash_combine_2 should work"""
        self._check_combiner(hash_combine_2)

    def test_binomial(self):
        """Test binomial coefficient function
        These tests are from mpmath library
        """
        self.assertEqual(binomial(0, 0), 1)
        self.assertEqual(binomial(1, 0), 1)
        # self.assertEqual(binomial(0, -1), 0)
        self.assertEqual(binomial(3, 2), 3)
        self.assertEqual(binomial(5, 2), 10)
        self.assertEqual(binomial(5, 3), 10)
        self.assertEqual(binomial(5, 5), 1)
        # self.assertEqual(binomial(-1, 0), 1)
        # self.assertEqual(binomial(-2, -4), 3)
        # self.assertEqual(binomial(4.5, 1.5), 6.5625)
        self.assertEqual(binomial(1100, 1), 1100)
        self.assertEqual(binomial(1100, 2), 604450)
        self.assertEqual(binomial(10**9, 2), 499999999500000000)
