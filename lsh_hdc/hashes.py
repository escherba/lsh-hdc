__author__ = 'space'

import random
import abc


class IHashFamily(object):
    """
    An interface for a hash family provider.  It provides a series of random
    hashes from a universal hash family.  This can then be used for minhashing.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self, num_hashes, num_buckets, seed=0, bits=32):
        """
        Initialize the hash family by indicating how many hashes are needed.
        Also indicate the number of buckets that will be hashed to (if that is
        necessary for choosing parameters).  The hash function is not required
        to return values less than num_buckets (They will be modulo'd afterwards)
        """
        random.seed(0)
        self.mod = 2 ** bits - 1

    @abc.abstractmethod
    def hashn(self, x):
        """
        return a sequence of n hashes of the value x.  n is provided in the
        construction of the hash family
        """


class XORHashFamily(IHashFamily):
    """
    Generate the series of hashes of the value to be used for finding the
    minhash. The implementation uses _xor_hashing with a series of random
    2-bit fields

    >>> xh = XORHashFamily(5, 2)
    >>> list(xh.hashn(2151748242))
    [1483543903, 3806231852, 1112038974, 1671828496, 3958081223]
    """

    def __init__(self, num_hashes, num_buckets, seed=0, bits=32):
        super(XORHashFamily, self).__init__(
            num_hashes, num_buckets, seed, bits)
        self._memomask = [
            int(random.getrandbits(bits)) for _ in xrange(num_hashes)]

    def hashn(self, x):
        x &= self.mod
        for mask in self._memomask:
            yield x ^ mask


class MultiplyHashFamily(IHashFamily):
    """
    An implementation of a hash family that uses random multiplication of the
    form a * (x>>4) + b * x + c.
    It assumes that the value is an integer.
    This method was described in an exercise
    http://www.cs.uoi.gr/~tsap/teaching/2012f-cs059/assignments/assignment2-en.pdf
    and implemented in java
    http://blogs.msdn.com/b/spt/archive/2008/06/10/set-similarity-and-min-hash.aspx

    >>> mh = MultiplyHashFamily(3, 5, bits=64)
    >>> list(mh.hashn(544439482613082563L))
    [2347895268768918555L, 1701373383165883012L, 1224988835879435769L]
    """

    def __init__(self, num_hashes, num_buckets, seed=0, bits=32):
        # Initialize a set of 3 random integers < num_buckets for each hash
        super(MultiplyHashFamily, self).__init__(
            num_hashes, num_buckets, seed, bits)
        self._params = [
            [random.randint(1, num_buckets) for _ in xrange(3)]
            for _ in xrange(num_hashes)]

    def hashn(self, x):
        mod = self.mod
        for a, b, c in self._params:
            yield (a * (x >> 4) + b * x + c) % mod
