__author__ = 'escherba'

import random
from itertools import imap
from abc import abstractmethod


class IHashFamily:
    """
    An interface for a hash family provider.  It provides a series of random hashes
    from a universal hash family.  This can then be used for minhashing.
    """

    @abstractmethod
    def __init__(self, num_hashes, num_buckets):
        """
        Initialize the hash family by indicating how many hashes are needed.
        Also indicate the number of buckets that will be hashed to (if that is necessary
        for choosing parameters).  The hash function is not required to return values less
        than num_buckets (They will be modulo'd afterwards)
        """

    @abstractmethod
    def hashn(self, x):
        """
        return a sequence of n hashes of the value x.  n is provided in the construction
        of the hash family
        """


class XORHashFamily(IHashFamily):
    """
    An implementation of a hash family.  This uses random 32-bit hash values which are
    xor'd with the value (It assumes that the value is an integer)
    """

    def __init__(self, num_hashes, num_buckets):
        """
        Initialize a random number of 32-bit fields for xoring
        """
        IHashFamily.__init__(self, num_hashes, num_buckets)
        self._memomask = [int(random.getrandbits(32))
                          for _ in xrange(num_hashes)]

    @staticmethod
    def _xor_hash(x, mask):
        """
        This is a simple hash function which returns the result of a bitwise XOR
        on the input x and the 32-bit random mask
        """
        return int(x ^ mask)

    def hashn(self, x):
        """
        generate the series of hashes of the value to be used for finding the minhash
        The implementation uses _xor_hashing with a series of random 32-bit fields
        """
        x &= 0xffffffff  # trim x to 32-bits
        return imap(lambda mask: self._xor_hash(x, mask), self._memomask)


class MultiplyHashFamily(IHashFamily):
    """
    An implementation of a hash family that uses random multiplication of the
    form a * (x>>4) + b * x + c.
    It assumes that the value is an integer.
    This method was described in an exercise (http://www.cs.uoi.gr/~tsap/teaching/2012f-cs059/assignments/assignment2-en.pdf)
    and implemented in java (http://blogs.msdn.com/b/spt/archive/2008/06/10/set-similarity-and-min-hash.aspx)
    """

    def __init__(self, num_hashes, num_buckets):
        """
        Initialize a set of 3 random integers < num_buckets for each hash
        """
        IHashFamily.__init__(self, num_hashes, num_buckets)
        self._params = [[random.randint(1, num_buckets) for _ in xrange(3)]
                        for _ in xrange(num_hashes)]

    @staticmethod
    def _mult_hash(x, params):
        return params[0] * (x >> 4) + params[1] * x + params[2]

    def hashn(self, x):
        return imap(lambda params: self._mult_hash(x, params), self._params)
