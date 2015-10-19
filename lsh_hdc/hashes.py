import random
import sys
import abc
from struct import unpack
from itertools import izip
from hashlib import md5


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


class HashCombiner(object):

    """use polynomial hashing to reduce a vector of hashes

    Note: a Cython implementation of this is ~2x faster so worth using
    instead
    """
    def __init__(self, size, prime=31, bits=64):
        self._coeffs = [prime ** i for i in xrange(size)]
        self._mask = 2 ** bits - 1

    def combine(self, hashes):
        ab = sum(hsh * coeff for hsh, coeff in izip(hashes, self._coeffs))
        return ab & self._mask


def hash_combine_boost(seed, val):
    """Combine seed with hash value (after Boost library)
    """
    return seed ^ (val + 0x9e3779b9 + (seed << 6) + (seed >> 2))


def hashable(value):
    if not isinstance(value, basestring):
        return repr(value)
    return value


def hash_md5_64(x, seed=0):
    """Return value is 128 bits
    """
    a, b = unpack('<QQ', md5(hashable(x)).digest())
    ab = hash_combine_boost(seed, hash_combine_boost(a, b))
    return ab & (2 ** 64 - 1)


def hash_md5_128(x, seed=0):
    """Return value is 128 bits
    """
    ab = hash_combine_boost(seed, long(md5(hashable(x)).hexdigest(), 16))
    return ab & (2 ** 128 - 1)


def hash_builtin_64(x, seed=0):
    """Return value is 32 bits

    Note: hash() is a terrible hash function. For more examples see
    http://michaelnielsen.org/blog/consistent-hashing/
    """
    a = hash(x)
    b = hash("_" + repr(x))
    ab = hash_combine_boost(seed, (1 << 32) * a + b)
    return ab & (2 ** 64 - 1)


def hash_builtin_128(x, seed=0):
    """Return value is 32 bits

    Note: hash() is a terrible hash function. For more examples see
    http://michaelnielsen.org/blog/consistent-hashing/
    """
    a = hash_builtin_64(x, seed)
    b = hash_builtin_64("_" + repr(x), seed)
    return (1 << 64) * a + b


class VarlenHash(object):
    """Create a hash function of arbitrary output length
    :param scale: integer or long indicating roughly how large should the
                  hashe values be
    :type scale: int,long

    Note: the return value of this function increases as the length of the
    text to be hashed increases. So the fuction has terrible distribution
    properties.
    """
    def __init__(self, scale=sys.maxint):
        self.scale = scale

    def __call__(self, value, seed=0):
        """A variable-length version of Python's builtin hash"""
        if isinstance(value, unicode):
            value = value.encode("utf-8")
        elif isinstance(value, str):
            pass
        else:
            value = repr(value)
        length_of_v = len(value)
        if length_of_v > 0:
            item = ord(value[0]) << 7
            mask = self.scale - 1
            for char in value:
                item = ((item * 1000003) ^ ord(char)) & mask
            item ^= length_of_v
            if item == -1:
                item = -2
            return hash_combine_boost(item, seed)
        else:
            return 0
