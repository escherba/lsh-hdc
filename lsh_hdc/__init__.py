__version__ = "0.0.27"

"""
Algorithms based on 'Mining of Massive Datasets'
"""

import re
import sys
from math import log1p
from operator import xor, itemgetter
from heapq import nsmallest
from logging import getLogger
from itertools import imap, izip, islice, chain, combinations
from collections import defaultdict
from abc import abstractmethod

from cityhash import CityHash64, CityHash64WithSeed, CityHash128WithSeed
from lsh_hdc.utils import totuple, tsorted


LOG = getLogger(__name__)


def long2int(x):
    """Lossily map a long type to the range of int

    :param x: input long variable
    :type x: long
    :return: input mapped to int range
    :rtype : int
    """

    smi1 = sys.maxint + 1
    return int(x % (smi1 + smi1) - smi1)


def chash(x):
    """Convenience function for calling CityHash64

    :param x: input string/hashable object
    :type x: object
    :return: integer
    :rtype: int
    """
    return long2int(CityHash64(x))


def bitlist(n):
    """Unpack number into a list, size-independent

    :param n: Some number
    :type n: int
    :returns: list of bits
    :rtype: list
    """
    if n == 0:
        return [0]
    s = []
    if n < 0:
        n = -n
    while n > 0:
        s.append(1 if n & 1 else 0)
        n >>= 1
    return s


def bitstring(n):
    """Unpack number into a string, size-independent

    :param n: Some number
    :type n: int
    :returns: string of bits
    :rtype: str
    """
    return '{0:b}'.format(n)[::-1]


def bitstring_padded(size, n):
    """Unpack number into a string, size-independent

    :param n: Some number
    :type n: int
    :returns: string of bits
    :rtype: str
    """
    res = '{0:b}'.format(n)[::-1]
    return res.ljust(size, '0')


def from_bitlist(l):
    """Undo bitlist"""
    return sum(1 << i for i, b in enumerate(l) if b)


def from_bitstring(l):
    """Undo bitstring"""
    return sum(1 << i for i, b in enumerate(l) if int(b))


def hamming_idist(s, t):
    """Return the Hamming distance between two lists of bits

    :param s: sequence 1
    :type s: list
    :param t: sequence 2
    :type t: list
    :returns: hamming distance between two lists of bits
    :rtype: int
    """

    # TODO: move this into utils
    d = len(s) - len(t)
    if d > 0:
        t += [0] * d
    else:
        s += [0] * (-d)
    return sum(ch1 != ch2 for ch1, ch2 in izip(s, t))


def hamming(a, b):
    """Return the Hamming distance between bits of two numbers

    :param a: some number
    :type a: long, int
    :param b: some number
    :type b: long, int
    :returns: hamming distance between two numbers
    :rtype: int
    """

    # TODO: move this into utils
    return bitlist(a ^ b).count(1)


def nskip(l, skip):
    """Skip some elements form a list

    :param l: Iterable
    :type l: collections.iterable
    :param skip: How many words to skip
    :type skip: int
    :returns: sequence with skipped words
    :rtype: collections.iterable

    >>> list(nskip("abcdefg", 2))
    ['a', 'd', 'g']
    >>> list(nskip("abc", 5))
    ['a']

    """
    n = skip + 1
    return (v for i, v in enumerate(l) if not i % n)


def shinglify(it, span, skip=0):
    """Extract shingles from an iterable

    :param it: Iterable
    :type it: collections.iterable
    :param span: shingle span
    :type span: int
    :param skip: How many words to skip
    :type skip: int
    :returns: sequence of tuples (shingles)
    :rtype : list

    >>> shingles = list(shinglify("abracadabra", 5, skip=1))
    >>> len(shingles)
    7
    >>> ('d', 'b', 'a') in shingles
    True

    Must return a single shingle when span > len(tokens)
    >>> list(shinglify("abc", 4))
    [('a', 'b', 'c')]

    Must return an empty list when span=0
    >>> list(shinglify("abc", 0))
    []

    Must return the last pair
    >>> list(shinglify("abcde", 4, skip=1))
    [('a', 'c'), ('b', 'd'), ('c', 'e')]

    Must skip tokens even when len(tokens) < span
    >>> list(shinglify("abc", 4, skip=1))
    [('a', 'c')]

    """
    tokens = list(it)
    if len(tokens) >= span:
        return izip(*nskip((tokens[i:] for i in range(span)), skip))
    else:
        return [tuple(nskip(tokens, skip))]


def mshinglify(it, span, skip=0):
    """Same as shingligy except repeatedly mask one word

    :param it: Iterable
    :type it: collections.iterable
    :param span: shingle span
    :type span: int
    :returns: sequence of tuples (shingles)
    :rtype : list

    >>> list(mshinglify("abcde", 10, skip=1))
    [('a', 'e'), ('a', 'c'), ('a', 'c', 'e'), ('c', 'e')]
    >>> list(mshinglify("a", 10, skip=1))
    []
    """
    tokens = list(it)

    # range of indices where masking is allowed
    mask = range(1, min(len(tokens) - skip, span // (skip + 1) + 1))
    s = None
    for s in shinglify(tokens, span, skip=skip):
        for m in mask:
            yield s[:m] + s[m + 1:]
    if mask and (s is not None) and (len(s) > 1):
        yield s[1:]


def create_getters(lot):
    """A wrapper that fixes operator.itemgetter behavior
    where it returns a scalar for tuple input of cardinality one

    :param lot: a list of tuples
    :type lot: list
    :returns: a generator of item getters
    :rtype: generator
    """
    for tup in lot:
        yield itemgetter(*tup) if tup else lambda x: ()


def cntuples(m, n):
    """Return circular n-tuples

    :param m: length of master range vector
    :type m: int
    :param n: tuple length
    :type n: int
    :returns: a sequence of n-tuples
    :rtype : list
    """
    vec = range(m)
    it = izip(*[chain(islice(vec, i, None), islice(vec, None, i)) for i in xrange(n)])
    return sorted(imap(tsorted, it))


def cntuplesx(m, n, kmin=1):
    """Return circular n-tuples

    :param m: length of master range vector
    :type m: int
    :param n: tuple length
    :type n: int
    :param kmin: k from k-min
    :type kmin: int
    :returns: a sequence of n-tuples
    :rtype : list
    """
    vec = range(m)
    nvec = (i * kmin for i in xrange(n))
    it = izip(*[chain(islice(vec, i, None), islice(vec, None, i)) for i in nvec])
    return map(tsorted, it)


def lsh_combinations(width, bandwidth, ramp):
    """Generate indexes for overlapping LSH band selectors

    :param width: expected signature length
    :type width: int
    :param bandwidth: band size
    :type bandwidth: int
    :param ramp: For each integer value between 1 and bandwidth,
                 return (preferably uniformly) sampled combinations
                 such that their number corresponds to (width choose ramp)
                 combinations
    :type ramp: int
    :return: a sequence of tuples with elements representing indexes
    :rtype: list
    """
    master = list(combinations(range(width), bandwidth))
    cols = set(range(bandwidth))
    left_cols = list(combinations(cols, ramp))
    right_cols = [tsorted(cols - s) for s in imap(set, left_cols)]
    left_getters = create_getters(left_cols)
    right_getters = create_getters(right_cols)
    d = defaultdict(list)
    for get_left, get_right in izip(left_getters, right_getters):
        for x in master:
            d[tsorted(totuple(get_left(x)))].append(tsorted(totuple(get_right(x))))
    return sorted(set(tsorted(k + v[0]) for k, v in d.iteritems()))


def lsh_bands(width, bandwidth):
    """Generate indexes for non-overlapping LSH band selectors

    :param width: expected signature length
    :type width: int
    :param bandwidth: band size
    :type bandwidth: int
    :return: a sequence of tuples with elements representing indexes in
             signature vector
    :rtype: list

    >>> lsh_bands(6, 2)
    [(0, 1), (2, 3), (4, 5)]
    >>> lsh_bands(6, 3)
    [(0, 1, 2), (3, 4, 5)]

    """
    return zip(*(iter(xrange(width)),) * bandwidth)


def create_sig_selectors(width, bandwidth, scheme):
    """Generate indexes for LSH band selectors

    :param width:
    :type width: int
    :param bandwidth:
    :type bandwidth: int
    :param scheme:
    :type scheme: str
    :return:
    :rtype: tuple
    """
    split_res = re.split(r'\b([a-zA-Z]+)(?=\d+\b)', scheme)
    _, scheme_code, ramp = split_res
    ramp = int(ramp)
    if width < 1:
        raise ValueError("width must be a positive integer larger than zero")
    if bandwidth < 1:
        raise ValueError("bandwidth must be a positive integer larger than zero")
    if scheme_code == "a":
        if ramp == 0:
            if width % bandwidth != 0:
                raise ValueError("when ramp is zero, width must be a multiple of bandwidth")
            bands = lsh_bands(width, bandwidth)
        elif ramp == 1:
            bands = cntuples(width, bandwidth)
        elif ramp > 0:
            if ramp > bandwidth:
                raise ValueError("ramp cannot be higher than bandwidth")
            bands = lsh_combinations(width, bandwidth, ramp)
        else:
            raise ValueError("ramp parameter cannot be negative")
        indexes = range(len(bands))
    elif scheme_code == "b":
        if ramp < 1:
            raise ValueError("for b-schemes, ramp value must be >= 1")
        bands = cntuplesx(width, bandwidth, ramp)
        # indexes = list(chain(*[[x] * ramp for x in range(width / ramp)]))
        indexes = range(len(bands))
    else:
        raise ValueError("Invalid scheme")
    LOG.info("Choosing LSH bands: " + ", ".join("{}: {}".format(index, band)
                                                for index, band in izip(indexes, bands)))
    return zip(indexes, create_getters(bands))


class Shingler(object):
    def __init__(self, span=3, skip=0, unique=True, tokenizer=None, normalizer=None):
        """
        :param span: How many words should a shingle span
        :type span: int
        :param skip: How many words should a shingle skip
        :type skip: int
        :param unique: whether to de-dupe shingles (de-dupe if False)
        :type unique: bool
        :param tokenizer: instance of Tokenizer class
        :type tokenizer: Tokenizer
        :param normalizer: instance of Normalizer class
        :type normalizer: Normalizer
        """
        self.span = span
        self.skip = skip
        self.unique = unique
        self.tokenizer = tokenizer
        self.normalizer = normalizer

    def get_shingles(self, input_text, prefix=None):
        """Return a vector of shingles from a source text

        :param input_text: Input sequence
        :type input_text: collections.iterable
        :param prefix: an object to prepend to token sequence
        :type prefix: object
        :return: A set of shingles (tuples)
        :rtype: set, list
        """
        text = input_text \
            if self.normalizer is None \
            else self.normalizer.normalize(input_text)
        it = text if self.tokenizer is None else self.tokenizer.tokenize(text)
        final_it = it if prefix is None else chain((prefix,), it)
        shingles = shinglify(final_it, self.span, skip=self.skip)
        if self.unique:
            return set(shingles)
        else:
            return list(shingles)


def jaccard_sim(x, y):
    """Return Jaccard similarity between two sets

    :param x: set 1
    :type x: collections.iterable
    :param y: set 2
    :type y: collections.iterable
    :returns: Jaccard similarity of two sets
    :rtype: float
    :raises ZeroDivisionError:
    """
    set_x = set(x) if type(x) != set else x
    set_y = set(y) if type(y) != set else y
    return float(len(set_x & set_y)) / float(len(set_x | set_y))


def get_bandwidth(n, threshold):
    """Approximates the bandwidth needed to achieve a threshold.

    Threshold t = (1/b) ** (1/r) where
    b = #bands
    r = #rows per band
    n = b * r = #elements in signature

    :returns: number of rows per band
    :rtype: int
    """

    best = n
    min_err = float("inf")
    for r in range(1, n + 1):
        try:
            b = 1. / (threshold ** r)
        except ZeroDivisionError:
            return best
        err = abs(n - b * r)
        if err < min_err:
            best = r
            min_err = err
    return best


def get_threshold(r, b):
    """Approximate threshold from bandwidth and number of rows

    :param r: rows per band
    :param b: number of bands
    :return: threshold value
    :rtype: float
    """
    return (1. / b) ** (1. / r)


class Signature(object):
    """Signature Base class."""

    @abstractmethod
    def create_hash_functions(self):
        """Returns an array of length self.width consisting of different hash functions
        :rtype : list
        """

    @abstractmethod
    def get_signature(self, obj):
        """Return the signature for object
        :param obj: object for which to return signature
        :type obj: object
        :rtype : list
        """


def extend(lst, k):
    """
    extend a list to length k by duplicating last item

    >>> extend([1, 2, 3], 5)
    [1, 2, 3, 3, 3]
    """
    len_l = len(lst)
    if len_l < k:
        lst.extend([lst[-1]] * (k - len_l))
    return lst


class MinHashSignature(Signature):
    """Creates signatures for sets/tuples using minhash."""

    def __init__(self, width, lsh_hasher=None, universe_size=None, kmin=1, seed=0):
        self.universe_size = universe_size
        if type(kmin) != int:
            raise TypeError("kmin must be an integer")
        if kmin < 1:
            raise ValueError("kmin must be >= 1")
        if width % kmin != 0:
            raise ValueError("width must be a multiple of kmin")
        self.width = width / kmin
        self.lsh_hasher = lsh_hasher
        self.kmin = kmin
        self.seed = seed
        self.hashes = self.create_hash_functions()

    def create_hash_functions(self):
        """Returns an array of length self.width consisting of
        different hash functions

        Note: hash() is not as uniform as haslib.md5. For more examples see
        See http://michaelnielsen.org/blog/consistent-hashing/


        Other possible hash funcitons include:

        .. code-block:: python

            lambda x: long2int(long(md5(repr(x)).hexdigest(), 16))
            lambda x: hash(repr(x))

        """

        universe_size_ = self.universe_size

        def hash_factory(seed):
            if universe_size_ is None:
                fun = lambda x: CityHash64WithSeed(repr(x), seed)
            else:
                fun = lambda x: CityHash64WithSeed(repr(x), seed) % universe_size_
            return fun

        return map(hash_factory, [xor(self.seed, i) for i in range(self.width)])

    def _get_minhashes(self, s):
        """Returns minhash signature from a feature vector

        :returns: a signature vector
        :rtype : list
        """
        kmin = self.kmin
        if kmin > 1:
            # Choose k smallest hashes
            if len(s) > 0:
                sig_fun = lambda f: extend(nsmallest(kmin, imap(f, s)), kmin)
            else:
                # support empty sets by treating them as empty strings
                sig_fun = lambda f: extend([f("")], kmin)
            result = sum(imap(sig_fun, self.hashes), [])
        else:
            # Choose one minimal hash
            if len(s) > 0:
                sig_fun = lambda f: min(imap(f, s))
            else:
                # support empty sets by treating them as empty strings
                sig_fun = lambda f: f("")
            result = map(sig_fun, self.hashes)
        return result

    def get_signature(self, s):
        """Returns minhash signature from a feature vector (with optional LSH)

        :returns: a signature vector
        :rtype : list
        """
        gen = self._get_minhashes(s)
        lsh = self.lsh_hasher
        if lsh is None:
            return ["{}:{}".format(i, num) for i, num in enumerate(gen)]
        else:
            return list(lsh.hash(gen))

    def get_threshold(self):
        """

        :return: similarity threshold used for building clusters
        :rtype: float
        """
        bandwidth = self.lsh_hasher.bandwidth
        return get_threshold(bandwidth, self.width / bandwidth)


class MinHashSketchSignature(MinHashSignature):

    def __init__(self, width=64, universe_size=None, kmin=1, seed=0):
        MinHashSignature.__init__(self, width,
                                  universe_size=universe_size,
                                  kmin=kmin)
        self._actual_width = width
        self.seed = seed

    def get_signature(self, tokens, *features):
        result = self._get_minhashes(tokens, *features)
        bits = [1 & i for i in result]
        return sum(1 << i for i in xrange(self._actual_width) if bits[i] > 0)


class SimHashSignature(Signature):

    def __init__(self, bit_depth=64, seed=0):
        """
        :param bit_depth: Length of binary vector (bit resolution)
        :type bit_depth: int
        """
        self.bits = range(bit_depth)
        self.seed = seed
        if bit_depth <= 64:
            self.hash_fun = self._hash_fun_64
        elif bit_depth <= 128:
            self.hash_fun = self._hash_fun_128
        else:
            self.hash_fun = self._hash_fun_long

    def create_hash_functions(self):
        raise NotImplementedError

    @staticmethod
    def _hash_fun_64(x, mod_base, seed=0):
        type_of_x = type(x)
        if type_of_x == str:
            v = x
        elif type_of_x == unicode:
            v = x.encode("utf-8")
        else:
            v = repr(x)
        return CityHash64WithSeed(v, seed) % mod_base

    @staticmethod
    def _hash_fun_128(x, mod_base, seed=0):
        type_of_x = type(x)
        if type_of_x == str:
            v = x
        elif type_of_x == unicode:
            v = x.encode("utf-8")
        else:
            v = repr(x)
        a, b = CityHash128WithSeed(v, (seed, seed))
        return ((1 << 64) * a + b) % mod_base

    @staticmethod
    def _hash_fun_long(x, mod_base, seed=0):
        """A variable-length version of Python's builtin hash"""
        type_of_x = type(x)
        v = x if type_of_x == str or type_of_x == unicode else repr(x)
        length_of_v = len(v)
        if length_of_v > 0:
            x = ord(v[0]) << 7
            m = 1000003
            mask = mod_base - 1
            for c in v:
                x = ((x * m) ^ ord(c)) & mask
            x ^= length_of_v
            if x == -1:
                x = -2
            return x
        else:
            return 0

    def get_signature(self, tokens, *features):
        """Returns weighted SimHash signature of a word vector
        and of (optional) feature vectors

        :param tokens: vector of length-weighted tokens
        :type tokens: collections.iterable
        :param features: custom-weighted tokens -- a collection of tuples
            where each tuple consists of a) a feature and b) its raw
            weight with range (-1, +inf)
        :type features: collections.iterable
        :return: SimHash signature
        :rtype: long
        :raises: OverflowError
        """
        bits = self.bits
        bit_depth = len(bits)
        mod_base = 1 << bit_depth
        v = [0] * bit_depth

        # iterate over word n-grams
        for token in tokens:
            # this assigns weight of >= 1.0 to zero-length tokens:
            self._ash(bits, v, 1.0 + log1p(sum(imap(len, token))),
                      self.hash_fun(token, mod_base, self.seed))

        # iterate over features
        for feature, weight in features:
            # unlike shingles, computed feature weight should be zero if we
            # set input weight to zero
            self._ash(bits, v, log1p(weight),
                      self.hash_fun(feature, mod_base, self.seed))

        return sum(1 << i for i in bits if v[i] > 0)

    @staticmethod
    def _ash(bits, v, nw, h):
        for i in bits:
            if h & (1 << i):
                v[i] += nw
            else:
                v[i] -= nw

    def average(self, hashes):
        """Find unweighted average of multiple signatures"""
        bits = self.bits
        bit_depth = len(bits)
        v = [0] * bit_depth
        for h in hashes:
            self._ash(bits, v, 1, h)
        return sum(1 << i for i in bits if v[i] > 0)


class LSHC(object):
    """Locality-sensitive hashing

    Use a banding approach to hash similar signatures to the same buckets.
    """
    def __init__(self, bandwidth, width, scheme="a1", seed=0):
        """
        :param bandwidth: Band size
        :type bandwidth: int
        :param width: Length of signature vector
        :type width: in
        :param scheme: When scheme code is "a":
                When following number is zero, get non-overlapping bands.
                When following number is equal to bandwidth, get all possible combinations
        :type scheme: str
        """
        self.seed = seed
        self.selectors = create_sig_selectors(width, bandwidth, scheme)

    def hash(self, sig):
        """Get combinatorial sketches from a signature

        :param sig: signature to process
        :type sig: collections.iterable
        :return: 64-bit hash digest
        :rtype: collections.iterable

        Note: we use XOR-ing because it seems to be the fastest way to combine
        hashes, but we could also use CityHash64 or even the poorly distributed
        default hash function. For example, we could do this:

        .. code-block:: python

            lsh_sig = CityHash64(repr(band))

        """

        seed = self.seed
        list_sig = sig if isinstance(sig, list) else list(sig)
        for prefix, selector in self.selectors:
            band = selector(list_sig)
            lsh_sig = reduce(xor, band, seed)
            yield '{}:{}'.format(prefix, lsh_sig)
