__version__ = "0.2.0"

"""
Algorithms based on 'Mining of Massive Datasets'
"""

import re
import sys
import random
import collections
from math import log1p
from operator import itemgetter
from heapq import nsmallest
from logging import getLogger
from itertools import imap, izip, islice, chain, combinations
from abc import abstractmethod
from pymaptools.iter import cycle, take, shinglify, isiterable
from lsh_hdc.utils import wrap_scalar, tsorted, fill_with_last
from lsh_hdc.ext import hashable, VarlenHash, PHashCombiner as HashCombiner

# Various hash functions
from metrohash import metrohash64, metrohash128
from cityhash import CityHash64WithSeed
from xxh import hash64 as xxh_hash64
from lsh_hdc.ext import hash_md5_64, hash_builtin_64

chash64 = metrohash64
chash128 = metrohash128


LOG = getLogger(__name__)


HASH_FUNC_TABLE = {
    "metrohash": (metrohash64,        False),
    "cityhash":  (CityHash64WithSeed, False),
    "xxh":       (xxh_hash64,         False),
    "builtin":   (hash_builtin_64,    True),
    "md5":       (hash_md5_64,        True),
}


def mshinglify(iterable, span, skip=0):
    """Same as shingligy except repeatedly mask one word

    After sparse binary polynomial hashing (SBPH).
    Note: mshinglify with span=4 and skip=0 produces as many shingles as
    shinglify with span=3 and skip=0 plus mshinglify with span=4 and skip=1.

    :param iterable: Iterable
    :type iterable: collections.Iterable
    :param span: shingle span
    :type span: int
    :returns: sequence of tuples (shingles)
    :rtype : list

    >>> list(mshinglify("abcd", 4, skip=0))
    [('a', 'c', 'd'), ('a', 'b', 'd'), ('a', 'b', 'c'), ('b', 'c', 'd')]
    >>> list(mshinglify("abcd", 4, skip=1))
    [('a', 'c', 'd'), ('a', 'b', 'd')]
    >>> list(mshinglify("a", 10))
    []
    """
    if skip > 1:
        raise NotImplementedError("Cannot use skip > 1 with SBPH")

    tokens = iterable if isinstance(iterable, list) else list(iterable)

    # range of indices where masking is allowed
    mask = range(1, min(len(tokens), span - skip))
    shingle = ()
    for shingle in shinglify(tokens, span, skip=0):
        for mask_el in mask:
            yield shingle[:mask_el] + shingle[mask_el + 1:]
    if skip == 0 and mask and len(shingle) > 1:
        yield shingle[1:]


def consistent_sampler(pool_length=24, step=3, sample_size=8):
    """Return samples from a list of runs starting from run heads
    """
    sample_indices = []
    class_indices = []
    count = 0
    for i in xrange(step):
        if count >= sample_size:
            break
        for j in xrange(i, pool_length, step):
            if count >= sample_size:
                break
            sample_indices.append(j)
            class_indices.append(i)
            count += 1
    return sample_indices, class_indices


def create_hash_factory(hashfun, complex_types=False, universe_size=None):
    """Create a function to make hash functions

    :param hashfun: hash function to use
    :type hashfun: callable
    :param complex_types: whether hash function supports hashing of complex
    types, either through native support or through repr
    :type complex_types: bool
    :param universe_size: upper limit to hash value
    :type universe_size: int, long
    :returns: a hash factory
    :rtype: callable

    """
    def hash_factory(seed):
        if complex_types:
            if universe_size is None:
                fun = lambda x: hashfun(x, seed)
            else:
                fun = lambda x: hashfun(x, seed) % universe_size
        else:
            if universe_size is None:
                fun = lambda x: hashfun(hashable(x), seed)
            else:
                fun = lambda x: hashfun(hashable(x), seed) % universe_size
        return fun
    return hash_factory


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
    iterable = izip(*[chain(islice(vec, i, None), islice(vec, None, i)) for i in xrange(n)])
    return sorted(imap(tsorted, iterable))


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
    iterable = izip(*[chain(islice(vec, i, None), islice(vec, None, i)) for i in nvec])
    return map(tsorted, iterable)


def lsh_combinations(width, bandwidth, ramp):
    """Generate indices for overlapping LSH band selectors

    :param width: expected signature length
    :type width: int
    :param bandwidth: band size
    :type bandwidth: int
    :param ramp: For each integer value between 1 and bandwidth, return
    (preferably uniformly) sampled combinations such that their number
    corresponds to (width choose ramp) combinations
    :type ramp: int
    :return: a sequence of tuples with elements representing indices
    :rtype: list

    """
    master = list(combinations(range(width), bandwidth))
    cols = set(range(bandwidth))
    left_cols = list(combinations(cols, ramp))
    right_cols = [tsorted(cols - s) for s in imap(set, left_cols)]
    left_getters = create_getters(left_cols)
    right_getters = create_getters(right_cols)
    mapping = collections.defaultdict(list)
    for get_left, get_right in izip(left_getters, right_getters):
        for element in master:
            mapping[tsorted(wrap_scalar(get_left(element)))].append(
                tsorted(wrap_scalar(get_right(element))))
    return sorted(set(tsorted(k + v[0]) for k, v in mapping.iteritems()))


def create_lsh_bands(width, bandwidth):
    """Generate indices for non-overlapping LSH band selectors

    :param width: expected signature length
    :type width: int
    :param bandwidth: band size
    :type bandwidth: int
    :return: a sequence of tuples with elements representing indices in
    signature vector
    :rtype: list

    >>> create_lsh_bands(6, 2)
    [(0, 1), (2, 3), (4, 5)]
    >>> create_lsh_bands(6, 3)
    [(0, 1, 2), (3, 4, 5)]

    """
    return zip(*(iter(xrange(width)),) * bandwidth)


def create_sig_selectors(width, bandwidth, scheme):
    """Generate indices for LSH band selectors

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
            bands = create_lsh_bands(width, bandwidth)
        elif ramp == 1:
            bands = cntuples(width, bandwidth)
        elif ramp > 0:
            if ramp > bandwidth:
                raise ValueError("ramp cannot be higher than bandwidth")
            bands = lsh_combinations(width, bandwidth, ramp)
        else:
            raise ValueError("ramp parameter cannot be negative")
        indices = range(len(bands))
    elif scheme_code == "b":
        if ramp < 1:
            raise ValueError("for b-schemes, ramp value must be >= 1")
        bands = cntuplesx(width, bandwidth, ramp)
        # indices = list(chain(*[[x] * ramp for x in range(width / ramp)]))
        indices = range(len(bands))
    else:
        raise ValueError("Invalid scheme")
    LOG.info("Choosing LSH bands: " + ", ".join("{}: {}".format(idx, band)
                                                for idx, band in izip(indices, bands)))
    return zip(indices, create_getters(bands))


class Shingler(object):

    _algorithms = {
        "standard": shinglify,  # standard algorithm
        "sbph": mshinglify      # after "sparse binary polynomial hashing"
    }

    def __init__(self, span=3, skip=0, kmin=0, algorithm="standard", unique=True,
                 tokenizer=None, normalizer=None):
        """
        :param span: How many words should a shingle span
        :type span: int
        :param skip: How many words should a shingle skip
        :type skip: int
        :param kmin: minimum expected number of shingles (not set if 0 or unique=True)
        :type kmin: int
        :param unique: whether to de-dupe shingles (de-dupe if False)
        :type unique: bool
        :param tokenizer: instance of Tokenizer class
        :type tokenizer: Tokenizer
        :param normalizer: instance of Normalizer class
        :type normalizer: Normalizer

        """
        self._algorithm = algorithm
        self._shinglify = self._algorithms[algorithm]
        self._span = span
        self._skip = skip
        self._kmin = kmin
        self._unique = unique
        self._tokenize = list if tokenizer is None else tokenizer.tokenize
        self._normalizer = normalizer

    def get_shingles(self, input_text, prefix=None):
        """Return a vector of shingles from a source text

        :param input_text: Input sequence
        :type input_text: collections.Iterable
        :param prefix: an object to prepend to token sequence
        :type prefix: object
        :return: A set of shingles (tuples)
        :rtype: set, list

        """
        normalizer = self._normalizer
        text = input_text \
            if normalizer is None \
            else normalizer.normalize(input_text)
        tokens = text if isiterable(text) else self._tokenize(text)
        span = self._span
        unique = self._unique
        kmin = self._kmin
        if not unique and kmin > 0:
            # cycle tokens until we can take kmin shingles
            token_count = len(tokens)
            prefix_token_count = 0 if prefix is None else 1
            num_shingles = token_count - span + prefix_token_count + 1
            append_num = kmin - num_shingles
            if append_num > 0:
                tokens = take(token_count + append_num, cycle(tokens))
        final_it = tokens if prefix is None else chain([prefix], tokens)
        shingles = self._shinglify(final_it, span, skip=self._skip)
        result = set(shingles) if unique else list(shingles)
        return result


def get_bandwidth(width, threshold):
    """Approximates the bandwidth needed to achieve a threshold.

    :param width:     = bands * rows  # = elements in signature
    :param threshold: = (1/bands) ** (1/rows) where
    :returns:         number of rows per band
    :rtype: int

    """

    best = width
    min_err = float("inf")
    for rows_per_band in range(1, width + 1):
        try:
            num_bands = 1. / (threshold ** rows_per_band)
        except ZeroDivisionError:
            return best
        err = abs(width - num_bands * rows_per_band)
        if err < min_err:
            best = rows_per_band
            min_err = err
    return best


def get_threshold(rows, bands):
    """Approximate threshold from bandwidth and number of rows

    :param rows: rows per band
    :param bands: number of bands
    :return: threshold value
    :rtype: float

    """
    return (1. / bands) ** (1. / rows)


class Signature(object):
    """Signature Base class."""

    @abstractmethod
    def create_hash_functions(self):
        """Returns an array of length self.width consisting of different hash functions
        :rtype : list

        """

    @abstractmethod
    def get_signature(self, vec, with_sketch=False):
        """Return the signature for vector
        :param vec: vector for which to return signature
        :type vec: collections.Iterable
        :rtype : list

        """


class MinHashSignature(Signature):
    """Obtain minhash signature"""

    def __init__(self, width, lsh_hasher=None, universe_size=None, kmin=1,
                 seed=0, hashfun='metrohash'):
        if width % kmin != 0:
            raise ValueError("width must be a multiple of kmin")
        if type(kmin) != int:
            raise TypeError("kmin must be an integer")
        elif kmin > 1:
            self._get_minhashes = self._get_minhashes_kmin1p
        elif kmin == 1:
            self._get_minhashes = self._get_minhashes_kmin1
        else:
            raise ValueError("kmin must be >= 1")
        self.width = width / kmin
        self.kmin = kmin
        self.hashfun, self.complex_types = HASH_FUNC_TABLE[hashfun]

        self.lsh_hasher = lsh_hasher
        self.seed = seed
        self.universe_size = universe_size

        self._sketch_getter = None
        self._simhash_sketcher = None
        self._sketch_weights = None

        self.hashes = self.create_hash_functions()

    def configure_sketcher(self, sketch_type='minhash', sketch_size=None,
                           sketch_base=1.414):

        sketch_indices, sketch_classes = \
            self.create_sketch_getter(sketch_size)
        self._sketch_getter = itemgetter(*sketch_indices)
        self._sketch_weights = \
            self.sketch_weight_builder(sketch_base, sketch_classes)
        if sketch_type == 'simhash':
            self._simhash_sketcher = SimHashSignature(sketch_size)
        elif sketch_type == 'minhash':
            self._simhash_sketcher = None
        else:
            raise ValueError("Sketch type %s not supported" % sketch_type)

    @staticmethod
    def sketch_weight_builder(base, class_indices):
        total = 0.0
        base = float(base)
        weight_map = dict()
        unique_classes = sorted(set(class_indices))
        for class_idx in unique_classes:
            try:
                weight = base ** -float(class_idx)
            except ZeroDivisionError:
                weight = 0.0
            total += weight
            weight_map[class_idx] = weight
        normalized_weight_map = {class_idx: weight / total
                                 for class_idx, weight
                                 in weight_map.iteritems()}
        normalized_weights = []
        for class_idx in class_indices:
            normalized_weights.append(normalized_weight_map[class_idx])
        return normalized_weights

    def create_sketch_getter(self, sketch_size):
        """Wrapper around consistent_sampler to return itemgetter instance"""
        if sketch_size is None:
            sketch_size = self.width
        pool_len = self.width * self.kmin
        assert sketch_size <= pool_len
        indices, classes = consistent_sampler(pool_len, self.kmin, sketch_size)
        return indices, classes

    def create_hash_functions(self):
        """Return a list of length self.width of different hash functions
        """
        # draw a sample of unique random integers from pool of [0, sys.maxint]
        random.seed(self.seed)
        seeds = random.sample(xrange(sys.maxint), self.width)
        hash_factory = create_hash_factory(
            self.hashfun,
            complex_types=self.complex_types,
            universe_size=self.universe_size
        )
        return map(hash_factory, seeds)

    def _get_minhashes_kmin1p(self, vec):
        """Returns minhash signature from a feature vector
        :returns: a signature vector
        :rtype : list
        """
        kmin = self.kmin
        # Choose k smallest hashes
        if len(vec) > 0:
            sig_fun = lambda f: fill_with_last(nsmallest(kmin, imap(f, vec)), kmin)
        else:
            # support empty sets by treating them as empty strings
            sig_fun = lambda f: fill_with_last([f("")], kmin)

        # flatten list of lists
        return sum(imap(sig_fun, self.hashes), [])

    def _get_minhashes_kmin1(self, vec):
        """Returns minhash signature from a feature vector
        :returns: a signature vector
        :rtype : list
        """
        # Choose one minimal hash
        if len(vec) > 0:
            sig_fun = lambda f: min(imap(f, vec))
        else:
            # support empty sets by treating them as empty strings
            sig_fun = lambda f: f("")
        return map(sig_fun, self.hashes)

    @staticmethod
    def _minhash_sketch(minhash_sample):
        bits = (1 & minhash for minhash in minhash_sample)
        sketch = sum(1 << idx for idx, bit in enumerate(bits) if bit > 0)
        return sketch

    def get_signature(self, vec, with_sketch=False):
        """Returns minhash signature from a feature vector (with optional LSH)

        :returns: a signature vector
        :rtype : list
        """
        minhashes = self._get_minhashes(vec)
        lsh = self.lsh_hasher
        if lsh is None:
            sig_vector = ["{}:{}".format(idx, minhash)
                          for idx, minhash in enumerate(minhashes)]
        else:
            sig_vector = list(lsh.hash(minhashes))

        if with_sketch:
            minhash_sample = self._sketch_getter(minhashes)
            if self._simhash_sketcher is not None:
                sketch = self._simhash_sketcher._sig_with_weights(
                    minhash_sample, self._sketch_weights)
            else:
                sketch = self._minhash_sketch(minhash_sample)
            return sig_vector, sketch
        else:
            return sig_vector

    def get_threshold(self):
        """

        :return: similarity threshold used for building clusters
        :rtype: float
        """
        bandwidth = self.lsh_hasher.bandwidth
        return get_threshold(bandwidth, self.width / bandwidth)


class MinHashSketchSignature(MinHashSignature):

    def __init__(self, width=64, universe_size=None, kmin=1, seed=0):
        MinHashSignature.__init__(self, width, universe_size=universe_size,
                                  kmin=kmin, seed=seed)
        self._actual_width = width
        self.configure_sketcher(sketch_type='minhash', sketch_size=width)

    def get_signature(self, tokens, *features):
        minhashes = self._get_minhashes(tokens)
        minhash_sample = self._sketch_getter(minhashes)
        return self._minhash_sketch(minhash_sample)


class SimHashSignature(Signature):

    def __init__(self, bit_depth=64, seed=0, hashfun_map=((64, chash64), (128, chash128))):
        """
        :param bit_depth: Length of binary vector (bit resolution)
        :type bit_depth: int
        """
        self.bits = range(bit_depth)
        self.seed = seed

        for key_bits, val_fun in hashfun_map:
            if bit_depth <= key_bits:
                self.hashfun = lambda value, seed=0: val_fun(hashable(value), seed)
                break
        else:
            self.hashfun = VarlenHash(bit_depth)

    def create_hash_functions(self):
        raise NotImplementedError

    def get_signature(self, tokens, *features):
        """Returns weighted SimHash signature of a word vector
        and of (optional) feature vectors

        :param tokens: vector of length-weighted tokens
        :type tokens: collections.Iterable
        :param features: custom-weighted tokens -- a collection of tuples where
                         each tuple consists of a) a feature and b) its raw
                         weight with range (-1, +inf)
        :type features: collections.Iterable
        :return: SimHash signature
        :rtype: long
        :raises: OverflowError
        """
        hashfun = self.hashfun
        seed = self.seed
        token_weights = (log1p(sum(imap(len, token))) for token in tokens)
        if features:
            features, feature_weights = izip(*features)
            fin_features = (hashfun(feature, seed)
                            for feature in chain(tokens, features))
            fin_weights = chain(token_weights, feature_weights)
        else:
            fin_features = (hashfun(feature, seed) for feature in tokens)
            fin_weights = token_weights
        return self._sig_with_weights(fin_features, fin_weights)

    def _sig_with_weights(self, hashed_features, feature_weights):
        """SimHash signature from a list of hashes and corresponding weights
        :param hashed_features: an iterable of hashed features
        :type hashed_features: collections.Iterable
        :param feature_weights: an iterable of weights (a weight can have
                                weight in range of (-1.0, +inf), with zero
                                meaning feature not considered)
        :type feature_weights: collections.Iterable
        """
        bits = self.bits
        bit_depth = len(bits)
        mod_base = 1 << bit_depth
        vec = [0] * bit_depth
        for feature, weight in izip(hashed_features, feature_weights):
            scaled_weight = log1p(weight)
            mod_feature = feature % mod_base
            for i in bits:
                if mod_feature & (1 << i):
                    vec[i] += scaled_weight
                else:
                    vec[i] -= scaled_weight
        return sum(1 << i for i in bits if vec[i] > 0)


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
        self.selectors = create_sig_selectors(width, bandwidth, scheme)
        self.combiner = HashCombiner(bandwidth)

    def hash(self, sig):
        """Get combinatorial sketches from a signature

        :param sig: signature to process
        :type sig: collections.Iterable
        :return: 64-bit hash digest
        :rtype: collections.Iterable

        Note: we use XOR-ing because it seems to be the fastest way to combine
        hashes, but we could also use chash64 or even the poorly distributed
        default hash function. For example, we could do this:

        .. code-block:: python

            lsh_sig = chash64(repr(band), 0)

        """
        list_sig = sig if isinstance(sig, list) else list(sig)
        hash_combine = self.combiner.combine
        for prefix, selector in self.selectors:
            yield '{}:{}'.format(prefix, hash_combine(selector(list_sig)))
