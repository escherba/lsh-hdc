# -*- coding: utf-8 -*-
"""
lsh.py

Algorithms based on 'Mining of Massive Datasets'
"""

import re
import sys
import HTMLParser
from itertools import imap, izip
from collections import defaultdict
from cityhash import CityHash64
from abc import abstractmethod

from .unionfind import UnionFind
from .utils import lapply


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
    #TODO: Convert to a C or Cython function and add to cityhash
    return long2int(CityHash64(x))


def shinglify(tokens, shingle_size=2):
    token_count = len(tokens)
    if token_count >= shingle_size:
        for offset in xrange(token_count - shingle_size + 1):
            yield tuple(tokens[offset:(offset + shingle_size)])
    else:
        yield tuple(tokens)


class Shingler:
    def __init__(self, n):
        self.n = n

    def get_shingles(self, text):
        """

        :param text:
        :return: A set of shingles (tuples)
        :rtype: set
        """
        return set(shinglify(text, shingle_size=self.n))


class WordShingler(Shingler):
    def __init__(self, n, pattern=None):
        """

        :param n: Specifies shingle length. If a string
                  has fewer tokens than n, get_shingles() will
                  return a short shingle of only those tokens.
        :param pattern: regex pattern that matches tokens from
                        which shingles are formed.
        """
        Shingler.__init__(self, n)
        if pattern is None:
            """
            pattern = ur'(?u)\w+'
            pattern = ur'(?:\B[#@$£€¥₩฿])?(?u)\w+(?:[%\+]\B)?'
            pattern = ur'''
                        (?:                # Either URL
                        http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+
                        |                  # or
                        (?:\B[#@$£€¥₩฿])?  # (preceded by optional pound-, at-, or currency signs)
                        (?u)\w+            # a Unicode word
                        (?:[%\+]\B)?       # optionally followed by percentage or plus signs
                        )
                        '''
            """
            pattern = ur'(?u)\w+'
        self.r = re.compile(pattern, (re.VERBOSE | re.UNICODE))
        self.html_parser = HTMLParser.HTMLParser()

    def normalize(self, text):
        """

        :param text:
        :return:
        :rtype: unicode
        """
        return self.html_parser.unescape(text).lower()

    def tokenize(self, text):
        """

        :param text:
        :return:
        :rtype: list
        """
        return self.r.findall(text)

    def get_shingles(self, text):
        """Get shingles (n-grams) from text

        :param text: Source text to process
        :type text: unicode
        :return: a collection of shingles found
        :rtype: set
        """
        normalized_text = self.normalize(text)
        tokens = self.tokenize(normalized_text)
        return set(shinglify(tokens, self.n))


def jaccard_sim(x, y):
    """

    :throws ZeroDivisionError:
    :returns: Jaccard similarity of two sets
    :rtype: float
    """
    set_x = set(x)
    set_y = set(y)
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
    """

    :param r: rows per band
    :param b: number of bands
    :return: threshold value
    :rtype: float
    """
    return (1. / b) ** (1. / r)


class Signature:
    """Signature Base class."""

    def __init__(self, width):
        self.width = width
        self.hashes = self.hash_functions()

    @abstractmethod
    def hash_functions(self):
        """Returns an array of length self.width consisting of different hash functions
        :rtype : list
        """

    @abstractmethod
    def get_signature(self, obj):
        """Return the signature for object
        :rtype : list
        """


class MinHashSignature(Signature):
    """Creates signatures for sets/tuples using minhash."""

    def hash_functions(self):
        """Returns an array of length self.width consisting of
        different hash functions

        Note: hash() is not as uniform as haslib.md5
        See http://michaelnielsen.org/blog/consistent-hashing/
        for examples
        """
        def hash_factory(seed):
            salt_seed = "salt" + repr(seed)
            return lambda x: chash(salt_seed + str(x) + "salt")
            #return lambda x: long2int(long(md5(prefix + str(x) + "salt").hexdigest(), 16))
            #return lambda x: hash(prefix + str(x) + "salt")
        return map(hash_factory, range(self.width))

    def get_signature(self, s):
        """Returns minhash signature for set s -- which is a list of lists
        consisting of hashings for each value and has function

        Alternatively, we can choose k smallest hashes like so:

            return heapq.smallest(k, imap(self.hashes[0], s))

        TODO: test the hypothesis that k-smallest technique is suboptimal for
        small documents
        """

        if len(s) > 0:
            sig_fun = lambda f: min(imap(f, s))
        else:
            # support empty sets by treating them as empty strings
            sig_fun = lambda f: f("")
        return imap(sig_fun, self.hashes)


class LSH:
    """Locality sensitive hashing.  Uses a banding approach to hash
    similar signatures to the same buckets."""
    def __init__(self, bandwidth):
        #self.length = length
        #self.threshold = threshold
        #self.bandwidth = get_bandwidth(length, threshold)
        self.bandwidth = bandwidth

    def hash(self, sig):
        """Generate hashvals for this signature

        The zip() clause converts a 1D-list to a list of b-dimensional
        tuples such that:
        [1,2,3,4,5,6] -> [(1,2), (3,4), (5,6)] if bandwidth == 2
                      -> [(1,2,3), (4,5,6)]    if bandwidth == 3

        Note: hash choice here doesn't matter (can use even poorly
        distributed default hash function), but we use CityHash64
        for consistency.

        :return: 64-bit hash digest
        :rtype: collections.iterable
        """
        for band in zip(*(iter(sig),) * self.bandwidth):
            yield chash("salt" + repr(band) + "tlas")
            #yield hash("salt" + repr(band) + "tlas")


class Cluster:
    """Clusters sets with Jaccard similarity above threshold with high
    probability.

    Algorithm based on Rajaraman, "Mining of Massive Datasets":
    1. Generate set signature
    2. Use LSH to map similar signatures to same buckets
    3. Use UnionFind to merge buckets containing same values
    """
    def __init__(self, bands=4, bandwidth=3):
        """

        :param bands: Number of bands
        :param bandwidth: Number of rows per band
        """
        self.bands = bands
        self.unionfind = UnionFind()
        self.signer = MinHashSignature(bands * bandwidth)
        self.hasher = LSH(bandwidth)
        self.hashmaps = lapply(bands, defaultdict, list)

    def add_set(self, s, label=None):
        # Set default label for this set
        if not label:
            label = s

        # Add to union-find structure
        uf_ = self.unionfind
        uf_.__getitem__(label)

        # Get signature vector and hash it
        sign_gen = list(self.signer.get_signature(s))
        hash_gen = list(self.hasher.hash(sign_gen))
        label_gen = (hf[hs] for hf, hs in izip(self.hashmaps, hash_gen))

        # Unite labels with same LSH keys
        for label_list in label_gen:
            if label_list:
                first_label = label_list[0]
                if label != first_label:
                    label_list.append(label)
                    uf_.union(first_label, label)
            else:
                label_list.append(label)

    def get_threshold(self):
        """

        :return: similarity threshold used for building clusters
        :rtype: float
        """
        return get_threshold(self.hasher.bandwidth, self.bands)

    def get_clusters(self):
        """

        :return: a list of sets representing clusters
        :rtype: list
        """
        return self.unionfind.sets()
