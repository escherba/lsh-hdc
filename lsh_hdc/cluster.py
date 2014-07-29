from functools import partial
from pymaptools import UnionFind
from collections import defaultdict, Counter
from abc import abstractmethod
from math import floor

from pymaptools.utils import deepupdate
from lsh_hdc import Shingler, SimHashSignature, hamming, \
    MinHashSketchSignature, MinHashSignature, LSHC
from lsh_hdc.utils import RegexTokenizer, HTMLNormalizer
from logging import getLogger

LOG = getLogger(__name__)


class Cluster(object):
    """Clusters sets with Jaccard similarity above threshold with high
    probability.

    Algorithm based on Rajaraman, "Mining of Massive Datasets":
    1. Generate set signature
    2. Use LSH to map similar signatures to same buckets
    3. Use UnionFind to merge buckets containing same values
    """
    def __init__(self, signer=None, sketch_dist_fn=None, max_dist=0,
                 min_support=1):
        self.union_find = UnionFind()
        self.signer = signer
        self.hash_map = defaultdict(dict)
        self.sketch_dist_fn = sketch_dist_fn
        self.max_dist = max_dist
        self.min_support = min_support

    def add_set(self, s, label=None, sketch=None):
        # Set default label for this set
        if not label:
            label = s

        # Add to union-find structure
        uf = self.union_find
        uf.__getitem__(label)

        # Get signature vector and hash it
        keys = s \
            if self.signer is None \
            else self.signer.get_signature(s)
        label_lists = map(self.hash_map.__getitem__, keys)

        if self.sketch_dist_fn is None:
            distance_from = None
        else:
            distance_from = partial(self.sketch_dist_fn, sketch)

        # Unite labels with same LSH keys
        counter = Counter()
        sketches = dict()
        for label_list in label_lists:
            label_list[label] = sketch
            counter.update(label_list.keys())
            sketches.update(label_list)

        min_supp, max_dist = self.min_support, self.max_dist
        for lbl, cnt in counter.iteritems():
            if lbl != label and cnt >= min_supp:
                skt = sketches[lbl]
                # Note: large improvement in precision when also
                # ensuring distance > 0 below:
                if distance_from is None or distance_from(skt) <= max_dist:
                    uf.union(lbl, label)

    def get_clusters(self):
        """
        :return: a list of sets representing clusters
        :rtype: list
        """
        return self.union_find.sets()


class MinHashCluster(Cluster):
    def __init__(self, width=12, bandwidth=3, lsh_scheme="a0",
                 universe_size=None, kmin=1):
        """

        :param width: Number of bands
        :type width: int
        :param lsh_scheme: Adjusts number of combinatorial bands
        :type lsh_scheme: str
        :param bandwidth: Number of rows per band
        :type bandwidth: int
        :param universe_size: A prime number of size close to token universe
                              cardinality
        :type universe_size: long
        """
        signer = MinHashSignature(width,
                                  lsh_hasher=LSHC(bandwidth, width=width,
                                                  scheme=lsh_scheme),
                                  universe_size=universe_size,
                                  kmin=kmin)
        super(MinHashCluster, self).__init__(signer=signer)


class SketchModel(object):
    """A pseudo-enum of supported models"""
    simhash = 0
    minhash = 1


class BaseContentFilter(object):

    @abstractmethod
    def accept(self, obj):
        """Process an input object according to our rules,
        returning Boolean"""


class HDClustering(object):

    def __init__(self, cfg, content_filter=None, opts=None, trace_every=0):

        """Read configuration"""
        self.cfg = cfg

        self.trace_every = trace_every
        common_kwargs = dict(
            normalizer=HTMLNormalizer(),
            tokenizer=RegexTokenizer()
        )
        deepupdate(common_kwargs, opts or {})

        # Set options
        self.content_filter = content_filter
        min_support = cfg['min_support']

        # Configure minhash signer
        sig_width = cfg['sig_width']
        lsh_hasher = LSHC(width=sig_width, **cfg['lsh_options'])
        self.signer = MinHashSignature(sig_width,
                                       lsh_hasher=lsh_hasher,
                                       kmin=cfg['kmin'])

        # Configure shingler
        cfg_key_shingle = cfg['shingler']
        cfg_key_shingle.update(common_kwargs)
        self.shingler = Shingler(**cfg_key_shingle)

        # Configure sketch comparison algorithm
        cfg_sketch = cfg['sketch']
        self.sketch_enabled = cfg_sketch['enabled']
        sketch_dist_fn = None
        xor_threshold = None
        if self.sketch_enabled:
            algorithm_name = cfg_sketch['algorithm']
            try:
                sketch_algorithm = getattr(SketchModel, algorithm_name)
            except AttributeError:
                raise RuntimeError("Unknown sketch model specified: '%s'"
                                   % algorithm_name)
            sketch_bits = cfg_sketch['size'] * 8
            cfg_sketch_shingle = cfg_sketch['shingler']
            cfg_sketch_shingle.update(common_kwargs)
            self.sketch_shingler = Shingler(**cfg_sketch_shingle)
            if sketch_algorithm == SketchModel.simhash:
                self.sketch_signer = SimHashSignature(bit_depth=sketch_bits)
            elif sketch_algorithm == SketchModel.minhash:
                self.sketch_signer = MinHashSketchSignature(sketch_bits)
            xor_threshold = \
                int(floor(sketch_bits *
                          (1.0 - float(cfg_sketch['resemblance']))))
            sketch_dist_fn = hamming

        self.cluster_builder = Cluster(sketch_dist_fn=sketch_dist_fn,
                                       max_dist=xor_threshold,
                                       min_support=min_support)

    def clusters_from_iter(self, data, get_body=None, get_label=None,
                           get_prefix=None):
        """Find clusters in an iterable"""

        cluster_builder = self.cluster_builder
        for i, obj in enumerate(data):
            if self.trace_every > 0 and (not i % self.trace_every):
                LOG.info("Processing line " + str(i))
            body = obj if get_body is None else get_body(obj)
            label = i if get_label is None else get_label(obj)
            prefix = None if get_prefix is None else get_prefix(obj)

            # Step 1: Extract features
            if self.content_filter is None or \
                    not self.content_filter.accept(obj):
                features = self.shingler.get_shingles(body, prefix=prefix)
                keys = self.signer.get_signature(features)
                if self.sketch_enabled:
                    sketch_features = self.sketch_shingler.get_shingles(body)
                    sketch = self.sketch_signer.get_signature(sketch_features)
                else:
                    sketch = None

            # Step 2: Cluster given keys, sketch
            cluster_builder.add_set(keys, label=label, sketch=sketch)

        return cluster_builder.get_clusters()
