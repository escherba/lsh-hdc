import operator
from functools import partial
from pymaptools import UnionFind
from pymaptools.bitwise import hamming
from itertools import imap
from collections import defaultdict, Counter
from math import floor

from lflearn.preprocess import HTMLNormalizer, RegexTokenizer
from lsh_hdc import Shingler, SimHashSignature, \
    MinHashSketchSignature, MinHashSignature, LSHC
from logging import getLogger

LOG = getLogger(__name__)

OPERATOR_MAP = {
    'and': operator.__and__,
    'or': operator.__or__
}


class Cluster(object):
    """Clusters sets with Jaccard similarity above threshold with high
    probability.

    Algorithm based on Rajaraman, "Mining of Massive Datasets":
    1. Generate set signature
    2. Use LSH to map similar signatures to same buckets
    3. Use UnionFind to merge buckets containing same values
    """
    def __init__(self, signer=None, sketch_dist_fn=None, max_dist=0,
                 min_support=1, sketch_operator=operator.__and__,
                 sketch_bits=0):
        self.union_find = UnionFind()
        self.signer = signer
        self.buckets = defaultdict(dict)
        self.sketch_dist_fn = sketch_dist_fn
        self.sketch_bits = sketch_bits
        self.max_dist = max_dist
        self.min_support = min_support
        self.sketch_operator = sketch_operator

    def _closeness_measure(self, sketch):
        min_support = self.min_support
        if sketch is None:
            return lambda support, sketch: \
                support >= min_support
        else:
            logical_op = self.sketch_operator
            max_dist = self.max_dist
            distance_from = partial(self.sketch_dist_fn, sketch)
            return lambda support, matched_sketch: \
                logical_op(support >= min_support,
                           distance_from(matched_sketch) <= max_dist)

    def add_item(self, item, label=None, sketch=None):
        # Set default label for this set
        if label is None:
            label = item

        # Add to union-find structure
        union_find = self.union_find
        union_find.__getitem__(label)

        # Get signature vector and hash it
        keys = item \
            if self.signer is None \
            else self.signer.get_signature(item)

        # Unite labels with same LSH keys
        counter = Counter()
        sketches = dict()
        for bucket in imap(self.buckets.__getitem__, keys):
            bucket[label] = sketch
            counter.update(bucket.keys())
            sketches.update(bucket)

        is_close = self._closeness_measure(sketch)
        for matched_label, support in counter.iteritems():
            if matched_label != label and \
                    is_close(support, sketches[matched_label]):
                union_find.union(matched_label, label)

    def add_key(self, key, label=None, sketch=None):
        """Add one LSH key only (with associated info)
        Cannot use min_support in this case (it is always equal to one)
        """
        # Set default label for this set
        if label is None:
            label = key

        # Add to union-find structure
        union_find = self.union_find
        union_find.__getitem__(label)

        # Unite labels with same LSH keys
        bucket = self.buckets[key]
        bucket[label] = sketch

        is_close = self._closeness_measure(sketch)
        for matched_label in bucket.keys():
            if matched_label != label:
                matched_sketch = bucket[matched_label]
                # Note: large improvement in precision when also ensuring that
                # distance > 0 below:
                if is_close(matched_sketch):
                    union_find.union(matched_label, label)

    def get_clusters(self):
        """
        :return: a list of sets representing clusters
        :rtype: list
        """
        return self.union_find.sets()


class MinHashCluster(Cluster):
    def __init__(self, width=12, bandwidth=3, lsh_scheme="a0",
                 universe_size=None, kmin=1, seed=0):
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
        lsh_hasher = LSHC(bandwidth, width=width, scheme=lsh_scheme) \
            if bandwidth > 1 \
            else None
        signer = MinHashSignature(width,
                                  lsh_hasher=lsh_hasher,
                                  universe_size=universe_size,
                                  kmin=kmin,
                                  seed=seed)
        super(MinHashCluster, self).__init__(signer=signer)


class SketchModel(object):
    """A pseudo-enum of supported models"""
    simhash = 0
    minhash = 1


class HDClustering(object):

    def __init__(self, cfg, content_filter=None, opts=None, trace_every=0,
                 get_body=None, get_label=None, get_prefix=None, seed=0):

        """Read configuration"""
        self.cfg = cfg
        self._get_body = get_body
        self._get_label = get_label
        self._get_prefix = get_prefix

        self.trace_every = trace_every

        # Set options
        self.content_filter = content_filter
        self.min_support = cfg['min_support']

        normalizer_opts = cfg.get('preprocessor', {}).get('normalizer', {})
        self.normalizer = HTMLNormalizer(**normalizer_opts)
        self.tokenizer = RegexTokenizer()

        # Configure minhash signer
        sig_width = cfg['sig_width']
        lsh_hasher = LSHC(width=sig_width, **cfg['lsh_options'])
        self.signer = MinHashSignature(sig_width,
                                       lsh_hasher=lsh_hasher,
                                       kmin=cfg['kmin'])

        # Configure shingler
        cfg_key_shingle = cfg['shingler']
        cfg_key_shingle.update(opts)
        self.shingler = Shingler(**cfg_key_shingle)
        self.shingler._normalizer = None
        self.shingler._tokenizer = None

        # Configure sketch comparison algorithm
        cfg_sketch = cfg['sketch']
        self.sketch_enabled = cfg_sketch['enabled']
        self.sketch_dist_fn = None
        self.max_dist = None
        if self.sketch_enabled:
            algorithm_name = cfg_sketch['algorithm']
            try:
                sketch_algorithm = getattr(SketchModel, algorithm_name)
            except AttributeError:
                raise RuntimeError("Unknown sketch model specified: '%s'"
                                   % algorithm_name)
            self.sketch_bits = cfg_sketch['size']
            cfg_sketch_shingler = cfg_sketch['shingler']
            cfg_sketch_shingler.update(opts)
            if not cfg_sketch_shingler['enabled']:
                # if sketch shingler is disabled, we also disable signer
                # as we will use default signer
                self.sketch_shingler = None
                self.sketch_signer = None
            elif sketch_algorithm == SketchModel.simhash:
                del cfg_sketch_shingler['enabled']
                self.sketch_shingler = Shingler(**cfg_sketch_shingler)
                self.sketch_signer = SimHashSignature(self.sketch_bits, seed=seed)
            elif sketch_algorithm == SketchModel.minhash:
                del cfg_sketch_shingler['enabled']
                self.sketch_shingler = Shingler(**cfg_sketch_shingler)
                self.sketch_signer = MinHashSketchSignature(self.sketch_bits, seed=seed)
            self.sketch_shingler._tokenizer = None
            self.sketch_shingler._normalizer = None
            self.max_dist = \
                int(floor(self.sketch_bits *
                          (1.0 - float(cfg_sketch['resemblance']))))
            self.sketch_dist_fn = hamming
            self.sketch_operator = OPERATOR_MAP[cfg_sketch.get('operator', 'and')]
        self.cluster_builder = Cluster(sketch_dist_fn=self.sketch_dist_fn,
                                       max_dist=self.max_dist,
                                       min_support=self.min_support,
                                       sketch_operator=self.sketch_operator)

    def _map_iter(self, data):
        """Find clusters in an iterable"""

        get_body = self._get_body
        get_label = self._get_label
        get_prefix = self._get_prefix

        for i, obj in enumerate(data):
            body = obj if get_body is None else get_body(obj)
            label = i if get_label is None else get_label(obj)
            prefix = None if get_prefix is None else get_prefix(obj)

            for feat in self._map_item(obj, body, label, prefix):
                yield feat

    def _map_item(self, obj, body, label, prefix=None):

        # Extract features
        obj_content = obj['content']
        normalized_content, meta = self.normalizer.normalize(obj_content)
        content_tokens = self.tokenizer.tokenize(normalized_content)

        if self.content_filter is None or \
                not self.content_filter.accept(obj, content_tokens=content_tokens, urls=meta['urls']):
            features = self.shingler.get_shingles(content_tokens, prefix=prefix)
            if self.sketch_enabled and (self.sketch_shingler is None or self.sketch_signer is None):
                keys, sketch = self.signer.get_signature(features, with_sketch=True)
            elif self.sketch_enabled and (self.sketch_shingler is not None and self.sketch_signer is not None):
                keys = self.signer.get_signature(features)
                sketch_features = self.sketch_shingler.get_shingles(content_tokens)
                sketch = self.sketch_signer.get_signature(sketch_features)
            else:
                keys = self.signer.get_signature(features)
                sketch = None
            yield (keys, (label, sketch))

    def clusters_from_iter(self, data):
        """Find clusters in an iterable"""

        cluster_builder = self.cluster_builder
        trace_every = self.trace_every
        for i, obj in enumerate(self._map_iter(data)):
            if trace_every > 0 and (not i % trace_every):
                LOG.info("Processing line " + str(i))

            keys, val = obj
            label, sketch = val \
                if isinstance(val, tuple) \
                else (val, None)
            cluster_builder.add_item(keys, label=label, sketch=sketch)

        return cluster_builder.get_clusters()

    def mapper(self, obj):
        """Perform a mapper task in MR"""
        get_body = self._get_body
        get_label = self._get_label
        get_prefix = self._get_prefix

        body = obj if get_body is None else get_body(obj)
        label = obj if get_label is None else get_label(obj)
        prefix = None if get_prefix is None else get_prefix(obj)

        for keys, val in self._map_item(obj, body, label, prefix):
            for key in keys:
                yield key, val

    def reducer(self, key, tuple_gen):
        """Perform a reducer task in MR

        If sketches enabled, data consists of:
            (key, [(lbl, sk), (lbl, sk), (lbl, sk)])
        Otherwise:
            (key, [lbl, lbl, lbl])
        """

        # If not using sketches, we are done
        if self.sketch_dist_fn is None:
            return key, list(set(tuple_gen))

        # create a dict mappipng a label to a sketch
        return key, dict(tuple_gen).items()
