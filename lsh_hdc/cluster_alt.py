import operator
from pymaptools import UnionFind
from itertools import imap
from collections import defaultdict, Counter
from lflearn.content import MessageSource
from lflearn.preprocess import HTMLNormalizer, RegexTokenizer, URLNormalizer
from lsh_hdc import Shingler, MinHashSignature, LSHC
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
    def __init__(self, signer=None, min_support=2):
        self.union_find = UnionFind()
        self.signer = signer
        self.buckets = defaultdict(dict)
        self.min_support = min_support

    def _closeness_measure(self, sketch=None):
        min_support = self.min_support
        return lambda support, sketch: support >= min_support

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

        is_close = self._closeness_measure()
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

        is_close = self._closeness_measure()
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


def get_default_normalizer(**opts):
    normalizer = HTMLNormalizer(**opts)
    normalizer.url_normalizer = URLNormalizer()
    return normalizer


def get_default_tokenizer(**opts):
    return RegexTokenizer(**opts)


def get_default_shingler(**opts):
    shingler = Shingler(**opts)
    shingler._normalizer = None
    shingler._tokenizer = None
    return shingler


class HDClustering(object):

    def __init__(self, cfg, content_filter=None, trace_every=0,
                 get_body=None, get_label=None, get_prefix=None, seed=None,
                 min_support=None, normalizer=None, tokenizer=None):

        """Read configuration"""
        self.cfg = cfg
        self._get_body = get_body
        self._get_label = get_label
        self._get_prefix = get_prefix

        self.trace_every = trace_every

        # Set options
        self.content_filter = content_filter
        self.random_state = cfg['random_state'] if seed is None else seed
        cfg_signer = cfg['signer']
        self.min_support = cfg_signer['min_support'] if min_support is None else min_support

        # normalizer and tokenizer
        self.normalizer = get_default_normalizer(
            **cfg.get('preprocessor', {}).get('normalizer', {})) \
            if normalizer is None else normalizer
        self.tokenizer = get_default_tokenizer() if tokenizer is None else tokenizer

        # Configure minhash signer
        sig_width = cfg_signer['width']
        lsh_hasher = LSHC(width=sig_width, seed=self.random_state, **cfg_signer['lsh'])
        self.signer = MinHashSignature(sig_width,
                                       lsh_hasher=lsh_hasher,
                                       universe_size=cfg_signer['universe_size'],
                                       kmin=cfg_signer['kmin'],
                                       seed=self.random_state)

        # Configure shingler
        cfg_key_shingle = cfg['shingler']
        self.shingler = get_default_shingler(**cfg_key_shingle)

        # Configure sketch comparison algorithm
        self.sketch_enabled = False
        self.cluster_builder = Cluster(min_support=self.min_support)

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
        src = MessageSource.source(obj)
        obj_content = obj['content']
        normalized_content, meta = self.normalizer.normalize(obj_content)
        content_tokens = self.tokenizer.tokenize(normalized_content)

        if self.content_filter is not None:
            rule_accept, rule_score = self.content_filter.accept(
                obj, content_tokens=content_tokens, urls=meta.get('url_components', []), src=src)
        else:
            rule_accept = False
        if not rule_accept:
            features = self.shingler.get_shingles(content_tokens, prefix=prefix)
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
        return key, list(set(tuple_gen))
