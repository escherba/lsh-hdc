from itertools import imap
from operator import itemgetter
from functools import partial
from abc import abstractmethod
from pymaptools import UnionFind
from collections import defaultdict

from lsh import Shingler, MinHashSignature, LSHC, SimHashSignature, hamming, \
    MinHashSketchSignature
from lsh.utils import RegexTokenizer, HTMLNormalizer, read_json_file


class Cluster(object):
    """Clusters sets with Jaccard similarity above threshold with high
    probability.

    Algorithm based on Rajaraman, "Mining of Massive Datasets":
    1. Generate set signature
    2. Use LSH to map similar signatures to same buckets
    3. Use UnionFind to merge buckets containing same values
    """
    def __init__(self, signer=None, sketch_sim_fn=lambda x, y: True):
        self.union_find = UnionFind()
        self.signer = signer
        self.hash_map = defaultdict(list)
        self.sketch_sim_fn = sketch_sim_fn

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

        similar_to = partial(self.sketch_sim_fn, sketch)
        # Unite labels with same LSH keys
        for label_list in label_lists:
            if label_list:
                fst_label = label_list[0][0]
                good_lbl_count = \
                    len([x for x in label_list if similar_to(x[1])])
                if good_lbl_count > 0:
                    if label != fst_label:
                        label_list.append((label, sketch))
                        uf.union(fst_label, label)
            else:
                label_list.append((label, sketch))

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


class BaseContentFilter(object):
    @abstractmethod
    def __init__(self):
        """Init"""

    @abstractmethod
    def accept(self, obj):
        """Yes/No final answer"""


class SketchModel(object):
    """A pseudo-enum of supported models"""
    simhash = 0
    minhash = 1


class HDClustering(object):

    def __init__(self, cfg, logger=None, content_filter=None):

        """Read configuration"""

        self.cfg = cfg  # for record-keeping

        common_kwargs = dict(
            normalizer=HTMLNormalizer(),
            tokenizer=RegexTokenizer()
        )

        # Set options
        self.logger = logger
        self.max_returned = cfg['max_returned']
        self.content_filter = content_filter
        self.min_support = cfg['min_support']

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
            self.sketch_resemblance = cfg_sketch['resemblance']
            if sketch_algorithm == SketchModel.simhash:
                self.sketch_signer = SimHashSignature(bit_depth=sketch_bits)
            elif sketch_algorithm == SketchModel.minhash:
                self.sketch_signer = MinHashSketchSignature(sketch_bits)

    def clusters_from_mac_log(self, filename):
        """Find clusters in a MAC-formatted file"""

        # TODO: make this work with generic inputs
        sketch_resemblance = self.cfg['sketch']['resemblance']
        sketch_sim_fn = lambda a, b: hamming(a, b) >= sketch_resemblance
        cluster_builder = Cluster(sketch_sim_fn=sketch_sim_fn)

        # TODO: add min_support parameter
        # min_support = self.min_support

        for i, obj in enumerate(imap(itemgetter('object'),
                                     read_json_file(filename))):
            if not i % 1000:
                print "Processing line " + str(i)
            obj_content = obj['content']
            obj_post_id = obj['post_id']
            obj_user_id = obj['user_id']

            # Step 1: Extract features
            if self.content_filter is not None and \
                    not self.content_filter.accept(obj):
                features = self.shingler.get_shingles(obj_content,
                                                      prefix=obj_user_id)
                keys = self.signer.get_signature(features)
                if self.sketch_enabled:
                    sketch_features = self.sketch_shingler.get_shingles(
                        obj_content)
                    sketch = self.sketch_signer.get_signature(sketch_features)
                else:
                    sketch = None

            # Step 2: Cluster given keys, sketch
            cluster_builder.add_set(keys, label=obj_post_id, sketch=sketch)

        return cluster_builder.get_clusters()
