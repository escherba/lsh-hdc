import unittest
import sys
import json
import yaml
from itertools import imap
from operator import itemgetter
from functools import partial
from itertools import islice
from logging import getLogger, config as logging_config
from pkg_resources import resource_filename

from pymaptools.utils import override
from lsh import Shingler, SimHashSignature, hamming, LSHC, MinHashSignature, \
    MinHashSketchSignature
from lsh.cluster import MinHashCluster as Cluster, Cluster as SimpleCluster
from lsh.utils import RegexTokenizer, HTMLNormalizer, read_json_file
from lsh.stats import FeatureClusterSummarizer, get_stats
from content_rules import ContentFilter

get_resource_name = partial(resource_filename, __name__)

with open(get_resource_name('conf/hdc.yaml'), 'r') as fh:
    hdc_config = yaml.load(fh)
logging_config.dictConfig(hdc_config['logging'])
LOG = getLogger(__name__)


class SketchModel(object):
    """A pseudo-enum of supported models"""
    simhash = 0
    minhash = 1


class HDClustering(object):

    def __init__(self, cfg_override=None, logger=LOG, content_filter=None):

        """Read configuration"""

        if cfg_override is None:
            cfg = hdc_config['model']
        else:
            cfg = override(hdc_config['model'], cfg_override)
        self.cfg = cfg

        common_kwargs = dict(
            normalizer=HTMLNormalizer(),
            tokenizer=RegexTokenizer()
        )

        # Set options
        self.logger = logger
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
        cluster_builder = SimpleCluster(sketch_sim_fn=sketch_sim_fn)

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


class TestFiles(unittest.TestCase):

    def test_mac(self):
        """Expect to match number of clusters using simhash"""

        hdc = HDClustering(content_filter=ContentFilter())
        clusters = \
            hdc.clusters_from_mac_log(get_resource_name('data/mac.json'))

        for cluster in clusters:
            if len(cluster) > 1:
                print cluster
        print len(clusters)
        print len([x for x in clusters if len(x) > 1])
        # TODO: finish writing this test

    def test_names(self):
        """Should return 281 clusters of names.
        """
        with open(get_resource_name('data/perrys.csv'), 'r') as f:
            data = set(line.rstrip() for line in f)
        cluster = Cluster(width=20, bandwidth=5)
        shingler = Shingler(3)
        s = FeatureClusterSummarizer()
        for name in data:
            shingles = shingler.get_shingles(name)
            s.add_features(name, shingles)
            cluster.add_set(shingles, name)
        clusters = cluster.get_clusters()
        ti = s.summarize_clusters(clusters)
        self.assertEqual(len(clusters), 281)
        self.assertAlmostEqual(ti, 0.9780512134223747)

    def test_names_kmin(self):
        """Should return 252 clusters of names.
        """
        with open(get_resource_name('data/perrys.csv'), 'r') as f:
            data = set(line.rstrip() for line in f)
        cluster = Cluster(width=20, bandwidth=5, kmin=2)
        shingler = Shingler(3)
        s = FeatureClusterSummarizer()
        for name in data:
            shingles = shingler.get_shingles(name)
            s.add_features(name, shingles)
            cluster.add_set(shingles, name)
        clusters = cluster.get_clusters()
        # for cluster in clusters:
        #     print cluster
        ti = s.summarize_clusters(clusters)
        self.assertEqual(len(clusters), 252)
        self.assertAlmostEqual(ti, 0.9732840816954408)

    def test_names_kmin_scheme(self):
        """Should return 145 clusters of names.
        """
        with open(get_resource_name('data/perrys.csv'), 'r') as f:
            data = set(line.rstrip() for line in f)
        cluster = Cluster(width=20, bandwidth=5, kmin=2, lsh_scheme="a1")
        shingler = Shingler(3)
        s = FeatureClusterSummarizer()
        for name in data:
            shingles = shingler.get_shingles(name)
            s.add_features(name, shingles)
            cluster.add_set(shingles, name)
        clusters = cluster.get_clusters()
        # for cluster in clusters:
        #     print cluster
        ti = s.summarize_clusters(clusters)
        self.assertEqual(len(clusters), 145)
        self.assertAlmostEqual(ti, 0.9693895180932199)

    def test_bills(self):
        """Should return 97 clusters of bills.
        """
        with open(get_resource_name('data/bills100.txt'), 'r') as f:
            data = [line.rstrip().split('|') for line in f]
        cluster = Cluster(width=20, bandwidth=5)
        shingler = Shingler(span=3, tokenizer=RegexTokenizer())
        s = FeatureClusterSummarizer()
        for label, text in data:
            shingles = shingler.get_shingles(text)
            s.add_features(label, shingles)
            cluster.add_set(shingles, label)
        clusters = cluster.get_clusters()
        ti = s.summarize_clusters(clusters)
        self.assertEqual(len(clusters), 97)
        self.assertAlmostEqual(ti, 1.0)

    @staticmethod
    def run_simulated_manually(filepath, lines_to_read=sys.maxint,
                               cluster_args=None):
        with open(get_resource_name(filepath), 'r') as f:
            data = [line.rstrip().split(' ')
                    for line in islice(f, lines_to_read)]
        if cluster_args is None:
            cluster_args = dict()
        cluster = Cluster(**cluster_args)
        shingler = Shingler(span=3)
        s = FeatureClusterSummarizer()
        content_dict = dict()
        for pair in data:
            if len(pair) > 1:
                label, text = pair
            else:
                label, text = pair[0], ''
            content_dict[label] = text
            shingles = shingler.get_shingles(text)
            s.add_features(label, shingles)
            cluster.add_set(shingles, label)
        clusters = cluster.get_clusters()

        is_label_positive = lambda lbl: len(lbl.split(':')) > 1
        return dict(stats=get_stats(clusters, is_label_positive),
                    uindex=s.summarize_clusters(clusters))

    def test_simulated(self, universe_size=None):
        results = TestFiles.run_simulated_manually(
            'data/simulated.txt', lines_to_read=1000,
            cluster_args=dict(width=20, bandwidth=5, lsh_scheme="a2",
                              universe_size=universe_size))
        c = results['stats']
        ti = results['uindex']
        recall = c.get_recall()
        precision = c.get_precision()
        self.assertGreaterEqual(recall, 0.10)
        self.assertGreaterEqual(precision, 0.10)
        print json.dumps(dict(
            stats=c.dict(),
            ratios=dict(
                precision=precision,
                recall=recall
            ),
            ti=ti
        ))

    def test_simulated_b(self, universe_size=None):
        results = TestFiles.run_simulated_manually(
            'data/simulated.txt', lines_to_read=1000,
            cluster_args=dict(width=15, bandwidth=3, lsh_scheme="b3", kmin=3,
                              universe_size=universe_size))
        c = results['stats']
        ti = results['uindex']
        recall = c.get_recall()
        precision = c.get_precision()
        self.assertGreaterEqual(recall, 0.10)
        # previous result for precision: 0.09836065573770492
        self.assertGreaterEqual(precision, 0.09090909090909091)
        print json.dumps(dict(
            stats=c.dict(),
            ratios=dict(
                precision=precision,
                recall=recall
            ),
            ti=ti
        ))

if __name__ == '__main__':
    unittest.main()
