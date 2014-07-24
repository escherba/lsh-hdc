import unittest
import sys
import json
import yaml
from logging import getLogger, config as logging_config
from functools import partial
from itertools import islice
from pkg_resources import resource_filename

from lsh import Shingler
from lsh.cluster import HDClustering, MinHashCluster as Cluster
from lsh.utils import RegexTokenizer
from lsh.stats import FeatureClusterSummarizer, get_stats
from content_rules import ContentFilter

get_resource_name = partial(resource_filename, __name__)

mac_config_filename = get_resource_name('config/mac.yaml')
mac_config = yaml.load(open(mac_config_filename, 'r'))
logging_config.dictConfig(mac_config['logging'])
LOG = getLogger(__name__)


class TestFiles(unittest.TestCase):

    def test_mac(self):
        """Expect to match number of clusters using simhash"""

        cfg = mac_config['model']
        hdc = HDClustering(cfg, logger=LOG, content_filter=ContentFilter())
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
