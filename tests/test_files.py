import unittest
import sys
import json
import yaml
from operator import itemgetter
from functools import partial
from itertools import islice
from pkg_resources import resource_filename

from lsh_hdc import Shingler
from lsh_hdc.cluster import MinHashCluster as Cluster, HDClustering
from lflearn.preprocess import RegexTokenizer
from lflearn.metrics import FeatureClusterSummarizer, describe_clusters

get_resource_name = partial(resource_filename, __name__)

SEED = 0


class TestFiles(unittest.TestCase):

    def test_names(self):
        """Should return 281 clusters of names.
        """
        with open(get_resource_name('data/perrys.csv'), 'r') as fhandle:
            data = set(line.rstrip() for line in fhandle)
        cluster = Cluster(width=20, bandwidth=5, seed=SEED)
        shingler = Shingler(3)
        s = FeatureClusterSummarizer()
        for name in data:
            shingles = shingler.get_shingles(name)
            s.add_features(name, shingles)
            cluster.add_item(shingles, name)
        clusters = cluster.get_clusters()
        ti = s.summarize_clusters(clusters)
        self.assertEqual(len(clusters), 254)
        self.assertAlmostEqual(ti, 0.97547297672548)

    def test_names_kmin(self):
        """Should return 252 clusters of names.
        """
        with open(get_resource_name('data/perrys.csv'), 'r') as fhandle:
            data = set(line.rstrip() for line in fhandle)
        cluster = Cluster(width=20, bandwidth=5, kmin=2, seed=SEED)
        shingler = Shingler(3)
        s = FeatureClusterSummarizer()
        for name in data:
            shingles = shingler.get_shingles(name)
            s.add_features(name, shingles)
            cluster.add_item(shingles, name)
        clusters = cluster.get_clusters()
        # for cluster in clusters:
        #     print cluster
        ti = s.summarize_clusters(clusters)
        self.assertEqual(len(clusters), 314)
        self.assertAlmostEqual(ti, 0.9842463414730864)

    def test_names_kmin_scheme(self):
        """Should return 145 clusters of names.
        """
        with open(get_resource_name('data/perrys.csv'), 'r') as fhandle:
            data = set(line.rstrip() for line in fhandle)
        cluster = Cluster(width=20, bandwidth=5, kmin=2, lsh_scheme="a1",
                          seed=SEED)
        shingler = Shingler(3)
        s = FeatureClusterSummarizer()
        for name in data:
            shingles = shingler.get_shingles(name)
            s.add_features(name, shingles)
            cluster.add_item(shingles, name)
        clusters = cluster.get_clusters()
        # for cluster in clusters:
        #     print cluster
        ti = s.summarize_clusters(clusters)
        self.assertEqual(len(clusters), 215)
        self.assertAlmostEqual(ti, 0.9696647441240106)

    def test_bills(self):
        """Should return 97 clusters of bills.
        """
        with open(get_resource_name('data/bills100.txt'), 'r') as fhandle:
            data = [line.rstrip().split('|') for line in fhandle]
        cluster = Cluster(width=20, bandwidth=5, seed=SEED)
        shingler = Shingler(span=3, tokenizer=RegexTokenizer())
        s = FeatureClusterSummarizer()
        for label, text in data:
            shingles = shingler.get_shingles(text)
            s.add_features(label, shingles)
            cluster.add_item(shingles, label)
        clusters = cluster.get_clusters()
        ti = s.summarize_clusters(clusters)
        self.assertEqual(len(clusters), 96)
        self.assertAlmostEqual(ti, 0.9999550989193674)

    @staticmethod
    def run_simulated_manually(filepath, lines_to_read=sys.maxint,
                               cluster_args=None):
        with open(get_resource_name(filepath), 'r') as fhandle:
            data = [line.rstrip().split(' ')
                    for line in islice(fhandle, lines_to_read)]
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
            cluster.add_item(shingles, label)
        clusters = cluster.get_clusters()

        is_label_positive = lambda lbl: ':' in lbl
        return dict(stats=describe_clusters(clusters, is_label_positive),
                    uindex=s.summarize_clusters(clusters))

    def test_simulated(self):
        results = TestFiles.run_simulated_manually(
            'data/simulated.txt',
            cluster_args=dict(width=30, bandwidth=3, lsh_scheme="a0",
                              seed=SEED))
        c = results['stats']
        ti = results['uindex']
        recall = c.get_recall()
        precision = c.get_precision()
        self.assertGreaterEqual(recall, 0.499)
        self.assertGreaterEqual(precision, 0.250)
        print json.dumps(dict(
            stats=c.dict(),
            ratios=dict(
                precision=precision,
                recall=recall
            ),
            ti=ti
        ))

    def test_simulated_b(self):
        results = TestFiles.run_simulated_manually(
            'data/simulated.txt',
            cluster_args=dict(width=15, bandwidth=3, lsh_scheme="b3",
                              kmin=3, seed=SEED))
        c = results['stats']
        ti = results['uindex']
        recall = c.get_recall()
        precision = c.get_precision()
        self.assertGreaterEqual(recall, 0.465)
        self.assertGreaterEqual(precision, 0.240)
        print json.dumps(dict(
            stats=c.dict(),
            ratios=dict(
                precision=precision,
                recall=recall
            ),
            ti=ti
        ))

    def test_simulated_hd(self):

        with open(get_resource_name('test_files.simulated.yaml'), 'r') as fhandle:
            sim_cfg = yaml.load(fhandle)

        with open(get_resource_name('data/simulated.txt'), 'r') as fhandle:
            data = [line.rstrip().split(' ') for line in fhandle]

        hdc = HDClustering(sim_cfg['model'],
                           opts=dict(tokenizer=None, normalizer=None),
                           get_body=itemgetter(1),
                           get_label=itemgetter(0),
                           seed=SEED)
        clusters = hdc.clusters_from_iter(data)

        num_clusters = len([x for x in clusters if len(x) > 1])
        print "Found %d clusters" % num_clusters
        print "Points not clustered: %d" % (len(data) - num_clusters)

        is_label_positive = lambda lbl: ':' in lbl
        results = dict(stats=describe_clusters(clusters, is_label_positive))

        c = results['stats']
        recall = c.get_recall()
        precision = c.get_precision()
        print json.dumps(dict(
            stats=c.dict(),
            ratios=dict(
                precision=precision,
                recall=recall
            )
        ))
        self.assertGreaterEqual(recall, 0.661)
        self.assertGreaterEqual(precision, 0.252)


if __name__ == '__main__':
    unittest.main()
