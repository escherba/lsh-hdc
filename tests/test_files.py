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
from lflearn.feature_extract import RegexTokenizer
from lflearn.metrics import FeatureClusterSummarizer, describe_clusters
from lsh_hdc.utils import read_json_file
from content_rules import ContentFilter

get_resource_name = partial(resource_filename, __name__)


class TestFiles(unittest.TestCase):

    def test_mac(self):
        """test a file in MAC log format"""

        with open(get_resource_name('test_files.mac.yaml'), 'r') as fh:
            mac_cfg = yaml.load(fh)

        data = []
        positives = set()
        for json_obj in read_json_file(get_resource_name('data/mac.json')):
            obj = json_obj['object']
            data.append(obj)
            imp_section = json_obj.get('impermium', {}) or {}
            imp_result = imp_section.get('result', {}) or {}
            imp_tags = imp_result.get('tag_details', {}) or {}
            if 'bulk' in imp_tags:
                positives.add(obj['post_id'])

        hdc = HDClustering(cfg=mac_cfg['model'],
                           content_filter=ContentFilter(),
                           get_body=itemgetter('content'),
                           get_label=itemgetter('post_id'),
                           get_prefix=itemgetter('user_id'),
                           )
        clusters = hdc.clusters_from_iter(data)

        num_clusters = len([x for x in clusters if len(x) > 1])
        print "Found %d clusters" % num_clusters
        print "Points not clustered: %d" % (len(data) - num_clusters)

        is_label_positive = lambda lbl: lbl in positives
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
        self.assertGreaterEqual(recall, 0.272)
        self.assertGreaterEqual(precision, 0.453)

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
            cluster.add_item(shingles, name)
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
            cluster.add_item(shingles, name)
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
            cluster.add_item(shingles, name)
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
            cluster.add_item(shingles, label)
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
            cluster.add_item(shingles, label)
        clusters = cluster.get_clusters()

        is_label_positive = lambda lbl: ':' in lbl
        return dict(stats=describe_clusters(clusters, is_label_positive),
                    uindex=s.summarize_clusters(clusters))

    def test_simulated(self):
        results = TestFiles.run_simulated_manually(
            'data/simulated.txt',
            cluster_args=dict(width=30, bandwidth=3, lsh_scheme="a0"))
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
            cluster_args=dict(width=15, bandwidth=3, lsh_scheme="b3", kmin=3))
        c = results['stats']
        ti = results['uindex']
        recall = c.get_recall()
        precision = c.get_precision()
        self.assertGreaterEqual(recall, 0.483)
        self.assertGreaterEqual(precision, 0.241)
        print json.dumps(dict(
            stats=c.dict(),
            ratios=dict(
                precision=precision,
                recall=recall
            ),
            ti=ti
        ))

    def test_simulated_hd(self):

        with open(get_resource_name('test_files.simulated.yaml'), 'r') as fh:
            sim_cfg = yaml.load(fh)

        with open(get_resource_name('data/simulated.txt'), 'r') as f:
            data = [line.rstrip().split(' ') for line in f]

        hdc = HDClustering(sim_cfg['model'],
                           opts=dict(tokenizer=None, normalizer=None),
                           get_body=itemgetter(1),
                           get_label=itemgetter(0)
                           )
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
        self.assertGreaterEqual(precision, 0.305)


if __name__ == '__main__':
    unittest.main()
