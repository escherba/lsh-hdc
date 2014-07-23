import unittest
import os
import sys
import json
from itertools import islice
from lsh import Cluster, Shingler, MinHashSignature, LSHC, \
    SimHashSignature, RegexTokenizer, Normalizer
from lsh.stats import FeatureClusterSummarizer, get_stats


def abs_path(rel_path):
    return os.path.join(os.path.dirname(__file__), rel_path)


class TestFiles(unittest.TestCase):

    def test_unicode_tokenizer(self):
        """Expect to match number of clusters using simhash"""
        minhash_signer = MinHashSignature(6, lsh_hasher=LSHC(2, width=6))
        simhash_signer = SimHashSignature(63)

        container = []
        with open(abs_path('data/unicode.json'), 'r') as fh:
            for line in fh:
                container.append(json.loads(line))

        self.assertEqual(len(container), 7)

        feature_extractor = Shingler(span=3,
                                     tokenizer=RegexTokenizer(),
                                     normalizer=Normalizer())

        for post in container:
            post_obj = post['object']
            content = post_obj['content']
            features = feature_extractor.get_shingles(content)
            minhash_keys = minhash_signer.get_signature(features)
            simhash = simhash_signer.get_signature(features)
            print "ok"
        # TODO: finish writing this test

    def test_names(self):
        """Should return 281 clusters of names.
        """
        with open(abs_path('data/perrys.csv'), 'r') as f:
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
        with open(abs_path('data/perrys.csv'), 'r') as f:
            data = set(line.rstrip() for line in f)
        cluster = Cluster(width=20, bandwidth=5, kmin=2)
        shingler = Shingler(3)
        s = FeatureClusterSummarizer()
        for name in data:
            shingles = shingler.get_shingles(name)
            s.add_features(name, shingles)
            cluster.add_set(shingles, name)
        clusters = cluster.get_clusters()
        #for cluster in clusters:
        #    print cluster
        ti = s.summarize_clusters(clusters)
        self.assertEqual(len(clusters), 252)
        self.assertAlmostEqual(ti, 0.9732840816954408)

    def test_names_kmin_scheme(self):
        """Should return 145 clusters of names.
        """
        with open(abs_path('data/perrys.csv'), 'r') as f:
            data = set(line.rstrip() for line in f)
        cluster = Cluster(width=20, bandwidth=5, kmin=2, lsh_scheme="a1")
        shingler = Shingler(3)
        s = FeatureClusterSummarizer()
        for name in data:
            shingles = shingler.get_shingles(name)
            s.add_features(name, shingles)
            cluster.add_set(shingles, name)
        clusters = cluster.get_clusters()
        #for cluster in clusters:
        #    print cluster
        ti = s.summarize_clusters(clusters)
        self.assertEqual(len(clusters), 145)
        self.assertAlmostEqual(ti, 0.9693895180932199)

    def test_bills(self):
        """Should return 97 clusters of bills.
        """
        with open(abs_path('data/bills100.txt'), 'r') as f:
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
    def run_simulated_manually(filepath, universe_size=None, lines_to_read=sys.maxint):
        with open(abs_path(filepath), 'r') as f:
            data = [line.rstrip().split(' ')
                    for line in islice(f, lines_to_read)]
        cluster = Cluster(width=20, bandwidth=5, lsh_scheme="a2", universe_size=universe_size)
        shingler = Shingler(span=3)
        s = FeatureClusterSummarizer()
        content_dict = dict()
        for pair in data:
            if len(pair) > 1:
                label, text = pair
            else:
                label = pair[0]
                text = ''
            content_dict[label] = text
            shingles = shingler.get_shingles(text)
            s.add_features(label, shingles)
            cluster.add_set(shingles, label)
        clusters = cluster.get_clusters()

        is_label_positive = lambda lbl: len(lbl.split(':')) > 1
        return dict(stats=get_stats(clusters, is_label_positive),
                    uindex=s.summarize_clusters(clusters))

    @staticmethod
    def run_simulated_manually_b(filepath, universe_size=None, lines_to_read=sys.maxint):
        with open(abs_path(filepath), 'r') as f:
            data = [line.rstrip().split(' ')
                    for line in islice(f, lines_to_read)]
        cluster = Cluster(width=15, bandwidth=3, lsh_scheme="b3", kmin=3, universe_size=universe_size)
        shingler = Shingler(span=3)
        s = FeatureClusterSummarizer()
        content_dict = dict()
        for pair in data:
            if len(pair) > 1:
                label, text = pair
            else:
                label = pair[0]
                text = ''
            content_dict[label] = text
            shingles = shingler.get_shingles(text)
            s.add_features(label, shingles)
            cluster.add_set(shingles, label)
        clusters = cluster.get_clusters()

        is_label_positive = lambda lbl: len(lbl.split(':')) > 1
        return dict(stats=get_stats(clusters, is_label_positive),
                    uindex=s.summarize_clusters(clusters))

    def test_simulated(self, universe_size=None):
        results = TestFiles.run_simulated_manually('data/simulated.txt',
                                                   lines_to_read=1000,
                                                   universe_size=universe_size)
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
        results = TestFiles.run_simulated_manually_b('data/simulated.txt',
                                                     lines_to_read=1000,
                                                     universe_size=universe_size)
        c = results['stats']
        ti = results['uindex']
        recall = c.get_recall()
        precision = c.get_precision()
        self.assertGreaterEqual(recall, 0.10)
        self.assertGreaterEqual(precision, 0.09836065573770492)
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
