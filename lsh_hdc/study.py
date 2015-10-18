import random
import sys
import string
import argparse
import operator
from collections import OrderedDict
from itertools import izip
from lsh_hdc import Shingler
from lsh_hdc.cluster import MinHashCluster as Cluster
from lsh_hdc.utils import random_string
from sklearn.metrics import homogeneity_completeness_v_measure
from pymaptools.io import GzipFileType
from pymaptools.iter import intersperse


ALPHABET = string.letters + string.digits


def gauss_uint(mu, sigma):
    """Draw a positive integer from Gaussian distribution
    :param mu: mean
    :param sigma: std. dev
    :return: positive integer drawn from Gaussian distribution
    :rtype: int
    """
    return abs(int(random.gauss(mu, sigma)))


def gauss_uint_threshold(threshold=1, **kwargs):
    result = -1
    while result < threshold:
        result = gauss_uint(**kwargs)
    return result


def discrete_sample(prob_dist):
    """Sample a random value from a discrete probability distribution
    represented as a dict: P(x=value) = prob_dist[value].

    Note: the prob_dist parameter doesn't have to be an ordered dict,
    however for performance reasons it is best if it is.

    :param prob_dist: the probability distribution
    :type prob_dist: collections.Mapping
    :returns: scalar drawn from the distribution
    :rtype: object
    """
    limit = 0.0
    r = random.random()
    for key, val in prob_dist.iteritems():
        limit += val
        if r <= limit:
            return key


class MarkovChainGenerator(object):

    def __init__(self, alphabet=ALPHABET):
        self.alphabet = alphabet
        self.chain = MarkovChainGenerator.get_markov_chain(alphabet)

    def generate(self, length):
        """Generate a string according to a Markov chain"""
        s = [random_string(length=1, alphabet=self.alphabet)]
        for _ in xrange(length - 1):
            prob_dist = self.chain[s[-1]]
            s.append(str(discrete_sample(prob_dist)))
        return ''.join(s)

    @staticmethod
    def get_markov_chain(alphabet):
        """
        :param alphabet: letters to use
        :type alphabet: str
        :return: transition probabilities
        :rtype: dict
        """
        l = len(alphabet)
        markov_chain = dict()
        second = operator.itemgetter(1)
        for from_letter in alphabet:
            slice_points = sorted([0] + [random.random() for _ in xrange(l - 1)] + [1])
            transition_probabilities = \
                [slice_points[i + 1] - slice_points[i] for i in xrange(l)]
            letter_probs = sorted(izip(alphabet, transition_probabilities),
                                  key=second, reverse=True)
            markov_chain[from_letter] = OrderedDict(letter_probs)
        return markov_chain


class MarkovChainMutator(object):

    delimiter = '-'

    def __init__(self, p_err=0.2, alphabet=ALPHABET):
        self.alphabet = alphabet
        self.chain = MarkovChainMutator.get_markov_chain(alphabet + self.delimiter, p_err=p_err)

    @staticmethod
    def get_markov_chain(alphabet, p_err=0.2):
        """
        :param p_err: probability of an error
        :type p_err: float
        :param alphabet: letters to use
        :type alphabet: str
        :return: transition probabilities
        :rtype: dict
        """
        markov_chain = dict()
        alpha_set = set(alphabet)
        l = len(alpha_set)
        for from_letter in alpha_set:
            slice_points = sorted([0] + [random.uniform(0, p_err) for _ in xrange(l - 2)]) + [p_err]
            transition_prob = \
                [slice_points[idx + 1] - slice_points[idx] for idx in xrange(l - 1)] + [1.0 - p_err]
            markov_chain[from_letter] = \
                dict(izip(list(alpha_set - {from_letter}) + [from_letter], transition_prob))
        return markov_chain

    def mutate(self, seq):
        """
        :param seq: sequence
        :type seq: str
        :returns: mutated sequence
        :rtype: str
        """
        delimiter = self.delimiter
        seq_list = list(intersperse(delimiter, seq)) + [delimiter]
        mutation_site = random.randint(0, len(seq_list) - 1)
        from_letter = seq_list[mutation_site]
        prob_dist = self.chain[from_letter]
        to_letter = discrete_sample(prob_dist)
        seq_list[mutation_site] = to_letter
        return ''.join(el for el in seq_list if el != delimiter)


def get_simulation(args):

    seq_len_mu = args.seq_len_mu
    seq_len_sigma = args.seq_len_sigma
    c_size_mu = args.c_size_mu
    c_size_sigma = args.c_size_sigma

    pos_count = 0
    mcg = MarkovChainGenerator()
    mcm = MarkovChainMutator(p_err=args.p_err)
    data = []
    for c_id in xrange(args.num_clusters):
        cluster_size = gauss_uint_threshold(
            threshold=2, mu=c_size_mu, sigma=c_size_sigma)
        seq_length = gauss_uint(mu=seq_len_mu, sigma=seq_len_sigma)
        master = mcg.generate(seq_length)
        for seq_id in xrange(cluster_size):
            data.append(("{}:{}".format(c_id + 1, seq_id), mcm.mutate(master)))
            pos_count += 1
    num_negatives = int(pos_count * (1.0 - args.pos_ratio) / args.pos_ratio)
    for neg_idx in xrange(num_negatives):
        seq_length = gauss_uint(mu=seq_len_mu, sigma=seq_len_sigma)
        data.append(("{}".format(neg_idx), mcg.generate(seq_length)))
    random.shuffle(data)
    return data


def get_clusters(args, data):
    cluster = Cluster(width=args.width,
                      bandwidth=args.bandwidth,
                      lsh_scheme=args.lsh_scheme)
    shingler = Shingler(span=args.shingle_span)
    content_dict = dict()
    for label, text in data:
        content_dict[label] = text
        shingles = shingler.get_shingles(text)
        cluster.add_item(shingles, label)
    return cluster.get_clusters()


def do_simulation(args):
    if args.seed is not None:
        random.seed(args.seed)
    data = get_simulation(args)
    for i, seq in data:
        print i, seq


def load_simulation(args):
    for line in args.input:
        label, text = line.split(" ")
        yield (label, text.strip())


def clusters_to_labels(cluster_iter):
    labels_true = []
    labels_pred = []
    for idx, cluster in enumerate(cluster_iter):
        if len(cluster) == 1:
            pred_cluster = 0
        else:
            pred_cluster = idx
        for point in cluster:
            if ':' in point:
                true_cluster, _ = point.split(':')
                true_cluster = int(true_cluster)
            else:
                true_cluster = 0
            labels_true.append(true_cluster)
            labels_pred.append(pred_cluster)
    return labels_true, labels_pred


def do_cluster(args):
    clusters = get_clusters(args, load_simulation(args))
    labels_true, labels_pred = clusters_to_labels(clusters)
    print """
Homogeneity:  %.3f
Completeness: %.3f
V-measure:    %.3f
""" % homogeneity_completeness_v_measure(labels_true, labels_pred)


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Simulate results and/or run tests on them")
    subparsers = parser.add_subparsers()

    p_simul = subparsers.add_parser('simulate', help='run tests')
    p_simul.add_argument(
        '--num_clusters', type=int, default=1000,
        help='number of clusters')
    p_simul.add_argument(
        '--seed', type=int, default=None,
        help='Random number generator seed for reproducibility')
    p_simul.add_argument(
        '--pos_ratio', type=float, default=0.1,
        help='ratio of positives')
    p_simul.add_argument(
        '--p_err', type=float, default=0.10,
        help='Probability of error at any location in sequence')
    p_simul.add_argument(
        '--seq_len_mu', type=float, default=3,
        help='Mean of sequence length')
    p_simul.add_argument(
        '--seq_len_sigma', type=float, default=10,
        help='Std. dev. of sequence length')
    p_simul.add_argument(
        '--c_size_mu', type=float, default=2,
        help='Mean of cluster size')
    p_simul.add_argument(
        '--c_size_sigma', type=float, default=10,
        help='Std. dev. of cluster size')
    p_simul.set_defaults(func=do_simulation)

    p_clust = subparsers.add_parser('cluster', help='run tests')
    p_clust.add_argument(
        '--input', type=GzipFileType('r'), default=sys.stdin,
        help='File input')
    p_clust.add_argument(
        '--shingle_span', type=int, default=3,
        help='shingle length (in tokens)')
    p_clust.add_argument(
        '--width', type=int, default=3,
        help='length of minhash feature vectors')
    p_clust.add_argument(
        '--bandwidth', type=int, default=3,
        help='rows per band')
    p_clust.add_argument(
        '--lsh_scheme', type=str, default="b2",
        help='LSH binning scheme')
    p_clust.set_defaults(func=do_cluster)

    namespace = parser.parse_args()
    return namespace


def run(args):
    args.func(args)


if __name__ == '__main__':
    run(parse_args())
