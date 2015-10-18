import random
import sys
import string
import argparse
from itertools import chain, izip, repeat, islice
from lsh_hdc import Shingler
from lsh_hdc.cluster import MinHashCluster as Cluster
from lsh_hdc.utils import random_string
from sklearn.metrics import homogeneity_completeness_v_measure
from pymaptools.io import GzipFileType


ALPHABET = string.letters + string.digits


def gauss_unsigned(mu, sigma):
    """Draw a positive integer from Gaussian distribution
    :param mu: mean
    :param sigma: std. dev
    :return: positive integer drawn from Gaussian distribution
    :rtype: int
    """
    return abs(int(random.gauss(mu, sigma)))


def gauss_unsigned2(threshold=1, **kwargs):
    result = -1
    while result < threshold:
        result = gauss_unsigned(**kwargs)
    return result


def intersperse(delimiter, seq):
    """Intersperse a sequence with a delimiter

    :param delimiter: scalar
    :type delimiter: object
    :param seq: some iterable sequence
    :type seq: collections.iterable
    :returns: sequence interspersed with a delimiter
    :returns: collections.iterable
    """
    return islice(chain.from_iterable(izip(repeat(delimiter), seq)), 1, None)


def draw(discrete_prob_dist):
    """Draw a random value from a discrete probability distribution
    represented as a dict: P(x=value) = discrete_prob_dist[value].

    Method: http://en.wikipedia.org/wiki/Pseudo-random_number_sampling

    :param discrete_prob_dist: the probability distribution
    :type discrete_prob_dist: dict
    :returns: scalar drawn from the distribution
    :rtype: object
    """
    limit = 0.0
    r = random.random()
    for key in discrete_prob_dist.iterkeys():
        limit += discrete_prob_dist[key]
        if r < limit:
            return key


class MarkovChainGenerator(object):

    def __init__(self, alphabet=ALPHABET):
        self.alphabet = alphabet
        self.chain = MarkovChainGenerator.get_markov_chain(alphabet)

    def generate(self, length):
        """Generate a string according to a Markov chain"""
        s = [random_string(length=1, alphabet=self.alphabet)]
        for _ in xrange(length - 1):
            s.append(str(draw(self.chain[s[-1]])))
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
        for from_letter in alphabet:
            slice_points = sorted([0] + [random.random() for _ in xrange(l - 1)] + [1])
            transition_probabilities = \
                [slice_points[i + 1] - slice_points[i] for i in xrange(l)]
            markov_chain[from_letter] = \
                {letter: p for letter, p in izip(alphabet, transition_probabilities)}
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
        seq_list = list(intersperse(self.delimiter, seq)) + [self.delimiter]
        mutation_site = random.randint(0, len(seq_list) - 1)
        from_letter = seq_list[mutation_site]
        to_letter = draw(self.chain[from_letter])
        seq_list[mutation_site] = to_letter
        return ''.join(el for el in seq_list if el != self.delimiter)


def get_simulation(opts):

    num_clusters = opts.num_clusters
    seq_len_mu = opts.seq_len_mu
    seq_len_sigma = opts.seq_len_sigma
    c_size_mu = opts.c_size_mu
    c_size_sigma = opts.c_size_sigma
    pos_ratio = opts.pos_ratio

    # Aim for 50% of strings in clusters, 50% outside of clusters
    pos_count = 0
    mcg = MarkovChainGenerator()
    mcm = MarkovChainMutator(p_err=opts.p_err)
    data = []
    for c_id in xrange(num_clusters):
        cluster_size = gauss_unsigned2(
            threshold=2, mu=c_size_mu, sigma=c_size_sigma)
        seq_length = gauss_unsigned(mu=seq_len_mu, sigma=seq_len_sigma)
        master = mcg.generate(seq_length)
        for seq_id in xrange(cluster_size):
            data.append(("{}:{}".format(c_id + 1, seq_id), mcm.mutate(master)))
            pos_count += 1
    num_negatives = int(pos_count * ((1.0 - pos_ratio) / pos_ratio))
    for neg_idx in xrange(num_negatives):
        seq_length = gauss_unsigned(mu=seq_len_mu, sigma=seq_len_sigma)
        data.append(("{}".format(neg_idx), mcg.generate(seq_length)))
    random.shuffle(data)
    return data


def get_clusters(opts, data):
    cluster = Cluster(width=opts.width,
                      bandwidth=opts.bandwidth,
                      lsh_scheme=opts.lsh_scheme)
    shingler = Shingler(span=opts.shingle_span)
    content_dict = dict()
    for label, text in data:
        content_dict[label] = text
        shingles = shingler.get_shingles(text)
        cluster.add_item(shingles, label)
    return cluster.get_clusters()


def do_simulation(args):
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
        '--num_clusters', type=int, dest='num_clusters',
        default=1000, help='number of clusters', required=False)
    p_simul.add_argument(
        '--pos_ratio', type=float, dest='pos_ratio', default=0.1,
        help='ratio of positives', required=False)
    p_simul.add_argument(
        '--p_err', type=float, dest='p_err', default=0.10, required=False,
        help='Probability of error at any location in sequence')
    p_simul.add_argument(
        '--seq_len_mu', type=float, dest='seq_len_mu', required=False,
        default=3, help='Mean of sequence length')
    p_simul.add_argument(
        '--seq_len_sigma', type=float, dest='seq_len_sigma', default=10,
        help='Std. dev. of sequence length', required=False)
    p_simul.add_argument(
        '--c_size_mu', type=float, dest='c_size_mu', default=2,
        help='Mean of cluster size', required=False)
    p_simul.add_argument(
        '--c_size_sigma', type=float, dest='c_size_sigma', default=10,
        help='Std. dev. of cluster size', required=False)
    p_simul.set_defaults(func=do_simulation)

    p_clust = subparsers.add_parser('cluster', help='run tests')
    p_clust.add_argument(
        '--input', type=GzipFileType('r'), default=sys.stdin,
        help='File input')
    p_clust.add_argument(
        '--shingle_span', type=int, dest='shingle_span', default=3,
        help='shingle length (in tokens)', required=False)
    p_clust.add_argument(
        '--width', type=int, dest='width', default=3,
        help='length of minhash feature vectors', required=False)
    p_clust.add_argument(
        '--bandwidth', type=int, dest='bandwidth', default=3,
        help='rows per band', required=False)
    p_clust.add_argument(
        '--lsh_scheme', type=str, dest='lsh_scheme', default="b2",
        help='LSH binning scheme', required=False)
    p_clust.set_defaults(func=do_cluster)

    namespace = parser.parse_args()
    return namespace


def run(args):
    args.func(args)


if __name__ == '__main__':
    run(parse_args())
