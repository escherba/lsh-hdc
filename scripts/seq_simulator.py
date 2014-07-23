#!/usr/bin/env python

import random
import string
import json
import argparse
from itertools import chain, izip, repeat, islice
from lsh import Cluster, Shingler
from lsh.stats import FeatureClusterSummarizer, get_stats


def random_string(length=4, alphabet=string.letters):
    """

    :param length: length of the string
    :type length: int
    :param alphabet: alphabet to draw letters from
    :type alphabet: str
    :return: random string of specified length
    :rtype: str
    """
    l = len(alphabet) - 1
    return ''.join(str(alphabet[random.randint(0, l)])
                   for _ in range(length))


def gauss_unsigned(mu=3, sigma=15):
    """
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
    """Draw random value from discrete probability distribution
    represented as a dict: P(x=value) = discrete_prob_dist[value].

    Method: http://en.wikipedia.org/wiki/Pseudo-random_number_sampling

    :param discrete_prob_dist: the probability distribution
    :type discrete_prob_dist: dict
    :returns: scalar drawn from the distribution
    :rtype: object
    """
    limit = 0
    r = random.random()
    for value in discrete_prob_dist:
        limit += discrete_prob_dist[value]
        if r < limit:
            return value


class MarkovChainGenerator:

    def __init__(self, alphabet=string.letters):
        self.alphabet = alphabet
        self.chain = MarkovChainGenerator.get_markov_chain(alphabet)

    def generate(self, length):
        """Generate a string according to a Markov chain"""
        s = [random_string(length=1, alphabet=self.alphabet)]
        for i in range(length - 1):
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
        markov_chain = {}
        for from_letter in alphabet:
            slice_points = sorted([0] + [random.random() for _ in range(l - 1)] + [1])
            transition_probabilities = \
                [slice_points[i + 1] - slice_points[i] for i in range(l)]
            markov_chain[from_letter] = \
                {letter: p for letter, p in zip(alphabet, transition_probabilities)}
        return markov_chain


class MarkovChainMutator:

    delimiter = '-'

    def __init__(self, p_err=0.2, alphabet=string.letters):
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
            slice_points = sorted([0] + [random.uniform(0, p_err) for _ in range(l - 2)]) + [p_err]
            transition_prob = \
                [slice_points[idx + 1] - slice_points[idx] for idx in range(l - 1)] + [1.0 - p_err]
            markov_chain[from_letter] = \
                dict(zip(list(alpha_set - {from_letter}) + [from_letter], transition_prob))
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

    # Shoot for 50% of strings in clusters, 50% outside of clusters
    pos_count = 0
    mcg = MarkovChainGenerator()
    mcm = MarkovChainMutator(p_err=opts.p_err)
    data = []
    for c_id in range(num_clusters):
        cluster_size = gauss_unsigned2(
            threshold=2, mu=c_size_mu, sigma=c_size_sigma)
        seq_length = gauss_unsigned(mu=seq_len_mu, sigma=seq_len_sigma)
        master = mcg.generate(seq_length)
        for seq_id in range(cluster_size):
            data.append(("{}:{}".format(c_id, seq_id), mcm.mutate(master)))
            pos_count += 1
    num_negatives = int(pos_count * (1 - pos_ratio) / pos_ratio)
    for neg_idx in range(num_negatives):
        seq_length = gauss_unsigned(mu=seq_len_mu, sigma=seq_len_sigma)
        data.append(("{}".format(neg_idx), mcg.generate(seq_length)))
    random.shuffle(data)
    return data


def test_simulated(opts, data):
    cluster = Cluster(width=opts.width,
                      bandwidth=opts.bandwidth,
                      lsh_scheme=opts.lsh_scheme)
    shingler = Shingler(span=opts.shingle_span)
    s = FeatureClusterSummarizer()
    content_dict = dict()
    for label, text in data:
        content_dict[label] = text
        shingles = shingler.get_shingles(text)
        s.add_features(label, shingles)
        cluster.add_set(shingles, label)
    clusters = cluster.get_clusters()

    c = get_stats(clusters, lambda x: len(x.split(':')) > 1)
    ti = s.summarize_clusters(clusters)
    print json.dumps(dict(
        stats=c.dict(),
        ratios=dict(
            precision=c.get_precision(),
            recall=c.get_recall()
        ),
        ti=ti
    ))


def do_simulation(args):
    data = get_simulation(args)
    for i, seq in data:
        print i, seq


def do_run_test(args):
    data = get_simulation(args)
    test_simulated(args, data)


if __name__ == '__main__':
        p = argparse.ArgumentParser(
            description="Simulate results and/or run tests on them")
        #p.add_argument('--filter', type=str, required=False, default='None',
        #               help='[TN, FN, FP, TP, None]')
        p.add_argument('--num_clusters', type=int, dest='num_clusters',
                       default=100, help='number of clusters', required=False)
        p.add_argument('--pos_ratio', type=float, dest='pos_ratio', default=0.1,
                       help='ratio of positives', required=False)
        p.add_argument('--p_err', type=float, dest='p_err', default=0.95,
                       help='Probability of error at any location in sequence',
                       required=False)
        p.add_argument('--seq_len_mu', type=float, dest='seq_len_mu',
                       default=3, help='Mean of sequence length',
                       required=False)
        p.add_argument('--seq_len_sigma', type=float, dest='seq_len_sigma', default=10,
                       help='Std. dev. of sequence length', required=False)
        p.add_argument('--c_size_mu', type=float, dest='c_size_mu', default=2,
                       help='Mean of cluster size', required=False)
        p.add_argument('--c_size_sigma', type=float, dest='c_size_sigma', default=10,
                       help='Std. dev. of cluster size', required=False)

        subparsers = p.add_subparsers()
        p_run_tests = subparsers.add_parser('run_test', help='run tests')
        p_run_tests.add_argument('--shingle_span', type=int, dest='shingle_span', default=3,
                                 help='shingle length (in tokens)', required=False)
        p_run_tests.add_argument('--width', type=int, dest='width', default=3,
                                 help='length of minhash feature vectors', required=False)
        p_run_tests.add_argument('--bandwidth', type=int, dest='bandwidth', default=3,
                                 help='rows per band', required=False)
        p_run_tests.add_argument('--lsh_scheme', type=str, dest='lsh_scheme', default="b2",
                                 help='LSH binning scheme',
                                 required=False)
        p_run_tests.set_defaults(func=do_run_test)

        p_simulate = subparsers.add_parser('simulate', help='run tests')
        p_simulate.set_defaults(func=do_simulation)

        args = p.parse_args()
        args.func(args)
        #main(args)
