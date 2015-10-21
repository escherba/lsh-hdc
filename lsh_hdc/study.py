import random
import os
import sys
import string
import operator
import logging
import json
from collections import OrderedDict
from itertools import izip, cycle
from lsh_hdc import Shingler, HASH_FUNC_TABLE
from lsh_hdc.cluster import MinHashCluster as Cluster
from lsh_hdc.utils import random_string
from pymaptools.io import GzipFileType, read_json_lines, ndjson2col, \
    PathArgumentParser
from pymaptools.iter import intersperse
from pymaptools.sample import discrete_sample
from pymaptools.benchmark import PMTimer


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


class MarkovChainGenerator(object):

    def __init__(self, alphabet=ALPHABET):
        self.alphabet = alphabet
        self.chain = MarkovChainGenerator.get_markov_chain(alphabet)

    def generate(self, start, length):
        """Generate a sequence according to a Markov chain"""
        for _ in xrange(length):
            prob_dist = self.chain[start]
            start = discrete_sample(prob_dist)
            yield start

    def generate_str(self, start, length):
        """Generate a string according to a Markov chain"""
        return ''.join(self.generate(start, length))

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
    seq_len_min = args.seq_len_min

    pos_count = 0
    mcg = MarkovChainGenerator()
    mcm = MarkovChainMutator(p_err=args.p_err)
    data = []

    stats = dict()

    # pick first letter at random
    start = random_string(length=1, alphabet=mcg.alphabet)

    if (args.num_clusters is not None) and (args.pos_ratio is not None):
        num_clusters = args.num_clusters
        stats['num_clusters'] = num_clusters
        pos_ratio = args.pos_ratio
    elif (args.num_clusters is not None) and (args.sim_size is not None) and (args.cluster_size is not None) and (args.pos_ratio is None):
        num_clusters = args.num_clusters
        stats['num_clusters'] = num_clusters
        stats['cluster_size'] = args.cluster_size
        pos_ratio = int(args.num_clusters * args.cluster_size / float(args.sim_size))
        logging.info("Setting positive/total ratio to %.3f", pos_ratio)
    elif (args.num_clusters is None) and (args.sim_size is not None) and (args.cluster_size is not None) and (args.pos_ratio is not None):
        # calculate from simulation size
        stats['cluster_size'] = args.cluster_size
        num_clusters = int(args.sim_size * args.pos_ratio / float(args.cluster_size))
        stats['num_clusters'] = num_clusters
        logging.info("Creating %d clusters of size %d", num_clusters, args.cluster_size)
        pos_ratio = args.pos_ratio
    else:
        raise RuntimeError("Could not compute num_clusters and pos_ratio")

    for c_id in xrange(num_clusters):
        if args.cluster_size is None:
            cluster_size = gauss_uint_threshold(
                threshold=2, mu=c_size_mu, sigma=c_size_sigma)
        else:
            cluster_size = args.cluster_size
        seq_length = gauss_uint_threshold(
            threshold=seq_len_min, mu=seq_len_mu, sigma=seq_len_sigma)
        master = mcg.generate_str(start, seq_length)
        if len(master) > 0:
            start = master[-1]
        for seq_id in xrange(cluster_size):
            data.append(("{}:{}".format(c_id + 1, seq_id), mcm.mutate(master)))
            pos_count += 1
    stats['num_positives'] = pos_count
    num_negatives = int(pos_count * ((1.0 - pos_ratio) / pos_ratio))
    for neg_idx in xrange(num_negatives):
        seq_length = gauss_uint_threshold(
            threshold=seq_len_min, mu=seq_len_mu, sigma=seq_len_sigma)
        master = mcg.generate_str(start, seq_length)
        if len(master) > 0:
            start = master[-1]
        data.append(("{}".format(neg_idx), master))
    logging.info("Positives: %d, Negatives: %d", pos_count, num_negatives)
    stats['num_negatives'] = num_negatives
    random.shuffle(data)
    return data, stats


def get_clusters(args, data):
    cluster = Cluster(width=args.width,
                      bandwidth=args.bandwidth,
                      lsh_scheme=args.lsh_scheme,
                      hashfun=args.hashfun)
    shingler = Shingler(span=args.shingle_span)
    content_dict = dict()
    for label, text in data:
        content_dict[label] = text
        shingles = shingler.get_shingles(text)
        cluster.add_item(shingles, label)
    return cluster.get_clusters()


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


def cluster_predictions(cluster_iter):
    y_true = []
    y_score = []
    for idx, cluster in enumerate(cluster_iter):
        pred_cluster = len(cluster)
        for point in cluster:
            true_cluster = ':' in point
            y_true.append(true_cluster)
            y_score.append(pred_cluster)
    return y_true, y_score


def do_simulation(args):
    if args.seed is not None:
        random.seed(args.seed)
    data, _ = get_simulation(args)
    output = args.output
    for i, seq in data:
        output.write("%s %s\n", (i, seq))


METRICS = [
    'homogeneity', 'completeness', 'nmi_score',
    'adj_rand_score', 'adj_nmi_score', 'roc_auc',
    'time_wall', 'time_cpu'
]

CLUSTER_METRICS_ALL = ['homogeneity', 'completeness', 'nmi_score', 'adj_rand_score', 'adj_nmi_score']
CLUSTER_METRICS = ['homogeneity', 'completeness', 'nmi_score']
ROC_METRICS = ['roc_auc']


def add_cluster_metrics(args, clusters, pairs):
    if (set(CLUSTER_METRICS_ALL) & set(args.metrics)):
        cluster_data = clusters_to_labels(clusters)
        if (set(CLUSTER_METRICS) & set(args.metrics)):
            from sklearn.metrics import homogeneity_completeness_v_measure
            pairs.extend(zip(CLUSTER_METRICS, homogeneity_completeness_v_measure(*cluster_data)))
        if 'adj_rand_score' in args.metrics:
            from sklearn.metrics import adjusted_rand_score
            pairs.append(('adj_rand_score', adjusted_rand_score(*cluster_data)))
        if 'adj_nmi_score' in args.metrics:
            from sklearn.metrics import adjusted_mutual_info_score
            pairs.append(('adj_nmi_score', adjusted_mutual_info_score(*cluster_data)))


def add_roc_metrics(args, clusters, pairs):
    if (set(ROC_METRICS) & set(args.metrics)):
        from sklearn.metrics import roc_auc_score
        roc_data = cluster_predictions(clusters)
        if 'roc_auc' in args.metrics:
            pairs.append(('roc_auc', roc_auc_score(*roc_data)))


def add_timer_metrics(args, timer, pairs):
    if 'time_wall' in args.metrics:
        pairs.append(('time_wall', timer.wall_interval))
    if 'time_cpu' in args.metrics:
        pairs.append(('time_cpu', timer.clock_interval))


def perform_clustering(args, data):
    with PMTimer() as timer:
        clusters = get_clusters(args, data)
    pairs = []
    pairs.append(('hash_function', args.hashfun))
    add_timer_metrics(args, timer, pairs)
    add_cluster_metrics(args, clusters, pairs)
    add_roc_metrics(args, clusters, pairs)
    return dict(pairs)


def do_cluster(args):
    results = perform_clustering(args, load_simulation(args))
    args.output.write("%s\n" % json.dumps(results))


def do_joint(args):
    if args.seed is not None:
        random.seed(args.seed)
    data, stats = get_simulation(args)
    results = perform_clustering(args, data)
    stats.update(results)
    args.output.write("%s\n" % json.dumps(stats))


def create_plots(args, df, metrics):
    import matplotlib.pyplot as plt
    from palettable import colorbrewer
    from matplotlib.font_manager import FontProperties
    fontP = FontProperties()
    fontP.set_size('small')

    groups = df.groupby(["hash_function"])
    palette_size = min(max(len(groups), 3), 9)
    for column in metrics:
        if column in df:
            colors = cycle(colorbrewer.get_map('Set1', 'qualitative', palette_size).mpl_colors)
            fig, ax = plt.subplots()
            for color, (label, dfel) in izip(colors, groups):
                dfel.plot(ax=ax, label=label, x="cluster_size", linewidth='1.3',
                          y=column, kind="scatter", logx=True, title=args.fig_title,
                          facecolors='none', edgecolors=color)
            fig_filename = "fig_%s.%s" % (column, args.fig_format)
            fig_path = os.path.join(args.output, fig_filename)
            ax.legend(prop=fontP)
            fig.savefig(fig_path)


def do_summa(args):
    import pandas as pd

    obj = ndjson2col(read_json_lines(args.input))
    df = pd.DataFrame.from_dict(obj)
    csv_path = os.path.join(args.output, "summary.csv")
    logging.info("Writing output summary to %s", csv_path)
    df.to_csv(csv_path)
    create_plots(args, df, METRICS)


def add_simul_args(p_simul):
    p_simul.add_argument(
        '--num_clusters', type=int, default=None,
        help='Number of clusters to create')
    p_simul.add_argument(
        '--sim_size', type=int, default=None,
        help='Simulation size (when number of clusters is not given)')
    p_simul.add_argument(
        '--cluster_size', type=int, default=None,
        help='cluster size')
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
        '--seq_len_min', type=int, default=2,
        help='Minimum sequence length')
    p_simul.add_argument(
        '--seq_len_mu', type=float, default=4,
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


def add_clust_args(p_clust):
    p_clust.add_argument(
        '--hashfun', type=str, default='metrohash',
        choices=HASH_FUNC_TABLE.keys(),
        help='Minimum sequence length')
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
    p_clust.add_argument(
        '--metrics', type=str, nargs='*', default=('nmi_score', 'time_cpu'),
        choices=METRICS, help='Which metrics to calculate')


def parse_args(args=None):
    parser = PathArgumentParser(
        description="Simulate data and/or run analysis")

    parser.add_argument(
        '--logging', type=str, default='WARN', help="Logging level",
        choices=[key for key in logging._levelNames.keys() if isinstance(key, str)])

    subparsers = parser.add_subparsers()

    p_simul = subparsers.add_parser('simulate', help='generate simulation')
    add_simul_args(p_simul)
    p_simul.set_defaults(func=do_simulation)

    p_clust = subparsers.add_parser('analyze', help='run analysis')
    p_clust.add_argument(
        '--input', type=GzipFileType('r'), default=sys.stdin, help='File input')
    add_clust_args(p_clust)
    p_clust.add_argument(
        '--output', type=GzipFileType('w'), default=sys.stdout, help='File output')
    p_clust.set_defaults(func=do_cluster)

    p_joint = subparsers.add_parser('joint', help='generate simulation and analyze')
    add_simul_args(p_joint)
    add_clust_args(p_joint)
    p_joint.add_argument(
        '--output', type=GzipFileType('w'), default=sys.stdout, help='File output')
    p_joint.set_defaults(func=do_joint)

    p_summa = subparsers.add_parser('summary', help='summarize analysis results')
    p_summa.add_argument(
        '--input', type=GzipFileType('r'), default=sys.stdin, help='File input')
    p_summa.add_argument(
        '--fig_title', type=str, default=None, help='Title (for figures generated)')
    p_summa.add_argument(
        '--fig_format', type=str, default='svg', help='Figure format')
    p_summa.add_argument(
        '--output', type=str, metavar='DIR', help='Output directory')
    p_summa.set_defaults(func=do_summa)

    namespace = parser.parse_args()
    return namespace


def run(args):
    logging.basicConfig(level=getattr(logging, args.logging))
    args.func(args)


if __name__ == '__main__':
    run(parse_args())
