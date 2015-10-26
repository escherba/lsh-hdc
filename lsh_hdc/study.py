"""
Simulation-generated data can provide an external criterion to validate
clustering methods. This module contains a set of command-line tools for
performing simulations, clustering their output, and producing analaysis
reports.
"""


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
    PathArgumentParser, write_json_line
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

    def __init__(self, p_err=0.1, alphabet=ALPHABET):
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
        doc_list = list(intersperse(delimiter, seq)) + [delimiter]
        mutation_site = random.randint(0, len(doc_list) - 1)
        from_letter = doc_list[mutation_site]
        prob_dist = self.chain[from_letter]
        to_letter = discrete_sample(prob_dist)
        doc_list[mutation_site] = to_letter
        return ''.join(el for el in doc_list if el != delimiter)


def perform_simulation(args):

    doc_len_mean = args.doc_len_mean
    doc_len_sigma = args.doc_len_sigma
    c_size_mean = args.c_size_mean
    c_size_sigma = args.c_size_sigma
    doc_len_min = args.doc_len_min

    pos_count = 0
    mcg = MarkovChainGenerator()
    mcm = MarkovChainMutator(p_err=args.p_err)
    data = []

    stats = dict()

    # pick first letter at random
    start = random_string(length=1, alphabet=mcg.alphabet)

    positive_ratio = args.pos_ratio
    cluster_size = args.cluster_size
    simulation_size = args.sim_size

    if cluster_size is None:
        # generate some cluster sizes until we approximately reach pos_ratio
        current_pos = 0
        expected_pos = positive_ratio * simulation_size
        cluster_sizes = []
        num_clusters = 0
        while current_pos < expected_pos:
            cluster_size = gauss_uint_threshold(
                threshold=2, mu=c_size_mean, sigma=c_size_sigma)
            cluster_sizes.append(cluster_size)
            current_pos += cluster_size
            num_clusters += 1
        logging.info("Creating %d variable-length clusters", num_clusters)
    else:
        # calculate from simulation size
        stats['cluster_size'] = cluster_size
        num_clusters = int(simulation_size * positive_ratio / float(cluster_size))
        cluster_sizes = [cluster_size] * num_clusters
        logging.info("Creating %d clusters of size %d", num_clusters, cluster_size)

    stats['num_clusters'] = num_clusters

    for c_id, cluster_size in enumerate(cluster_sizes):
        doc_length = gauss_uint_threshold(
            threshold=doc_len_min, mu=doc_len_mean, sigma=doc_len_sigma)
        master = mcg.generate_str(start, doc_length)
        if len(master) > 0:
            start = master[-1]
        for doc_id in xrange(cluster_size):
            data.append(("{}:{}".format(c_id + 1, doc_id), mcm.mutate(master)))
            pos_count += 1
    stats['num_positives'] = pos_count
    num_negatives = max(0, simulation_size - pos_count)
    for neg_idx in xrange(num_negatives):
        doc_length = gauss_uint_threshold(
            threshold=doc_len_min, mu=doc_len_mean, sigma=doc_len_sigma)
        master = mcg.generate_str(start, doc_length)
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
                      kmin=args.kmin,
                      hashfun=args.hashfun)
    shingler = Shingler(
        span=args.shingle_span,
        skip=args.shingle_skip,
        kmin=args.shingle_kmin,
        unique=bool(args.shingle_uniq)
    )
    content_dict = dict()
    for label, text in data:
        content_dict[label] = text
        shingles = shingler.get_shingles(text)
        cluster.add_item(shingles, label)
    return cluster.get_clusters()


def load_simulation(args):

    def iter_simulation(sim_iter):
        for line in sim_iter:
            label, text = line.split(" ")
            yield (label, text.strip())

    iterator = args.input
    namespace = json.loads(iterator.next())
    return namespace, iter_simulation(iterator)


def load_clustering(args):

    def iter_clustering(clust_iter):
        for line in clust_iter:
            yield json.loads(line)

    iterator = args.input
    namespace = json.loads(iterator.next())
    return namespace, iter_clustering(iterator)


def is_pos(x):
    return ':' in x


def clusters_to_labels(cluster_iter):
    labels_true = []
    labels_pred = []
    for idx, cluster in enumerate(cluster_iter):
        if len(cluster) == 1:
            pred_cluster = 0
        else:
            pred_cluster = idx
        for point in cluster:
            if is_pos(point):
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
            true_cluster = is_pos(point)
            y_true.append(true_cluster)
            y_score.append(pred_cluster)
    return y_true, y_score


def serialize_args(args):
    namespace = dict(args.__dict__)
    fields_to_delete = ["input", "output", "func", "logging"]
    for field in fields_to_delete:
        try:
            del namespace[field]
        except KeyError:
            pass
    return namespace


def do_simulation(args):
    if args.seed is not None:
        random.seed(args.seed)
    data, stats = perform_simulation(args)
    namespace = serialize_args(args)
    namespace.update(stats)
    output = args.output
    write_json_line(output, namespace)
    for i, seq in data:
        output.write("%s %s\n" % (i, seq))


BENCHMARKS = ['time_cpu']

ENTROPY_METRICS = ['homogeneity', 'completeness', 'nmi_score']
CONF_ENTROPY_METRICS = ['conf_' + m for m in ENTROPY_METRICS]
CONFUSION_METRICS = [
    # best
    'adj_rand_score',
    'g_corr', 'g_corr_row', 'g_corr_col',
    'matthews_corr', 'informedness', 'markedness',
    # ok
    'yule_coeff', 'accuracy',
    # not good
    'jaccard_coeff', 'fscore'
] + CONF_ENTROPY_METRICS

INCIDENCE_METRICS = CONFUSION_METRICS + ENTROPY_METRICS

ROC_METRICS = ['roc_max_info', 'roc_auc']
LIFT_METRICS = ['aul_score']

METRICS = ROC_METRICS + LIFT_METRICS + INCIDENCE_METRICS + BENCHMARKS

LEGEND_METRIC_KWARGS = {
    'homogeneity': dict(loc='lower right'),
    'roc_auc': dict(loc='lower right'),
    'roc_max_info': dict(loc='lower right'),
    'aul_score': dict(loc='lower right'),
    'adj_rand_score': dict(loc='lower right'),
    'time_wall': dict(loc='upper left'),
    'time_cpu': dict(loc='upper left'),
}


def add_incidence_metrics(args, clusters, pairs):
    """Add metrics based on incidence matrix of classes and clusters
    """
    args_metrics = INCIDENCE_METRICS  # args.metrics
    if (set(INCIDENCE_METRICS) & set(args_metrics)):

        from lsh_hdc.metrics import ClusteringMetrics
        cm = ClusteringMetrics.from_labels(*clusters_to_labels(clusters))

        if (set(ENTROPY_METRICS) & set(args_metrics)):
            pairs.extend(zip(ENTROPY_METRICS, cm.entropy_metrics()))

        if (set(CONFUSION_METRICS) & set(args_metrics)):
            conf = cm.confusion_matrix_

            # the coefficients below are arguably the best

            if (set(CONF_ENTROPY_METRICS) & set(args_metrics)):
                pairs.extend(zip(CONF_ENTROPY_METRICS, conf.entropy_metrics()))

            if 'adj_rand_score' in args_metrics:
                pairs.append(('adj_rand_score', conf.kappa()))

            if 'matthews_corr' in args_metrics:
                pairs.append(('matthews_corr', conf.matthews_corr()))
            if 'informedness' in args_metrics:
                pairs.append(('informedness', conf.informedness()))
            if 'markedness' in args_metrics:
                pairs.append(('markedness', conf.markedness()))

            if 'g_corr' in args_metrics:
                pairs.append(('g_corr', conf.g_corr()))
            if 'g_corr_row' in args_metrics:
                pairs.append(('g_corr_row', conf.g_corr_row()))
            if 'g_corr_col' in args_metrics:
                pairs.append(('g_corr_col', conf.g_corr_col()))

            # coefficients below are not corrected for chance
            if 'accuracy' in args_metrics:
                pairs.append(('accuracy', conf.accuracy()))
            if 'yule_coeff' in args_metrics:
                pairs.append(('yule_coeff', conf.yule_coeff()))

            # coefficients below don't consider true negatives
            if 'fscore' in args_metrics:
                pairs.append(('fscore', conf.fscore()))
            if 'jaccard_coeff' in args_metrics:
                pairs.append(('jaccard_coeff', conf.jaccard_coeff()))


def add_roc_metrics(args, clusters, pairs):
    """Add metrics based on ROC Curve
    """
    if (set(ROC_METRICS) & set(args.metrics)):
        from lsh_hdc.metrics import RocCurve
        rc = RocCurve.from_binary(*cluster_predictions(clusters))
        if 'roc_auc' in args.metrics:
            pairs.append(('roc_auc', rc.auc_score()))
        if 'roc_max_info' in args.metrics:
            pairs.append(('roc_max_info', rc.max_informedness()))


def add_lift_metrics(args, clusters, pairs):
    """Add metrics based on Lift Curve
    """
    if (set(LIFT_METRICS) & set(args.metrics)):
        from lsh_hdc.metrics import clustering_aul_score as aul_score
        if 'aul_score' in args.metrics:
            pairs.append(('aul_score', aul_score(clusters, is_pos)))


def perform_clustering(args, data):
    with PMTimer() as timer:
        clusters = get_clusters(args, data)
    return clusters, timer.to_dict()


def perform_analysis(args, clusters):
    clusters = list(clusters)
    pairs = []
    add_lift_metrics(args, clusters, pairs)
    add_incidence_metrics(args, clusters, pairs)
    add_roc_metrics(args, clusters, pairs)
    return dict(pairs)


def do_cluster(args):
    namespace = {}
    sim_namespace, simulation = load_simulation(args)
    namespace.update(sim_namespace)
    clustering_results, clustering_stats = perform_clustering(args, simulation)
    clustering_namespace = serialize_args(args)
    namespace.update(clustering_namespace)
    namespace.update(clustering_stats)
    write_json_line(args.output, namespace)
    for cluster in clustering_results:
        write_json_line(args.output, cluster)


def do_analyze(args):
    namespace = {}
    clustering_namespace, clustering = load_clustering(args)
    namespace.update(clustering_namespace)
    analysis_stats = perform_analysis(args, clustering)
    namespace.update(analysis_stats)
    write_json_line(args.output, namespace)


def create_plots(args, df, metrics):
    import matplotlib.pyplot as plt
    from palettable import colorbrewer
    from matplotlib.font_manager import FontProperties
    fontP = FontProperties()
    fontP.set_size('small')

    groups = df.groupby([args.group_by])
    palette_size = min(max(len(groups), 3), 9)
    for metric in metrics:
        if metric in df:
            colors = cycle(colorbrewer.get_map('Set1', 'qualitative', palette_size).mpl_colors)
            fig, ax = plt.subplots()
            for color, (label, dfel) in izip(colors, groups):
                try:
                    dfel.plot(
                        ax=ax, label=label, x=args.x_axis, linewidth='1.3',
                        y=metric, kind="scatter", logx=True, title=args.fig_title,
                        facecolors='none', edgecolors=color)
                except Exception:
                    logging.exception("Exception caught plotting %s:%s", metric, label)
            fig_filename = "fig_%s.%s" % (metric, args.fig_format)
            fig_path = os.path.join(args.output, fig_filename)
            ax.legend(prop=fontP, **LEGEND_METRIC_KWARGS.get(metric, {}))
            fig.savefig(fig_path)


def do_simul_clust_analy(args):
    if args.seed is not None:
        random.seed(args.seed)
    namespace = serialize_args(args)
    simulation, simulation_stats = perform_simulation(args)
    namespace.update(simulation_stats)
    clustering, clustering_stats = perform_clustering(args, simulation)
    namespace.update(clustering_stats)
    analysis_stats = perform_analysis(args, clustering)
    namespace.update(analysis_stats)
    args.output.write("%s\n" % json.dumps(namespace))


def create_df_subset(df, fields):
    subset_fields = [field for field in fields if field in df]
    return df[subset_fields]


def do_summa(args):
    import pandas as pd
    obj = ndjson2col(read_json_lines(args.input))
    df = pd.DataFrame.from_dict(obj)
    subset = create_df_subset(
        df, [args.group_by, args.x_axis, args.trial] + METRICS)
    csv_path = os.path.join(args.output, "summary.csv")
    logging.info("Writing brief summary to %s", csv_path)
    subset.to_csv(csv_path)
    create_plots(args, subset, METRICS)


def add_simul_args(p_simul):
    p_simul.add_argument(
        '--seed', type=int, default=None,
        help='Random number generator seed for reproducibility')
    p_simul.add_argument(
        '--sim_size', type=int, default=1000,
        help='Simulation size (when number of clusters is not given)')
    p_simul.add_argument(
        '--cluster_size', type=int, default=None,
        help='cluster size (overrides cluster mean and sigma)')
    p_simul.add_argument(
        '--c_size_mean', type=float, default=4,
        help='Mean of cluster size')
    p_simul.add_argument(
        '--c_size_sigma', type=float, default=10,
        help='Std. dev. of cluster size')
    p_simul.add_argument(
        '--pos_ratio', type=float, default=0.1,
        help='ratio of positives to all')
    p_simul.add_argument(
        '--p_err', type=float, default=0.05,
        help='Probability of error at any location in sequence')
    p_simul.add_argument(
        '--doc_len_min', type=int, default=3,
        help='Minimum sequence length')
    p_simul.add_argument(
        '--doc_len_mean', type=float, default=8,
        help='Mean of sequence length')
    p_simul.add_argument(
        '--doc_len_sigma', type=float, default=10,
        help='Std. dev. of sequence length')


def add_parameterization_args(parser):
    parser.add_argument(
        '--group_by', type=str, default='hashfun',
        help='Field to group by')
    parser.add_argument(
        '--x_axis', type=str, default='cluster_size',
        help='Which column to plot as X axis')
    parser.add_argument(
        '--trial', type=str, default='seed',
        help='Which column to average')


def add_clust_args(p_clust):
    p_clust.add_argument(
        '--hashfun', type=str, default='builtin',
        choices=HASH_FUNC_TABLE.keys(),
        help='Hash function to use')
    p_clust.add_argument(
        '--shingle_span', type=int, default=4,
        help='shingle length (in tokens)')
    p_clust.add_argument(
        '--shingle_skip', type=int, default=0,
        help='words to skip')
    p_clust.add_argument(
        '--shingle_uniq', type=int, default=1,
        help='whether to unique shingles')
    p_clust.add_argument(
        '--shingle_kmin', type=int, default=0,
        help='minimum expected shingles')
    p_clust.add_argument(
        '--width', type=int, default=3,
        help='length of minhash feature vectors')
    p_clust.add_argument(
        '--bandwidth', type=int, default=3,
        help='rows per band')
    p_clust.add_argument(
        '--kmin', type=int, default=3,
        help='number of minhashes to sample')
    p_clust.add_argument(
        '--lsh_scheme', type=str, default="a0",
        help='LSH binning scheme')


def add_analy_args(p_clust):
    add_parameterization_args(p_clust)
    p_clust.add_argument(
        '--metrics', type=str, nargs='*', choices=METRICS,
        default=('roc_auc', 'nmi_score', 'time_cpu'),
        help='Which metrics to calculate')


def parse_args(args=None):
    parser = PathArgumentParser(
        description="Simulate data and/or run analysis")

    parser.add_argument(
        '--logging', type=str, default='WARN', help="Logging level",
        choices=[key for key in logging._levelNames.keys() if isinstance(key, str)])

    subparsers = parser.add_subparsers()

    p_simul = subparsers.add_parser('simulate', help='generate simulation')
    add_simul_args(p_simul)
    p_simul.add_argument(
        '--output', type=GzipFileType('w'), default=sys.stdout, help='File output')
    p_simul.set_defaults(func=do_simulation)

    p_clust = subparsers.add_parser('cluster', help='run clustering')
    p_clust.add_argument(
        '--input', type=GzipFileType('r'), default=sys.stdin, help='File input')
    add_clust_args(p_clust)
    p_clust.add_argument(
        '--output', type=GzipFileType('w'), default=sys.stdout, help='File output')
    p_clust.set_defaults(func=do_cluster)

    p_analy = subparsers.add_parser('analyze', help='run analysis')
    p_analy.add_argument(
        '--input', type=GzipFileType('r'), default=sys.stdin, help='File input')
    add_analy_args(p_analy)
    p_analy.add_argument(
        '--output', type=GzipFileType('w'), default=sys.stdout, help='File output')
    p_analy.set_defaults(func=do_analyze)

    p_simul_clust_analy = subparsers.add_parser(
        'simul_clust_analy', help='Perform multiple steps')
    add_simul_args(p_simul_clust_analy)
    add_clust_args(p_simul_clust_analy)
    add_analy_args(p_simul_clust_analy)
    p_simul_clust_analy.add_argument(
        '--output', type=GzipFileType('w'), default=sys.stdout, help='File output')
    p_simul_clust_analy.set_defaults(func=do_simul_clust_analy)

    p_summa = subparsers.add_parser('summarize', help='summarize analysis results')
    add_parameterization_args(p_summa)
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
