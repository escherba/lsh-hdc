import numpy as np
from math import log as logn
from collections import defaultdict, Counter, Mapping, Set
from itertools import izip
from operator import itemgetter
from scipy.special import binom
from sklearn.metrics.ranking import roc_curve, auc
from pymaptools.iter import aggregate_tuples


def jaccard_similarity(set1, set2):
    """Return Jaccard similarity between two sets

    :param set1: set 1
    :param set2: set 2
    :returns: Jaccard similarity of two sets
    :rtype: float
    """
    if not isinstance(set1, Set):
        set1 = set(set1)
    if not isinstance(set2, Set):
        set2 = set(set2)
    denominator = len(set1 | set2)
    if denominator == 0:
        return np.nan
    else:
        return len(set1 & set2) / float(denominator)


def entropy_of_counts(counts):
    """Returns entropy of a list of counts

    Assumes every entry in the list belongs to a different class.

    The parameter `counts` is expected to be an list or tuple-like iterable.
    For convenience, it can also be a dict/mapping type, in which case
    its values will be used to calculate entropy.

    The entropy is calculated using natural base, which may not be what
    you want, so caveat emptor.

    TODO: Cythonize this using NumPy's buffer interface for arrays
    """
    if isinstance(counts, Mapping):
        counts = counts.values()
    sum_counts = sum(counts)
    if sum_counts == 0:
        return 0.0
    # to avoid loss of precision, calculate 'log(a/b)' as 'log(a) - log(b)'
    log_row_total = logn(sum_counts)
    return sum(c * (log_row_total - logn(c)) for c in counts if c != 0)


def harmonic_mean(x, y):
    """Harmonic mean of two numbers. Returns a float
    """
    # the condition below is only for numeric safety when x and y are small
    return float(x) if x == y else 2.0 * (x * y) / (x + y)


def harmonic_mean_weighted(x, y, beta=1.0):
    """Harmonic mean of two numbers. Returns a float
    """
    # the condition below is only for numeric safety when x and y are small
    beta **= 2.0
    return float(x) if x == y else (1.0 + beta) * (x * y) / (beta * x + y)


class ContingencyTable(object):

    # TODO: use one of Scipy's sparse matrix representations instead of
    # a dict of dicts

    def __init__(self, rows=None, cols=None, row_totals=None, col_totals=None,
                 grand_total=0):
        self.rows = rows or defaultdict(Counter)
        self.cols = cols or defaultdict(Counter)
        self.row_totals = row_totals or Counter()
        self.col_totals = col_totals or Counter()
        self.grand_total = grand_total

    def iter_cells(self):
        for row in self.rows.itervalues():
            for cell in row.itervalues():
                yield cell

    def iter_cols(self):
        return self.cols.itervalues()

    def iter_rows(self):
        return self.rows.itervalues()

    def iter_col_totals(self):
        return self.col_totals.itervalues()

    def iter_row_totals(self):
        return self.row_totals.itervalues()

    @classmethod
    def from_labels(cls, labels_true, labels_pred):
        classes = defaultdict(Counter)
        klusters = defaultdict(Counter)
        class_total = Counter()
        kluster_total = Counter()
        grand_total = 0
        for c, k in izip(labels_true, labels_pred):
            classes[c][k] += 1
            klusters[k][c] += 1
            class_total[c] += 1
            kluster_total[k] += 1
            grand_total += 1
        return cls(rows=classes, cols=klusters, row_totals=class_total,
                   col_totals=kluster_total, grand_total=grand_total)


class ConfMatBinary(object):
    """A binary confusion matrix
    """

    def __init__(self, TP, FP, TN, FN):
        self.TP = TP
        self.FP = FP
        self.TN = TN
        self.FN = FN
        self.N = TP + FP + TN + FN

    def kappa(self):
        """Calculate Cohen's kappa of a binary confusion matrix
        """
        n_choose_2 = self.N        # (grand_total choose 2)
        if n_choose_2 == 0:
            return np.nan
        sum_n = self.TP            # Sum{(n choose 2) over all cells}
        sum_a = self.TP + self.FP  # Sum{(n choose 2) over column totals}
        sum_b = self.TP + self.FN  # Sum{(n choose 2) over row totals}

        sum_a_sum_b__n_choose_2 = (sum_a * sum_b) / n_choose_2
        numerator = sum_n - sum_a_sum_b__n_choose_2
        denominator = 0.5 * (sum_a + sum_b) - sum_a_sum_b__n_choose_2
        return numerator / denominator

    def accuracy(self):
        """Accuracy (also known as Rand Index)

        This is generally the wrong metric to use. You probably want either
        precision, recall, F-score, or a chance-corrected version of accuracy
        known as Cohen's kappa (see kappa() method).
        """
        denominator = self.N
        return np.nan if denominator == 0 else float(self.TP + self.TN) / denominator

    def precision(self):
        """Calculate precision from the pairwise confusion matrix
        """
        denominator = self.TP + self.FP
        return np.nan if denominator == 0 else float(self.TP) / denominator

    def recall(self):
        """Calculate recall from the pairwise confusion matrix
        """
        denominator = self.TP + self.FN
        return np.nan if denominator == 0 else float(self.TP) / denominator

    def fscore(self, beta=1.0):
        """Calculate F-score from the pairwise confusion metric

        As beta tends to infinity, F-score will approach recall
        As beta tends to zero, F-score will approach precision
        """
        return harmonic_mean_weighted(self.precision(), self.recall(), beta)

    def jaccard_coeff(self):
        """Calculate Jaccard coefficient of clustering performance

        This metric is similar to accuracy except it ignores true negatives
        (of which there can be very many)
        """
        denominator = self.TP + self.FP + self.FN
        return np.nan if denominator == 0 else float(self.TP) / denominator


class ClusteringMetrics(ContingencyTable):

    """Provides external clustering evaluation metrics

    A subclass of ContingencyTable that provides four external clustering
    evaluation metrics: homogeneity, completeness, V-measure, and adjusted
    Rand index.

    The motivation behind this implementation was to avoid the high memory
    usage of equivalent methods in Scikit-Learn. The Scikit-Learn
    implementations create an incidence matrix for computation
    of these metrics resulting in O(n^2) memory usage, something that is
    felt particularly on large data sets and in multiprocessing environment.
    The given implementation uses sparse methods on dictionary maps instead
    of building incidence matrices.
    """

    def __init__(self, *args, **kwargs):
        super(ClusteringMetrics, self).__init__(*args, **kwargs)
        self.confusion_matrix_ = self.confusion_matrix()

    def precision(self):
        return self.confusion_matrix_.precision()

    def recall(self):
        return self.confusion_matrix_.recall()

    def fscore(self, beta=1.0):
        return self.confusion_matrix_.fscore(beta)

    def rand_index(self):
        return self.confusion_matrix_.accuracy()

    def jaccard_coeff(self):
        return self.confusion_matrix_.jaccard_coeff()

    def confusion_matrix(self):
        """Calculate a binary confusion matrix from object pair distribution

        Order of objects returned: TP, FP, TN, FN
        """
        TP_plus_FP = sum(binom(a, 2) for a in self.iter_col_totals())
        TP_plus_FN = sum(binom(b, 2) for b in self.iter_row_totals())
        TP = sum(binom(cell, 2) for cell in self.iter_cells())
        FP = TP_plus_FP - TP
        FN = TP_plus_FN - TP
        TN = binom(self.grand_total, 2) - TP - FP - FN
        return ConfMatBinary(TP, FP, TN, FN)

    def entropy_metrics(self):
        """Calculate three entropy metrics used for clustering evaluation

        The metrics are: Homogeneity, Completeness, and V-measure

        The V-measure metric is also known as Normalized Mutual Informmation,
        and is the harmonic mean of Homogeneity and Completeness. The latter
        two metrics are complementary of each other (dual).

        This code is replaces an equivalent function in Scikit-Learn known as
        `homogeneity_completeness_v_measure`, which alas takes up O(n^2)
        space because it creates a dense contingency matrix during calculation.
        Here we use sparse dict-based methods to achieve the same result while
        using much less RAM.

        The entropy variables used in the code here are improperly defined
        because they ought to be divided by N (the grand total for the
        contigency table). However, numerically it is more efficient not to
        perform the division.
        """
        H_C = entropy_of_counts(self.row_totals)
        H_K = entropy_of_counts(self.col_totals)
        H_CK = sum(entropy_of_counts(col) for col in self.iter_cols())
        H_KC = sum(entropy_of_counts(row) for row in self.iter_rows())
        # The '<=' comparisons below both prevent division by zero errors
        # and ensure that the scores are non-negative.
        homogeneity = 0.0 if H_C <= H_CK else (H_C - H_CK) / H_C
        completeness = 0.0 if H_K <= H_KC else (H_K - H_KC) / H_K
        nmi_score = harmonic_mean(homogeneity, completeness)
        return homogeneity, completeness, nmi_score

    def adjusted_rand_index(self):
        """Calculate Adjusted Rand Index in a memory-efficient way

        Adjusted Rand Index measures overall agreement between two clusterings.
        It is Rand index adjusted for chance, and has the property that
        the resulting metric is independent of cluster size.

        The index has an equivalence to Cohen's Kappa described in [1].
        Milligan [2] suggests that the two measures (ARI and Kappa) be
        used in replication studies when relevant.

        References
        ----------

        [1] Warrens, M. J. On the equivalence of Cohen's Kappa and the
        Hubert-Arabie Adjusted Rand Index. 2008. J. Classif. 25: 177-183.
        https://doi.org/10.1007/s00357-008-9023-7

        [2] Milligan, G.W. Clustering validation: results and
        implications for applied analysis. In Arabie P., de Soete, G. (ed)
        Clustering and Classification, 1996: 358-369.
        https://doi.org/10.1142/9789812832153_0010
        """
        return self.confusion_matrix_.kappa()


def homogeneity_completeness_v_measure(labels_true, labels_pred):
    """Memory-efficient replacement for equivalently named Sklearn function
    """
    ct = ClusteringMetrics.from_labels(labels_true, labels_pred)
    return ct.entropy_metrics()


def adjusted_rand_score(labels_true, labels_pred):
    """Memory-efficient replacement for equivalently named Sklearn function

    Example (given in supplement to "An empirical study on Principal
    Component Analysis for clustering gene expression data" by K.Y. Yeung,
    W. L. Ruzzo (2001)

    >>> classes = [1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
    >>> clusters = [1, 2, 1, 2, 2, 3, 3, 3, 3, 3]
    >>> round(adjusted_rand_score(classes, clusters), 3)
    0.313
    """
    ct = ClusteringMetrics.from_labels(labels_true, labels_pred)
    return ct.adjusted_rand_index()


class RocCurve(object):

    def __init__(self, fprs, tprs, thresholds=None, pos_label=None,
                 sample_weight=None):
        self.fprs = fprs
        self.tprs = tprs
        self.thresholds = thresholds
        self.pos_label = pos_label
        self.sample_weight = sample_weight

    @classmethod
    def from_binary(cls, y_true, y_score, sample_weight=None):
        """Convenience constructor which assumes 1 to be the positive label
        """
        fprs, tprs, thresholds = roc_curve(
            y_true, y_score, pos_label=True, sample_weight=sample_weight)
        return cls(fprs, tprs, thresholds=thresholds, sample_weight=sample_weight)

    def auc_score(self):
        """Override Sklearn's method so as not to raise error

        If number of Y classes is other than two, a warning will be triggered
        but no exception thrown (the return value will be a NaN). This differes
        from the import behavior of Scikit-Learn's roc_auc_score method
        (it always raises an exception) taht I find annoying!

        Also, we don't reorder arrays during ROC calculation since they
        are assumed to be in order.

        Example:

        >>> rc = RocCurve.from_binary([0, 0, 1, 1],
        ...                           [0.1, 0.4, 0.35, 0.8])
        >>> rc.auc_score()
        0.75
        >>> rc.max_deltap()
        0.5
        """
        return auc(self.fprs, self.tprs, reorder=False)

    def optimal_cutoff(self, method):
        """Calculate optimal cutoff point according to a method lambda
        """
        max_index = np.NINF
        opt_pair = (np.nan, np.nan)
        for pair in izip(self.fprs, self.tprs):
            index = method(*pair)
            if index > max_index:
                opt_pair = pair
                max_index = index
        return opt_pair, max_index

    @staticmethod
    def _informedness(fpr, tpr):
        return tpr - fpr

    def max_deltap(self):
        """Calculates Maximum DeltaP value Informedness (Youden's Index)
        https://en.wikipedia.org/wiki/Youden%27s_J_statistic
        """
        return self.optimal_cutoff(self._informedness)[1]


def roc_auc_score(y_true, y_score, sample_weight=None):
    """Replaces Scikit Learn implementation (for binary y_true vectors only)
    >>> roc_auc_score([0, 0, 1, 1],
    ...               [0.1, 0.4, 0.35, 0.8])
    0.75
    """
    return RocCurve.from_binary(y_true, y_score).auc_score()


def _plot_lift(xs, ys):  # pragma: no cover
    """Shortcut to plot a lift chart (for clustering_aul_score debugging)
    """
    from matplotlib import pyplot
    pyplot.plot(xs, ys, marker="o", linestyle='-')
    pyplot.xlim(xmin=0.0, xmax=1.0)
    pyplot.ylim(ymin=0.0, ymax=1.0)
    pyplot.show()


def clustering_aul_score(clusters, is_pos):
    """Area under lift curve for clusters of a binary class distribution

    The AUL score calculated is very similar to the Gini index of inequality
    (area between equality and the Lorenz curve) except we do not subtract
    0.5. Note that it can be lower than 0.5 because, in a very bad clustering,
    small clusters of size 1 will be sorted by negative of the number of
    positives.

    Useful when primary criterion of clustering quality is bigger clusters
    containing more data in the positive class, while unassigned data (or
    clusters of size one) ought to belong to the negative class
    """
    def count_pos(cluster):
        # count negatives
        return sum(is_pos(point) for point in cluster)

    def make_sortable(cluster):
        return len(cluster), count_pos(cluster)

    sortable = [make_sortable(cluster) for cluster in clusters]
    # sort just by cluster size
    data = sorted(sortable, key=itemgetter(0), reverse=True)
    data = list(aggregate_tuples(data))

    # in first pass, calculate some totals and cumulatives
    total_pos = 0
    max_horizontal = 0
    max_vertical = 0

    for cluster_size, pos_counts in data:
        num_clusters = len(pos_counts)
        total_pos += sum(pos_counts)
        total_in_group = cluster_size * num_clusters
        max_horizontal += total_in_group

        if cluster_size > 1:
            max_vertical += total_in_group
        else:
            max_vertical += sum(pos_counts)

    assert max_horizontal >= total_pos

    if max_vertical == 0:
        return np.nan

    # in the second pass, calculate the AUL metric
    aul_score = 0.0
    bin_height = 0.0
    bin_right_edge = 0

    # xs = []
    # ys = []

    # for each group of clusters of the same size...
    for cluster_size, pos_counts in data:
        avg_pos_count = sum(pos_counts) / float(len(pos_counts))

        for _ in pos_counts:

            # xs.append(bin_right_edge / float(max_horizontal))

            bin_width = cluster_size
            bin_height += avg_pos_count
            bin_right_edge += bin_width
            aul_score += bin_height * bin_width

            # ys.append(bin_height / float(max_vertical))
            # xs.append(bin_right_edge / float(max_horizontal))
            # ys.append(bin_height / float(max_vertical))

    assert max_horizontal == bin_right_edge
    aul_score /= (max_vertical * max_horizontal)
    return aul_score
