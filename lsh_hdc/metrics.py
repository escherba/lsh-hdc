from math import log as logn
from collections import defaultdict, Counter, Mapping, Set
from itertools import izip
from scipy.special import binom
from sklearn.metrics.ranking import roc_curve, auc


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
        return float('nan')
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
    return float(x) if x == y else (2.0 * x * y) / (x + y)


class ContingencyTable(object):

    def __init__(self, rows=None, cols=None, row_totals=None, col_totals=None,
                 grand_total=0):
        self.rows = rows or defaultdict(Counter)
        self.cols = cols or defaultdict(Counter)
        self.row_totals = row_totals or Counter()
        self.col_totals = col_totals or Counter()
        self.grand_total = grand_total

    def iter_cells(self):
        return (cell for row in self.rows.itervalues()
                for cell in row.itervalues())

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
        N = 0
        for c, k in izip(labels_true, labels_pred):
            classes[c][k] += 1
            klusters[k][c] += 1
            class_total[c] += 1
            kluster_total[k] += 1
            N += 1
        return cls(classes, klusters, class_total, kluster_total, N)


class ClusteringMetrics(ContingencyTable):

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

        Adjusted Rand index is Rand index adjusted for chance, which makes
        the resulting measure independent of cluster size.
        """
        N = self.grand_total
        if N <= 1:
            return float('nan')
        sum_n = sum(binom(cell, 2) for cell in self.iter_cells())
        sum_a = sum(binom(a, 2) for a in self.iter_col_totals())
        sum_b = sum(binom(b, 2) for b in self.iter_row_totals())
        n_choose_2 = float(binom(N, 2))
        sum_a_sum_b__n_choose_2 = (sum_a / n_choose_2) * sum_b
        numerator = sum_n - sum_a_sum_b__n_choose_2
        denominator = 0.5 * (sum_a + sum_b) - sum_a_sum_b__n_choose_2
        return numerator / denominator


def homogeneity_completeness_v_measure(labels_true, labels_pred):
    """Memory-efficient replacement for equivalently named Sklearn function
    """
    ct = ClusteringMetrics.from_labels(labels_true, labels_pred)
    return ct.entropy_metrics()


def adjusted_rand_score(labels_true, labels_pred):
    """Memory-efficient replacement for equivalently named Sklearn function
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
            y_true, y_score, pos_label=1, sample_weight=sample_weight)
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
        max_index = float('-inf')
        opt_pair = (float('nan'), float('nan'))
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
