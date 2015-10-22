from math import log as logn
from collections import defaultdict, Counter, Mapping
from itertools import izip
from scipy.special import binom


def cond_entropy(counts, N):
    """Returns conditional entropy

    The parameter `counts` is expected to be an list or tuple-like iterable.
    For convenience, it can also be a dict/mapping type, in which case
    its values will be used to calculate entropy

    TODO: Cythonize this using NumPy's buffer interface for arrays
    """
    if isinstance(counts, Mapping):
        counts = counts.values()
    log_row_total = logn(sum(counts))
    # to avoid loss of precision, calculate 'log(a/b)' as 'log(a) - loh(b)'
    return -sum(c * (logn(c) - log_row_total) for c in counts if c != 0) / N


def harmonic_mean(x, y):
    """Harmonic mean of two numbers. Returns a float
    """
    # Since harmonic mean converges to arithmetic mean as x approaches y,
    # return the latter when x == y, which is numerically safer.
    return (x + y) / 2.0 if x == y else (2.0 * x * y) / (x + y)


class ContingencyTable(object):

    def __init__(self, rows=None, cols=None, row_totals=None, col_totals=None, grand_total=0):
        self.rows = rows or defaultdict(Counter)
        self.cols = cols or defaultdict(Counter)
        self.row_totals = row_totals or Counter()
        self.col_totals = col_totals or Counter()
        self.grand_total = grand_total

    def iter_cells(self):
        return (cell for row in self.rows.itervalues() for cell in row.itervalues())

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

    def clustering_metrics(self):
        N = self.grand_total
        H_C = cond_entropy(self.row_totals, N)
        H_K = cond_entropy(self.col_totals, N)
        H_CK = sum(cond_entropy(col, N) for col in self.iter_cols())
        H_KC = sum(cond_entropy(row, N) for row in self.iter_rows())
        # The '<=' comparisons below both prevent division by zero errors
        # and ensure that the scores are non-negative.
        homogeneity = 0.0 if H_C <= H_CK else 1.0 - H_CK / H_C
        completeness = 0.0 if H_K <= H_KC else 1.0 - H_KC / H_K
        nmi_score = harmonic_mean(homogeneity, completeness)
        return homogeneity, completeness, nmi_score

    def adjusted_rand_index(self):
        sum_n = sum(binom(cell, 2) for cell in self.iter_cells())
        sum_a = sum(binom(a, 2) for a in self.iter_col_totals())
        sum_b = sum(binom(b, 2) for b in self.iter_row_totals())
        n_choose_2 = float(binom(self.grand_total, 2))
        sum_a_sum_b__n_choose_2 = (sum_a / n_choose_2) * sum_b
        numerator = sum_n - sum_a_sum_b__n_choose_2
        denominator = 0.5 * (sum_a + sum_b) - sum_a_sum_b__n_choose_2
        return numerator / denominator


def homogeneity_completeness_v_measure(labels_true, labels_pred):
    """Calculate three common clustering metrics at once

    The metrics are: Homogeneity, Completeness, and V-measure

    The V-measure metric is also known as Normalized Mutual Informmation,
    and is the harmonic mean of Homogeneity and Completeness. The latter
    two metrics are symmetric (one is a complement of another).

    This code is replaces an equivalent function in Scikit-Learn known as
    `homogeneity_completeness_v_measure`, which alas takes up O(n^2)
    space because it creates a dense contingency matrix during calculation.
    Here we use sparse dict-based methods to achieve the same result while
    using much less RAM.
    """
    ct = ClusteringMetrics.from_labels(labels_true, labels_pred)
    return ct.clustering_metrics()


def adjusted_rand_score(labels_true, labels_pred):
    """Calculate Adjusted Rand Index in a memory-efficient way
    """
    ct = ClusteringMetrics.from_labels(labels_true, labels_pred)
    return ct.adjusted_rand_index()
