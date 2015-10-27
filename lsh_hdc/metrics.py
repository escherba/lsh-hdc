"""

Motivation
----------

The original motivation behind the given re-implementation of some clustering
metrics was to avoid the high memory usage of equivalent methods in
Scikit-Learn.  The Scikit-Learn implementations create an incidence matrix for
computation of these metrics resulting in O(n^2) memory usage, something that is
felt particularly strongly on large data sets and in multiprocessing environment
where each child process runnning on a multicore machine tries to allocate
memory for itself. This implementation uses sparse methods on dictionary maps
instead of building incidence matrices.

Secondly, it was interesting to investigate the type of confusion matrices
typically produced during pairwise cluster comparisons, and whether it has
any implications on the choice of overall quality measure.

Validation vs Replication Studies, ARI, and Kappa
-------------------------------------------------

ARI has an equivalence to Cohen's Kappa described in [1].  Milligan [2]
suggested that the two measures (ARI and Kappa) be used in replication studies
when relevant. Note that replication studies like the ones proposed in [2]
correspond to Model I test for independence as described in [3]. In a Model I
study, random sampling occurs both row- and column-wise, and only the grand
total is fixed.  In a Model II study, one side (either row or column totals) is
fixed.  Simulation-based clustering evaluations that fix the number and size of
ground truth classes belong to Model II category. Finally, in a Model III study,
both rows and column totals are fixed. Model III studies require careful
experimental design but have potentially more discriminative power. An example
of a Model III study would be a tea tasting party (to follow Fisher's classic
example) where both the true class totals (the portion and number of tea cups
with milk poured first) are known and the respondees are told ahead of time,
that, for example, 50% of cups they will encounter have had milk poured first.

To sum up the above, in a replication study where two clusterings are compared
that were both created by the same algorithm, the Kappa score seems to be an
appropriate metric (it is independent of row/column assignemnts).
The same applies to Matthew's Correlation Coefficient. However it
seems that a better chance-corrected metric should exist that would be
appropriate for Model II studies where the true class portions are fixed.

To be exhaustive, a Model O study would be one where even the grand total is not
fixed. An example would be an astronomy study that, for example, tests a
hypothesis about a generalizable property such as dark matter content by looking
at all galaxies in the Local Group, and the researchers obviously don't get to
choose ahead of time how many galaxies there are near ours.

References
----------

[0] Hasenclever, D., & Scholz, M. (2013). Comparing measures of association in
2x2 probability tables. arXiv preprint arXiv:1302.6161.
http://arxiv.org/pdf/1302.6161v1.pdf

[1] Warrens, M. J. (2008). On the equivalence of Cohen's kappa and the
Hubert-Arabie adjusted Rand index. Journal of Classification, 25(2), 177-183.
https://doi.org/10.1007/s00357-008-9023-7

[2] Arabie, P., Hubert, L. J., & De Soete, G. (1996). Clustering validation:
results and implications for applied analyses (p. 341). World Scientific Pub Co
Inc.
https://doi.org/10.1142/9789812832153_0010

[3] Sokal, R. R., & Rohlf, F. J. (2012). Biometry (4th edn). pp 742-744.

[4] Dunning, T. (1993). Accurate methods for the statistics of surprise and
coincidence. Computational linguistics, 19(1), 61-74.
http://dl.acm.org/citation.cfm?id=972454

[5] Ted Dunning's personal blog
http://tdunning.blogspot.com/2008/03/surprise-and-coincidence.html

[6] Warrens, M. J. (2008). On association coefficients for 2x2 tables and
properties that do not depend on the marginal distributions. Psychometrika,
73(4), 777-789.
https://doi.org/10.1007/s11336-008-9070-3
"""


import numpy as np
from math import log as logn, sqrt, copysign
from collections import defaultdict, Counter, Mapping, Set, namedtuple
from itertools import izip
from operator import itemgetter
from scipy.special import binom
from sklearn.metrics.ranking import roc_curve, auc
from pymaptools.iter import aggregate_tuples, iter_items, iter_vals


def _div(numer, denom):
    """Divide without raising zero division error or losing decimal part
    """
    if denom == 0:
        if numer == 0:
            return np.nan
        elif numer > 0:
            return np.PINF
        else:
            return np.NINF
    return float(numer) / denom


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
    return _div(len(set1 & set2), len(set1 | set2))


def centropy(counts):
    """Returns centropy of an iterable of counts

    Assumes every entry in the list belongs to a different class.

    The parameter `counts` is expected to be an list or tuple-like iterable.
    For convenience, it can also be a dict/mapping type, in which case its
    values will be used to calculate centropy.

    The centropy is calculated using natural base, which may not be what you
    want, so caveat emptor.

    TODO: Cythonize this using NumPy's buffer interface for arrays
    """
    if isinstance(counts, Mapping):
        counts = counts.itervalues()

    n = 0
    sum_c_logn_c = 0.0
    for c in counts:
        if c != 0:
            n += c
            sum_c_logn_c += c * logn(c)
    return 0.0 if n == 0 else n * logn(n) - sum_c_logn_c


def geometric_mean(x, y):
    """Geometric mean of two numbers

    Although geometric mean is defined for negative numbers, Scipy function
    doesn't allow it... sigh
    """
    return copysign(1, x) * sqrt(x * y)


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

    def iter_cells_with_indices(self):
        for ri, row in iter_items(self.rows):
            for ci, cell in iter_items(row):
                yield ri, ci, cell

    def iter_cells_with_margins(self):
        col_totals = self.col_totals
        row_totals = self.row_totals
        for ri, row in iter_items(self.rows):
            rm = row_totals[ri]
            for ci, cell in iter_items(row):
                cm = col_totals[ci]
                yield rm, cm, cell

    def iter_cells(self):
        for row in self.iter_rows():
            for cell in row:
                yield cell

    def iter_cols(self):
        for col in iter_vals(self.cols):
            yield iter_vals(col)

    def iter_rows(self):
        for row in iter_vals(self.rows):
            yield iter_vals(row)

    def iter_col_totals(self):
        return iter_vals(self.col_totals)

    def iter_row_totals(self):
        return iter_vals(self.row_totals)

    @classmethod
    def from_labels(cls, labels_true, labels_pred):
        rows = defaultdict(Counter)
        cols = defaultdict(Counter)
        row_totals = Counter()
        col_totals = Counter()
        grand_total = 0
        for c, k in izip(labels_true, labels_pred):
            rows[c][k] += 1
            cols[k][c] += 1
            row_totals[c] += 1
            col_totals[k] += 1
            grand_total += 1
        return cls(rows=rows, cols=cols, row_totals=row_totals,
                   col_totals=col_totals, grand_total=grand_total)

    def chisq_score(self):
        """Pearson's chi-square statistic
        """
        N = float(self.grand_total)
        score = 0.0
        for rm, cm, observed in self.iter_cells_with_margins():
            numer = rm * cm
            if numer != 0:
                expected = numer / N
                score += (observed - expected) ** 2 / expected
        return score

    def mutual_information(self):
        """Mutual information for expected vs actual contingency table
        """
        return (centropy(self.row_totals) +
                centropy(self.col_totals) -
                centropy(self.iter_cells()))

    def g_score(self):
        """Returns G-statistic for RxC contingency table

        Note that this doesn't calculate any corrections to this statistic
        (e.g. Williams', Yates' corrections).

        The statistic is equivalent to the negative of Mutual Information
        times two. Mututal Information on a contingency table is defined as
        the differnce between the information in the table and the
        information in an independent table with the same marginals.
        """
        return 2.0 * self.mutual_information()

    def mutinf_metrics(self):
        """Metrics based on mutual information

        The coefficient decomposes into regression coefficients defined
        according to fixed-margin tables. The `mi_info` coefficient, for
        example, is obtained by dividing the G-score by the maximum achievable
        value on a table with true class counts (in this case, row totals)
        fixed. the `mi_mark` is its dual, defined by dividing the G-score by
        its maximum achievable value when predicted label counts (in this case,
        column totals) are fixed.

        Note that G scores directly correspond to mutual information of
        a contingency table.
        """
        h, c, rsquare = self.entropy_metrics()
        return sqrt(h), sqrt(c), sqrt(rsquare)

    def entropy_metrics(self):
        """Calculate three centropy metrics used for clustering evaluation

        The metrics are: Homogeneity, Completeness, and V-measure

        The V-measure metric is also known as Normalized Mutual Informmation,
        and is the harmonic mean of Homogeneity and Completeness. The latter two
        metrics are complementary of each other (dual).

        This code is replaces an equivalent function in Scikit-Learn known as
        `homogeneity_completeness_v_measure`, which alas takes up O(n^2) space
        because it creates a dense contingency matrix during calculation.  Here
        we use sparse dict-based methods to achieve the same result while using
        much less RAM.

        The entropy variables used in the code here are improperly defined
        because they ought to be divided by N (the grand total for the
        contigency table). However, numerically it is more efficient not to
        perform the division.

        For a symmetric matrix, all three scores should be the same.
        """
        # ensure non-negative values by taking max of 0 and given value
        mut_info = max(0.0, self.mutual_information())
        max_h = max(0.0, centropy(self.row_totals))
        max_c = max(0.0, centropy(self.col_totals))
        h = 0.0 if max_h == 0.0 else mut_info / max_h
        c = 0.0 if max_c == 0.0 else mut_info / max_c
        rsquare = harmonic_mean(h, c)
        return h, c, rsquare


confmat2_type = namedtuple("Table2CCW", "TP FP TN FN")


class ConfusionMatrix2(ContingencyTable):
    """A confusion matrix (2x2 contingency table)

    For a binary variable (where one is measuring either presence vs absence of
    a particular feature), a confusion matrix where the ground truth levels are
    rows looks like:

        TP  FN
        FP  TN

    For a nominal variable, the negative class becomes a distinct label, and
    TP/FP/FN/TN terminology does not apply, although the algorithms should work
    the same way (with the obvious distinction that different assumptions will
    be made).
    """

    @classmethod
    def from_random_counts(cls, low=0, high=100):
        """Return a matrix instance initialized with random values
        """
        return cls.from_ccw(*np.random.randint(low=low, high=high, size=(4,)))

    @classmethod
    def from_ccw(cls, TP, FP, TN, FN):
        return cls(
            rows=((TP, FN), (FP, TN)),
            cols=((TP, FP), (FN, TN)),
            row_totals=(TP + FN, FP + TN),
            col_totals=(TP + FP, FN + TN),
            grand_total=(TP + FP + TN + FN)
        )

    def to_ccw(self):
        return confmat2_type(TP=self.TP, FP=self.FP, TN=self.TN, FN=self.FN)

    @property
    def TP(self):
        return self.rows[0][0]

    @property
    def FN(self):
        return self.rows[0][1]

    @property
    def FP(self):
        return self.rows[1][0]

    @property
    def TN(self):
        return self.rows[1][1]

    def ACC(self):
        """Accuracy (Simple Matching Coefficient, Rand Index)
        """
        return _div(self.TP + self.TN, self.grand_total)

    def PPV(self):
        """Precision (Positive Predictive Value)
        """
        return _div(self.TP, self.TP + self.FP)

    def NPV(self):
        """Negative predictive value
        """
        return _div(self.TN, self.TN + self.FN)

    def TPR(self):
        """Recall (Sensitivity)
        """
        return _div(self.TP, self.TP + self.FN)

    def FPR(self):
        """Fallout (False Positive Rate)
        """
        return _div(self.FP, self.TN + self.FP)

    def TNR(self):
        """Specificity (True Negative Rate)
        """
        return _div(self.TN, self.FP + self.TN)

    def FNR(self):
        """Miss Rate (False Negative Rate)
        """
        return _div(self.FN, self.TP + self.FN)

    def FDR(self):
        """False discovery rate
        """
        return _div(self.FP, self.TP + self.FP)

    def FOR(self):
        """False omission rate
        """
        return _div(self.FN, self.TN + self.FN)

    def PLL(self):
        """Positive likelihood ratio
        """
        return _div(self.TPR(), self.FPR())

    def NLL(self):
        """Negative likelihood ratio
        """
        return _div(self.FNR(), self.TNR())

    def DOR(self):
        """Diagnostics odds ratio

        Equal to PLL / NLL
        """
        return _div(self.TP * self.TN, self.FP * self.FN)

    def fscore(self, beta=1.0):
        """F-score

        As beta tends to infinity, F-score will approach recall.  As beta tends
        to zero, F-score will approach precision. For a similarity coefficient
        see dice_coeff.
        """
        return harmonic_mean_weighted(self.precision(), self.recall(), beta)

    def dice_coeff(self):
        """Dice similarity coefficient (Nei-Li coefficient)

        Thi is the same as F1-score but calculated slightly differently here.
        Note that Dice can be zero if total number of positives is zero,
        buit F-score is undefined in that case (beccause recall is undefined).

        Other metrics from the same family: jaccard_coeff, ochiai_coeff
        """
        a = self.TP
        return _div(2 * a, 2 * a + self.FN + self.FP)

    def jaccard_coeff(self):
        """Jaccard similarity coefficient

        Other metrics from the same family: dice_coeff, ochiai_coeff
        """
        return _div(self.TP, self.TP + self.FP + self.FN)

    def ochiai_coeff(self):
        """Ochiai (Cosine) similarity coefficient

        Other metrics from the same family: jaccard_coeff, dice_coeff
        """
        a, b, c = self.TP, self.FN, self.FP
        return _div(a, sqrt((a + b) * (a + c)))

    def prevalence(self):
        """Prevalence
        """
        return _div(self.TP, self.grand_total)

    def informedness(self):
        """Informedness = Sensitivity + Specificity - 1

        Informedness can be thought of as renormalization of recall after
        correcting for chance.
        """
        return self.sensitivity() + self.specificity() - 1.0

    def markedness(self):
        """Markedness = Precision + NPV - 1

        Informedness can be thought of as renormalization of precision after
        correcting for chance.
        """
        return self.precision() + self.NPV() - 1.0

    def loevinger_coeff(self):
        """Loevinger association coefficient
        """
        p1, q1 = self.row_totals
        p2, q2 = self.col_totals
        cov = self.covariance()
        if cov == 0 and self.grand_total != 0:
            return 0.0
        return _div(cov, min(p1 * q2, p2 * q1))

    def kappa(self):
        """Calculate Cohen's kappa of a binary confusion matrix

        Kappa index comes from psychology and was designed to measure interrater
        agreement. It is also a proper metric for measuring replication. It
        forms the basis of Adjusted Rand Index used for evaluation of
        clustering. However its applicability for evaluating experiments where
        ground truth vectors are known ahead of time is questionable.
        """
        p1, q1 = self.row_totals
        p2, q2 = self.col_totals
        cov = self.covariance()
        if cov == 0 and self.grand_total != 0:
            return 0.0
        return _div(2 * cov, p1 * q2 + p2 * q1)

    def mp_corr(self):
        """Maxwell & Pilliner's chance-corrected association index

        Thie formula is like that for Cohen's Kappa, but with a different
        denominator [6].
        """
        p1, q1 = self.row_totals
        p2, q2 = self.col_totals
        cov = self.covariance()
        if cov == 0 and self.grand_total != 0:
            return 0.0
        return _div(2 * cov, p1 * q1 + p2 * q2)

    def matthews_corr(self):
        """Matthews Correlation Coefficient (Phi coefficient)

        For a table of shape

        a b
        c d

        MCC is (ad - bc) / sqrt((a + b)(c + d)(a + c)(b + d))

        MCC is directly related to the Chi-square statitstic. Its value is equal
        to the the Chi-square value normalized by the maximum value Chi-Square
        can achieve with given margins (for a 2x2 table, the maximum Chi-square
        score is N) and transformed to correlation space by taking a square
        root. MCC is a also a geometric mean of informedness and markedness (the
        regression coefficients of the problem and its dual).

        Also known as Phi Coefficient and as Yule's Q with correction for
        chance.
        """
        p1, q1 = self.row_totals
        p2, q2 = self.col_totals
        cov = self.covariance()
        if cov == 0 and self.grand_total != 0:
            return 0.0
        return _div(cov, sqrt(p1 * q1 * p2 * q2))

    def mutinf_signed(self):
        """Assigns a sign to mututal information-based metrics
        """
        info, mark, corr = self.mutinf_metrics()
        sgn = copysign(1, self.covariance())
        return (sgn * info, sgn * mark, sgn * corr)

    def yule_q(self):
        """Yule's Q (index of association)
        For a table of shape

        a b
        c d

        Yule's Q is (ad - bc) / (ad + bc)

        It relates to odds ratio (here DOR() method):

                   OR - 1
           Q  =   --------.
                   OR + 1

        """
        cov = self.covariance()
        if cov == 0 and self.grand_total != 0:
            return 0.0
        return _div(cov, self.TP * self.TN + self.FP * self.FN)

    def yule_y(self):
        """Colligation coefficient (Yule's Y)
        For a table of shape

        a b
        c d

        Yule's Y is (sqrt(ad) - sqrt(bc)) / (sqrt(ad) + sqrt(bc))
        """
        ad = self.TP * self.TN
        bc = self.FN * self.FP
        numer = sqrt(ad) - sqrt(bc)
        if numer == 0 and self.grand_total != 0:
            return 0.0
        return _div(numer, sqrt(ad) + sqrt(bc))

    def covariance(self):
        """Determinant of a 2x2 matrix
        """
        return self.TP * self.TN - self.FP * self.FN

    # various silly terminologies folow

    # information retrieval
    precision = PPV
    recall = TPR
    fallout = FPR

    # clinical diagnostics
    sensitivity = TPR
    specificity = TNR

    # sales/marketing
    hit_rate = TPR
    miss_rate = FNR

    # ecology
    sm_coeff = ACC
    phi_coeff = matthews_corr

    # other
    accuracy = ACC
    rand_index = ACC


class ClusteringMetrics(ContingencyTable):

    """Provides external clustering evaluation metrics

    A subclass of ContingencyTable that provides four external clustering
    evaluation metrics: homogeneity, completeness, V-measure, and adjusted Rand
    index.
    """

    def __init__(self, *args, **kwargs):
        super(ClusteringMetrics, self).__init__(*args, **kwargs)
        self.confusion_matrix_ = self.pairwise_confusion_matrix()

    def pairwise_confusion_matrix(self):
        """Calculate a binary confusion matrix from object pair distribution

        Order of objects returned: TP, FP, TN, FN
        """
        TP_plus_FP = sum(binom(a, 2) for a in self.iter_col_totals())
        TP_plus_FN = sum(binom(b, 2) for b in self.iter_row_totals())
        TP = sum(binom(cell, 2) for cell in self.iter_cells())
        FP = TP_plus_FP - TP
        FN = TP_plus_FN - TP
        TN = binom(self.grand_total, 2) - TP - FP - FN
        return ConfusionMatrix2.from_ccw(TP, FP, TN, FN)

    def adjusted_rand_index(self):
        """Calculate Adjusted Rand Index in a memory-efficient way

        Adjusted Rand Index measures overall agreement between two clusterings.
        It is Rand index adjusted for chance, and has the property that the
        resulting metric is independent of cluster size.
        """
        return self.confusion_matrix_.kappa()


def homogeneity_completeness_v_measure(labels_true, labels_pred):
    """Memory-efficient replacement for equivalently named Sklearn function
    """
    ct = ClusteringMetrics.from_labels(labels_true, labels_pred)
    return ct.entropy_metrics()


def adjusted_rand_score(labels_true, labels_pred):
    """Memory-efficient replacement for equivalently named Sklearn function

    Example (given in supplement to "An empirical study on Principal Component
    Analysis for clustering gene expression data" by K.Y. Yeung, W. L. Ruzzo
    (2001)

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
        from the import behavior of Scikit-Learn's roc_auc_score method (it
        always raises an exception) taht I find annoying!

        Also, we don't reorder arrays during ROC calculation since they are
        assumed to be in order.

        Example:

        >>> rc = RocCurve.from_binary([0, 0, 1, 1],
        ...                           [0.1, 0.4, 0.35, 0.8])
        >>> rc.auc_score()
        0.75
        >>> rc.max_informedness()
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

    def max_informedness(self):
        """Calculates maximum value of Informedness (Youden's Index)
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


def matthews_corr(*args, **kwargs):
    """Return MCC score for a 2x2 contingency table
    """
    return ConfusionMatrix2(*args, **kwargs).matthews_corr()


def cohen_kappa(*args, **kwargs):
    """Return Cohen's Kappa for a 2x2 contingency table
    """
    return ConfusionMatrix2(*args, **kwargs).kappa()


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
    (area between equality and the Lorenz curve) except we do not subtract 0.5.
    Note that it can be lower than 0.5 because, in a very bad clustering, small
    clusters of size 1 will be sorted by negative of the number of positives.

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
