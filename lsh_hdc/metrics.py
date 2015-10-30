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

Secondly, it was interesting to investigate the type of co-association matrices
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

[7] Sim, J., & Wright, C. C. (2005). The kappa statistic in reliability studies:
use, interpretation, and sample size requirements. Physical therapy, 85(3),
257-268.
http://www.ncbi.nlm.nih.gov/pubmed/15733050
"""

import numpy as np
from math import log as logn, sqrt, copysign
from collections import Mapping, Set, namedtuple
from itertools import izip
from operator import itemgetter
from sklearn.metrics.ranking import roc_curve, auc
from pymaptools.iter import aggregate_tuples
from pymaptools.containers import TableOfCounts


def nchoose2(n):
    """Binomial coefficient for k=2

    Scipy has ``scipy.special.binom`` and ``scipy.misc.comb``, however on
    individaul (non-vectorized) ops used in memory-constratined stream
    computation, a simple definition below is faster. It is possible to get the
    best of both worlds by writing a generator that returns NumPy arrays of
    limited size and then calling a vectorized n-choose-2 function on those,
    however the current way is fast enough for computing coincidence matrices
    (turns out memory was the bottleneck, not raw computation speed).
    """
    return (n * (n - 1)) >> 1


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


def ratio2weights(ratio):
    """Numerically accurate conversion of ratio of two weights to weights
    """
    if ratio <= 1.0:
        lweight = ratio / (1.0 + ratio)
    else:
        lweight = 1.0 / (1.0 + 1.0 / ratio)
    return lweight, 1.0 - lweight


def geometric_mean(x, y):
    """Geometric mean of two numbers. Returns a float

    Although geometric mean is defined for negative numbers, Scipy function
    doesn't allow it... sigh
    """
    prod = x * y
    if prod < 0.0:
        raise ValueError("x and y have different signs")
    return copysign(1, x) * sqrt(prod)


def geometric_mean_weighted(x, y, ratio=1.0):
    """Geometric mean of two numbers with a weight ratio. Returns a float

    >>> geometric_mean_weighted(1, 4, ratio=1.0)
    2.0
    >>> geometric_mean_weighted(1, 4, ratio=0.0)
    1.0
    >>> geometric_mean_weighted(1, 4, ratio=float('inf'))
    4.0
    """
    lweight, rweight = ratio2weights(ratio)
    lsign = copysign(1, x)
    rsign = copysign(1, y)
    if lsign != rsign and x != y:
        raise ValueError("x and y have different signs")
    return lsign * (abs(x) ** rweight) * (abs(y) ** lweight)


def harmonic_mean(x, y):
    """Harmonic mean of two numbers. Returns a float
    """
    return float(x) if x == y else 2.0 * (x * y) / (x + y)


def harmonic_mean_weighted(x, y, ratio=1.0):
    """Harmonic mean of two numbers with a weight ratio. Returns a float

    >>> harmonic_mean_weighted(1, 3, ratio=1.0)
    1.5
    >>> harmonic_mean_weighted(1, 3, ratio=0.0)
    1.0
    >>> harmonic_mean_weighted(1, 3, ratio=float('inf'))
    3.0
    """
    lweight, rweight = ratio2weights(ratio)
    return float(x) if x == y else (x * y) / (lweight * x + rweight * y)


class ContingencyTable(TableOfCounts):

    # TODO: subclass pandas.DataFrame instead

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

    def _entropies(self):
        """Return H_C, H_K, and mutual information

        Not normalized by N
        """
        H_C = centropy(self.row_totals)
        H_K = centropy(self.col_totals)
        H_actual = centropy(self.iter_cells())
        H_expected = H_C + H_K
        I_CK = H_expected - H_actual
        return H_C, H_K, I_CK

    def vi_distance(self):
        """Variation of Information distance
        """
        H_C, H_K, I_CK = self._entropies()
        VI_CK = (H_C - I_CK) + (H_K - I_CK)
        return VI_CK / self.grand_total

    def split_join_distance(self):
        """Projection distance between partitions

        Used in graph commmunity analysis. Originally defined by van Dogen.
        Example given in [0]:

        >>> p1 = [{1, 2, 3, 4}, {5, 6, 7}, {8, 9, 10, 11, 12}]
        >>> p2 = [{2, 4, 6, 8, 10}, {3, 9, 12}, {1, 5, 7}, {11}]
        >>> cm = ClusteringMetrics.from_partitions(p1, p2)
        >>> cm.split_join_distance()
        11

        References
        ----------

        [0] Dongen, S. V. (2000). Performance criteria for graph clustering and
        Markov cluster experiments. Information Systems [INS], (R 0012), 1-36.

        """
        pa_B = sum(max(x) for x in self.iter_rows())
        pb_A = sum(max(x) for x in self.iter_cols())
        return 2 * self.grand_total - pa_B - pb_A

    def g_score(self):
        """Returns G-statistic for RxC contingency table

        Note that this doesn't calculate any corrections to this statistic
        (e.g. Williams', Yates' corrections).

        The statistic is equivalent to the negative of Mutual Information
        times two. Mututal Information on a contingency table is defined as
        the differnce between the information in the table and the
        information in an independent table with the same marginals.
        """
        _, _, I_CK = self._entropies()
        return 2.0 * I_CK

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
        """Calculate three entropy-based metrics used for clustering evaluation

        The metrics are: Homogeneity, Completeness, and V-measure

        The V-measure metric is also known as Normalized Mutual Information
        (NMI), and is defined here as the harmonic mean of Homogeneity and
        Completeness.  Homogeneity and Completeness are duals of each other and
        can be thought of as squared regression coefficients of a given
        clustering vs true labels (homogeneity) and of the dual problem of true
        labels vs given clustering (completeness). Because of the dual property,
        in a symmetric matrix, all three scores are the same.

        This code is replaces an equivalent function in Scikit-Learn known as
        `homogeneity_completeness_v_measure` (the Scikit-Learn version takes up
        O(n^2) space because it stores data in a dense NumPy array) while the
        given version is subquadratic because of sparse underlying storage.

        Note that the entropy variables as used directly in the code below are
        improperly defined because they ought to be divided by N (the grand
        total for the contigency table). However, the N variable cancels out
        during normalization.
        """
        # ensure non-negative values by taking max of 0 and given value
        H_C, H_K, I_CK = self._entropies()
        h = 0.0 if H_C == 0.0 else max(0.0, I_CK / H_C)
        c = 0.0 if H_K == 0.0 else max(0.0, I_CK / H_K)
        rsquare = harmonic_mean(h, c)
        return h, c, rsquare


class ClusteringMetrics(ContingencyTable):

    """Provides external clustering evaluation metrics

    A subclass of ContingencyTable that builds a pairwise co-association matrix
    for clustering comparisons.
    """

    def __init__(self, *args, **kwargs):
        super(ClusteringMetrics, self).__init__(*args, **kwargs)
        self.coassoc_ = self.compute_coassoc()

    def compute_coassoc(self):
        """Calculate a pairwise co-association matrix from two partitionings
        """
        TP_plus_FP = sum(nchoose2(a) for a in self.iter_col_totals())
        TP_plus_FN = sum(nchoose2(b) for b in self.iter_row_totals())
        TP = sum(nchoose2(cell) for cell in self.iter_cells())
        FP = TP_plus_FP - TP
        FN = TP_plus_FN - TP
        TN = nchoose2(self.grand_total) - TP - FP - FN
        return ConfusionMatrix2.from_ccw(TP, FP, TN, FN)

    def adjusted_rand_index(self):
        """Calculate Adjusted Rand Index in a memory-efficient way

        Adjusted Rand Index measures overall agreement between two clusterings.
        It is Rand index adjusted for chance, and has the property that the
        resulting metric is independent of cluster size.
        """
        return self.coassoc_.kappa()


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
        return cls(rows=[(TP, FN), (FP, TN)])

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

        Also known as hit rate
        """
        return _div(self.TP, self.TP + self.FN)

    def FPR(self):
        """Fallout (False Positive Rate)

        Synonyms: fallout, false alarm rate
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
        return harmonic_mean_weighted(self.precision(), self.recall(), beta ** 2)

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

    def prevalence_index(self):
        """Prevalence

        From [7]:

            ...a prevalence effect exists when the proportion of agreements on
            the positive classification differs from that of the negative
            classification

        Example high-prevalence matrix:

            3   27
            28  132

        """
        return _div(abs(self.TP - self.TN), self.grand_total)

    def bias_index(self):
        """Bias

        From [7]:

            Bias is the extent to which the raters disagree on the proportion of
            positive (or negative) cases

        Example high-bias matrix:

            17  14
            78  81

        """
        return _div(abs(self.FN - self.FP), self.grand_total)

    def informedness(self):
        """Informedness (Recall corrected for chance)

        Alternative formulations:

            Informedness = Sensitivity + Specificity - 1.0
                         = TPR - FPR

        Synonyms: True Skill Score, Hannssen-Kuiper Score
        """
        p1, q1 = self.row_totals.values()
        return _div(self.covar(), p1 * q1)

    def markedness(self):
        """Markedness (Precision corrected for chance)

        Alternative formulation:

            Markedness = PPV + NPV - 1.0

        """
        p2, q2 = self.col_totals.values()
        return _div(self.covar(), p2 * q2)

    def loevinger_coeff(self):
        """Loevinger two-sided coefficient of homogeneity

        Given a clustering (numbers correspond to class labels, inner groups to
        clusters) with perfect homogeneity but imperfect completeness, Loevinger
        coefficient returns a perfect score on the corresponding pairwise
        co-association matrix:

        >>> clusters = [[0, 0], [0, 0, 0, 0], [1, 1, 1, 1]]
        ...
        >>> cm = ClusteringMetrics.from_clusters(clusters)
        >>> cm.coassoc_.loevinger_coeff()
        1.0

        At the same time, kappa and matthews coefficients are 0.63 and 0.68,
        respectively. Being symmetrically defined, Loevinger coefficient will
        also return a perfect score in the dual (opposite) situation:

        >>> clusters = [[0, 2, 2, 0, 0, 0], [1, 1, 1, 1]]
        >>> cm = ClusteringMetrics.from_clusters(clusters)
        >>> cm.coassoc_.loevinger_coeff()
        1.0

        """
        p1, q1 = self.row_totals.values()
        p2, q2 = self.col_totals.values()
        return _div(self.covar(), min(p1 * q2, p2 * q1))

    def kappa(self):
        """Cohen's kappa (interrater agreement index)

        Kappa index comes from psychology and was originally introduced to
        measure interrater agreement. It is also appropriate for evaluating
        replication. In clustering applications, it is known as the 'Adjusted
        Rand Index'. Kappa is derived by correcting Accuracy (Simple Matching
        Coefficient, Rand Index) for chance. Tbe general formula for chance
        correction of an association coefficient ``k`` is:

                   k - E(k)
            k' = ------------ ,
                 k_max - E(k)

        where ``k_max`` is the maximum value the score can achieve given the
        same table margins, and ``E(k)`` is the expected value of ``k`` under
        statistical independence given existing margins.

        Kappa can be decomposed into a harmonic mean of two components
        (regression coefficients for a problem and its dual):

            k0 = cov / (p2 * q1)
            k1 = cov / (p1 * q2)

        From these components, it turns out, one can also derive Matthews'
        correlation coefficient simply by calculating a geometric mean.  However
        k0 and k1 don't have a lower bound. For that reason, it is preferable to
        use informedness and markedness (these two have a clear range from -1.0
        to 1.0 and their geometric mean also results in a correllation
        coefficient, Matthews') if one is looking for regression-like
        coefficients. On real-world data, k0 and k1 have a similar behavior
        to informedness and markedness.
        """
        p1, q1 = self.row_totals.values()
        p2, q2 = self.col_totals.values()
        a, c, d, b = self.to_ccw()
        n = self.grand_total
        if a == n or b == n or c == n or d == n:
            # only one cell is non-zero
            return np.nan
        elif p1 == 0 or p2 == 0 or q1 == 0 or q2 == 0:
            # one row or column is zero, another non-zero
            return 0.0
        else:
            # no more than one cell is zero
            return _div(2 * self.covar(), p1 * q2 + p2 * q1)

    def mp_corr(self):
        """Maxwell & Pilliner's chance-corrected association index

        Another covariance-based association index.
        """
        p1, q1 = self.row_totals.values()
        p2, q2 = self.col_totals.values()
        a, c, d, b = self.to_ccw()
        n = self.grand_total
        if a == n or b == n or c == n or d == n:
            # only one cell is non-zero
            return np.nan
        elif p1 == 0 or p2 == 0 or q1 == 0 or q2 == 0:
            # one row or column is zero, another non-zero
            return 0.0
        else:
            # no more than one cell is zero
            return _div(2 * self.covar(), p1 * q1 + p2 * q2)

    def matthews_corr(self):
        """Matthews Correlation Coefficient (Phi coefficient)

        MCC is directly related to the Chi-square statitstic. Its value is equal
        to the the Chi-square value normalized by the maximum value Chi-Square
        can achieve with given margins (for a 2x2 table, the maximum Chi-square
        score is N) and transformed to correlation space by taking a square
        root. MCC is a also a geometric mean of informedness and markedness (the
        regression coefficients of the problem and its dual).

        MCC is also related to Cohen's Kappa (see description for kappa method)
        and together they are the two most commonly used chance-corrected
        association coefficients.

        MCC is laso known as Phi Coefficient or as Yule's Q with correction for
        chance.
        """
        p1, q1 = self.row_totals.values()
        p2, q2 = self.col_totals.values()
        a, c, d, b = self.to_ccw()
        n = self.grand_total
        if a == n or b == n or c == n or d == n:
            # only one cell is non-zero
            return np.nan
        elif p1 == 0 or p2 == 0 or q1 == 0 or q2 == 0:
            # one row or column is zero, another non-zero
            return 0.0
        else:
            # no more than one cell is zero
            return _div(self.covar(), sqrt(p1 * q1 * p2 * q2))

    def mutinf_signed(self):
        """Assigns a sign to mututal information-based metrics
        """
        info, mark, corr = self.mutinf_metrics()
        sgn = copysign(1, self.covar())
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
        p1, q1 = self.row_totals.values()
        p2, q2 = self.col_totals.values()
        a, c, d, b = self.to_ccw()
        n = self.grand_total
        if a == n or b == n or c == n or d == n:
            # only one cell is non-zero
            return np.nan
        elif p1 == 0 or p2 == 0 or q1 == 0 or q2 == 0:
            # one row or column is zero, another non-zero
            return 0.0
        else:
            # no more than one cell is zero
            return _div(self.covar(), a * d + b * c)

    def yule_y(self):
        """Colligation coefficient (Yule's Y)
        For a table of shape

        a b
        c d

        Yule's Y is (sqrt(ad) - sqrt(bc)) / (sqrt(ad) + sqrt(bc))
        """
        p1, q1 = self.row_totals.values()
        p2, q2 = self.col_totals.values()
        a, c, d, b = self.to_ccw()
        n = self.grand_total
        if a == n or b == n or c == n or d == n:
            # only one cell is non-zero
            return np.nan
        elif p1 == 0 or p2 == 0 or q1 == 0 or q2 == 0:
            # one row or column is zero, another non-zero
            return 0.0
        else:
            # no more than one cell is zero
            ad = a * d
            bc = b * c
            numer = sqrt(ad) - sqrt(bc)
            return _div(numer, sqrt(ad) + sqrt(bc))

    def covar(self):
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


def homogeneity_completeness_v_measure(labels_true, labels_pred):
    """Memory-efficient replacement for equivalently named Sklearn function
    """
    ct = ContingencyTable.from_labels(labels_true, labels_pred)
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
        from the behavior of Scikit-Learn's roc_auc_score method (it always
        raises an exception) that I find annoying!

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
    return ConfusionMatrix2.from_ccw(*args, **kwargs).matthews_corr()


def cohen_kappa(*args, **kwargs):
    """Return Cohen's Kappa for a 2x2 contingency table
    """
    return ConfusionMatrix2.from_ccw(*args, **kwargs).kappa()


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
