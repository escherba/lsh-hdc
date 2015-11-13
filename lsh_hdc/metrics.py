# -*- coding: utf-8 -*-

"""

Motivation
----------

The motivation behind the given re-implementation of some clustering metrics is
to avoid the high memory usage of equivalent methods in Scikit-Learn.  Using
sparse dictionary maps avoids storing co-incidence matrices in memory leading to
more acceptable performance in multiprocessing environment or on very large data
sets.

A side goal was to investigate different association metrics with the aim of
applying them to evaluation of clusterings in semi-supervised learning and
feature selection in supervised learning.

Finally, I was interested in the applicability of different association metrics
to different types of experimental design. At present, there seems to be both
(1) a lot of confusion about the appropriateness of different metrics, and (2)
relatively little attention paid to the type of experimental design used. I
believe that, at least partially, (1) stems from (2), and that different types
of experiments call for different categories of metrics.

Contingency Tables and Experimental Design
------------------------------------------

Consider studies that deal with two variables whose respective realizations can
be represented as rows and columns in a table.  Roughly adhering to the
terminology proposed in [1]_, we distinguish four types of experimental design
all involving contingency tables.

========= ===================================
Model O   all margins and totals are variable
Model I   only the grand total is fixed
Model II  one margin (either row or column totals) is fixed
Model III both margins are fixed
========= ===================================

Model O is rarely employed in practice because researchers almost always have
some rough total number of samples in mind that they would like to measure
before they begin the actual measuring. However, Model O situation might occur
when the grand total is not up to researchers to fix, and so they are forced to
treat it as a random variable. An example of this would be astronomy research
that tests a hypothesis about a generalizable property such as dark matter
content by looking at all galaxies in the Local Group, and the researchers
obviously don't get to choose ahead of time how many galaxies there are near
ours.

Model I and Model II studies are the most common and usually the most confusion
arises from mistaking one for the other. In psychology, interrater agreement is
an example of Model I approach. A replication study, if performed by the
original author, is a Model I study, but if performed by another group of
researchers, becomes a Model II study.

Fisher's classic example of tea tasting is an example of a Model III study [2]_.
The key difference from a Model II study here is that the subject was asked to
call four cups as prepared by one method and four by the other. The subject was
not free to say, for example, that none of the cups were prepared by adding milk
first. The hypergeometric distribution used in the subsequent Fisher's exact
test shares the assumption of the experiment that both row and column counts are
fixed.

Choosing an Association Metric
------------------------------

Given the types of experimental design listed above, some metrics seem to be
more appropriate than others. For example, two-way correlation coefficients
appear to be inappropriate for Model II studies where their respective regression
components seem more suited to judging association.

Additionally, if there is implied causality relationship, one-sided measures
might be preferred. For example, when performing feature selection, it seems
logical to measure the influence of features on the class label, not the other
way around.

Using Monte Carlo methods, it should be possible to test the validity of the
above two propositions as well as to visualize the effect of the assumptions
made.

References
----------

.. [1] `Sokal, R. R., & Rohlf, F. J. (2012). Biometry (4th edn). pp. 742-744.
       <http://www.amazon.com/dp/0716786044>`_

.. [2] `Wikipedia entry on Fisher's "Lady Tasting Tea" experiment
       <https://en.wikipedia.org/wiki/Lady_tasting_tea>`_

"""

import numpy as np
from math import log, sqrt, copysign
from collections import Set, namedtuple
from pymaptools.containers import CrossTab, OrderedCrossTab
from pymaptools.iter import ilen
from lsh_hdc.entropy import centropy, nchoose2, emi_from_margins


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


def jaccard_similarity(iterable1, iterable2):
    """Jaccard similarity between two sets

    Parameters
    ----------
    iterable1 : collections.Iterable
        first bag of items (order irrelevant)

    iterable2 : collections.Iterable
        second bag of items (order irrelevant)

    Returns
    -------

    jaccard_similarity : float
    """
    t = ConfusionMatrix2.from_sets(iterable1, iterable2)
    return t.jaccard_coeff()


def ratio2weights(ratio):
    """Numerically accurate conversion of ratio of two weights to weights
    """
    if ratio <= 1.0:
        lweight = ratio / (1.0 + ratio)
    else:
        lweight = 1.0 / (1.0 + 1.0 / ratio)
    return lweight, 1.0 - lweight


def geometric_mean(x, y):
    """Geometric mean of two numbers. Always returns a float

    Although geometric mean is defined for negative numbers, Scipy function
    doesn't allow it. Hence this function
    """
    prod = x * y
    if prod < 0.0:
        raise ValueError("x and y have different signs")
    return copysign(1, x) * sqrt(prod)


def geometric_mean_weighted(x, y, ratio=1.0):
    """Geometric mean of two numbers with a weight ratio. Returns a float

    ::

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
    """Harmonic mean of two numbers. Always returns a float
    """
    return float(x) if x == y else 2.0 * (x * y) / (x + y)


def harmonic_mean_weighted(x, y, ratio=1.0):
    """Harmonic mean of two numbers with a weight ratio. Returns a float

    ::

        >>> harmonic_mean_weighted(1, 3, ratio=1.0)
        1.5
        >>> harmonic_mean_weighted(1, 3, ratio=0.0)
        1.0
        >>> harmonic_mean_weighted(1, 3, ratio=float('inf'))
        3.0
    """
    lweight, rweight = ratio2weights(ratio)
    return float(x) if x == y else (x * y) / (lweight * x + rweight * y)


class ContingencyTable(CrossTab):

    # Note: not subclassing Pandas DataFrame because the goal is to specifically
    # optimize for sparse use cases when >90% of the table consists of zeros.
    # As of today, Pandas 'crosstab' implementation of frequency tables forces
    # one to iterate on all the zeros, which is horrible...

    __init__ = CrossTab.__init__

    def to_array(self):
        """Convert to NumPy array
        """
        return np.array(self.to_rows())

    def chisq_score(self):
        """Pearson's chi-square statistic
        """
        N = float(self.grand_total)
        score = 0.0
        for rm, cm, observed in self.iter_vals_with_margins():
            numer = rm * cm
            if numer != 0:
                expected = numer / N
                score += (observed - expected) ** 2 / expected
        return score

    def g_score(self):
        """G-statistic for RxC contingency table

        This method does not perform any corrections to this statistic (e.g.
        Williams', Yates' corrections).

        The statistic is equivalent to the negative of Mutual Information times
        two.  Mututal Information on a contingency table is defined as the
        difference between the information in the table and the information in
        an independent table with the same margins.  For application of mutual
        information (in the form of G-score) to search for collocated words in
        NLP, see [1]_ and [2]_.

        References
        ----------

        .. [1] `Dunning, T. (1993). Accurate methods for the statistics of
               surprise and coincidence. Computational linguistics, 19(1), 61-74.
               <http://dl.acm.org/citation.cfm?id=972454>`_

        .. [2] `Ted Dunning's personal blog entry and the discussion under it.
               <http://tdunning.blogspot.com/2008/03/surprise-and-coincidence.html>`_

        """
        _, _, I_CK = self._entropies()
        return 2.0 * I_CK

    def _entropies(self):
        """Return H_C, H_K, and mutual information

        Not normalized by N
        """
        H_C = centropy(self.row_totals)
        H_K = centropy(self.col_totals)
        H_actual = centropy(self.itervalues())
        H_expected = H_C + H_K
        I_CK = H_expected - H_actual
        return H_C, H_K, I_CK

    def mutual_info_score(self):
        """Mutual Information Score

        Mutual Information (divided by N).

        The metric is equal to the Kullback-Leibler divergence of the joint
        distribution with the product distribution of the marginals.
        """
        _, _, I_CK = self._entropies()
        return I_CK / self.grand_total

    def entropy_metrics(self):
        """Gives three entropy-based metrics for a RxC table

        The metrics are: Homogeneity, Completeness, and V-measure

        The V-measure metric is also known as Normalized Mutual Information
        (NMI), and is calculated here as the harmonic mean of Homogeneity and
        Completeness (:math:`NMI_{sum}`). There exist other definitions of NMI (see
        Table 2 in [1]_ for a good review).

        Homogeneity and Completeness are duals of each other and can be thought
        of (although this is not technically accurate) as squared regression
        coefficients of a given clustering vs true labels (homogeneity) and of
        the dual problem of true labels vs given clustering (completeness).
        Because of the dual property, in a symmetric matrix, all three scores
        are the same.

        This method replaces ``homogeneity_completeness_v_measure`` method in
        Scikit-Learn.  The Scikit-Learn version takes up :math:`O(n^2)` space
        because it stores data in a dense NumPy array, while the given version
        is sub-quadratic because of sparse underlying storage.

        Note that the entropy variables H in the code below are improperly
        defined because they ought to be divided by N (the grand total for the
        contingency table). However, the N variable cancels out during
        normalization.

        References
        ----------

        .. [1] `Vinh, N. X., Epps, J., & Bailey, J. (2010). Information theoretic
               measures for clusterings comparison: Variants, properties,
               normalization and correction for chance. The Journal of Machine
               Learning Research, 11, 2837-2854.
               <http://www.jmlr.org/papers/v11/vinh10a.html>`_

        """
        # ensure non-negative values by taking max of 0 and given value
        H_C, H_K, I_CK = self._entropies()
        h = 1.0 if H_C == 0.0 else max(0.0, I_CK / H_C)
        c = 1.0 if H_K == 0.0 else max(0.0, I_CK / H_K)
        rsquare = harmonic_mean(h, c)
        return h, c, rsquare

    homogeneity_completeness_v_measure = entropy_metrics

    def adjusted_mutual_info_score(self):
        """Adjusted Mutual Information between two clusterings

        For a mathematical definition, see [1]_ and [2]_.

        References
        ----------

        .. [1] `Vinh, N. X., Epps, J., & Bailey, J. (2009, June). Information
               theoretic measures for clusterings comparison: is a correction
               for chance necessary?. In Proceedings of the 26th Annual
               International Conference on Machine Learning (pp. 1073-1080).
               ACM.
               <https://doi.org/10.1145/1553374.1553511>`_

        .. [2] `Vinh, N. X., Epps, J., & Bailey, J. (2010). Information theoretic
               measures for clusterings comparison: Variants, properties,
               normalization and correction for chance. The Journal of Machine
               Learning Research, 11, 2837-2854.
               <http://www.jmlr.org/papers/v11/vinh10a.html>`_

        """
        # Prepare row totals and check for special cases
        row_totals = np.fromiter(self.iter_row_totals(), dtype=np.int64)
        col_totals = np.fromiter(self.iter_col_totals(), dtype=np.int64)
        R = len(row_totals)
        C = len(col_totals)
        if R == C == 1 or R == C == 0:
            # No clustering since the data is not split. This is a perfect match
            # hence return 1.0.
            return 1.0

        # In one step, calculate entropy for each labeling and mutual
        # information
        h_true, h_pred, mi = self._entropies()
        mi_max = max(h_true, h_pred)

        # Calculate the expected value for the MI
        emi = emi_from_margins(row_totals, col_totals)

        # Calculate the adjusted MI score
        ami = (mi - emi) / (mi_max - emi)
        return ami

    def vi_distance(self, normalize=True):
        """Variation of Information distance

        Defined in [1]_. This measure is one of several possible entropy- based
        distance measure that could be defined on a RxC matrix. The given
        measure is equivalent to :math:`2 D_{sum}` as listed in Table 2 in
        [2]_.

        Note that the entropy variables H below are calculated using natural
        logs, so a base correction may be necessary if you need your result in
        base 2 for example.

        References
        ----------

        .. [1] `Meila, M. (2007). Comparing clusterings -- an information based
               distance. Journal of multivariate analysis, 98(5), 873-895.
               <https://doi.org/10.1016/j.jmva.2006.11.013>`_

        .. [2] `Vinh, N. X., Epps, J., & Bailey, J. (2010). Information theoretic
               measures for clusterings comparison: Variants, properties,
               normalization and correction for chance. The Journal of Machine
               Learning Research, 11, 2837-2854.
               <http://www.jmlr.org/papers/v11/vinh10a.html>`_

        """
        H_C, H_K, I_CK = self._entropies()
        VI_CK = (H_C + H_K) - (I_CK + I_CK)
        score = _div(VI_CK, self.grand_total)
        if normalize:
            score /= log(self.grand_total)
        return score

    def vi_similarity(self, normalize=True):
        """Inverse of ``vi_distance``
        """
        dist = self.vi_distance(normalize=False)
        max_dist = log(self.grand_total)
        score = max_dist - dist
        if normalize:
            score /= max_dist
        return score

    def split_join_distance(self, normalize=True):
        """Projection distance between partitions

        Used in graph community analysis. Originally defined in [1]_.
        Example::

            >>> p1 = [{1, 2, 3, 4}, {5, 6, 7}, {8, 9, 10, 11, 12}]
            >>> p2 = [{2, 4, 6, 8, 10}, {3, 9, 12}, {1, 5, 7}, {11}]
            >>> t = ClusteringMetrics.from_partitions(p1, p2)
            >>> t.split_join_distance(normalize=False)
            11

        References
        ----------

        .. [1] `Dongen, S. V. (2000). Performance criteria for graph clustering
               and Markov cluster experiments. Information Systems [INS],
               (R 0012), 1-36.
               <http://dl.acm.org/citation.cfm?id=868979>`_

        """
        sim = self.split_join_similarity(normalize=False)
        max_sim = 2 * self.grand_total
        score = max_sim - sim
        if normalize:
            score /= max_sim
        return score

    def split_join_similarity(self, normalize=True):
        """Split-join similarity score

        A relatively decent clustering::

            >>> a = [ 1,  1,  1,  2,  2,  2,  2,  3,  3,  4]
            >>> b = [43, 56, 56,  5, 36, 36, 36, 74, 74, 66]
            >>> t = ContingencyTable.from_labels(a, b)
            >>> t.split_join_similarity()
            0.9

        Less good clustering::

            >>> clusters = [[1, 1], [1, 1, 1, 1], [2, 3], [2, 2, 3, 3],
            ...             [3, 3, 4], [3, 4, 4, 4, 4, 4, 4, 4, 4, 4]]
            >>> t = ContingencyTable.from_clusters(clusters)
            >>> t.split_join_similarity()
            0.74

        """
        pa_B = sum(max(row) for row in self.iter_rows())
        pb_A = sum(max(col) for col in self.iter_cols())
        score = pa_B + pb_A
        if normalize:
            score /= float(2 * self.grand_total)
        return score

    def mirkin_match_coeff(self, normalize=True):
        """Equivalence match (similarity) coefficient

        ::

            >>> C3 = [{1, 2, 3, 4}, {5, 6, 7, 8, 9, 10}, {11, 12, 13, 14, 15, 16}]
            >>> C4 = [{1, 2, 3, 4}, {5, 6, 7, 8, 9, 10, 11, 12}, {13, 14, 15, 16}]
            >>> t = ClusteringMetrics.from_partitions(C3, C4)
            >>> t.mirkin_match_coeff(normalize=False)
            216
        """
        max_score = self.grand_total ** 2
        score = max_score - self.mirkin_mismatch_coeff(normalize=False)
        if normalize:
            score /= float(max_score)
        return score

    def mirkin_mismatch_coeff(self, normalize=True):
        """Equivalence mismatch (distance) coefficient

        Described in [1]_.

        ::

            >>> C1 = [{1, 2, 3, 4, 5, 6, 7, 8}, {9, 10, 11, 12, 13, 14, 15, 16}]
            >>> C2 = [{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, {11, 12, 13, 14, 15, 16}]
            >>> t = ClusteringMetrics.from_partitions(C1, C2)
            >>> t.mirkin_mismatch_coeff(normalize=False)
            56

        References
        ----------

        .. [1] `Mirkin, B (1996). Mathematical Classification and Clustering.
               Kluwer Academic Press: Boston-Dordrecht.
               <http://www.amazon.com/dp/0792341597>`_

        """
        score = (
            sum(x ** 2 for x in self.iter_row_totals()) +
            sum(x ** 2 for x in self.iter_col_totals()) -
            2 * sum(x ** 2 for x in self.itervalues())
        )
        if normalize:
            score /= float(self.grand_total ** 2)
        return score

    def talburt_wang_index(self):
        """Talburt-Wang index of similarity of two partitions

        A relatively decent clustering::

            >>> a = [ 1,  1,  1,  2,  2,  2,  2,  3,  3,  4]
            >>> b = [43, 56, 56,  5, 36, 36, 36, 74, 74, 66]
            >>> t = ContingencyTable.from_labels(a, b)
            >>> round(t.talburt_wang_index(), 3)
            0.816

        Less good clustering (example from [1]_)::

            >>> clusters = [[1, 1], [1, 1, 1, 1], [2, 3], [2, 2, 3, 3],
            ...             [3, 3, 4], [3, 4, 4, 4, 4, 4, 4, 4, 4, 4]]
            >>> t = ContingencyTable.from_clusters(clusters)
            >>> round(t.talburt_wang_index(), 2)
            0.49

        References
        ----------

        .. [1] `Talburt, J., Wang, R., Hess, K., & Kuo, E. (2007). An algebraic
               approach to data quality metrics for entity resolution over large
               datasets.  Information quality management: Theory and
               applications, 1-22.
               <http://www.igi-global.com/chapter/algebraic-approach-data-quality-metrics/23022>`_
        """
        A_card = ilen(self.iter_row_totals())
        B_card = ilen(self.iter_col_totals())
        V_card = sum(ilen(row) for row in self.iter_rows())
        return _div(sqrt(A_card * B_card), V_card)


class ClusteringMetrics(ContingencyTable):

    """Provides external clustering evaluation metrics

    A subclass of ContingencyTable that builds a pairwise co-association matrix
    for clustering comparisons.
    """

    def __init__(self, *args, **kwargs):
        ContingencyTable.__init__(self, *args, **kwargs)
        self._pairwise_ = None

    @property
    def pairwise_(self):
        """Confusion matrix on all pair assignments from two partitions

        A partition of N is a set of disjoint clusters s.t. every point in N
        belongs to one and only one cluster and every cluster consits of at
        least one point. Given two partitions A and B and a co-occurrence
        matrix of point pairs,

        == =============================================================
        TP count of pairs found in the same partition in both A and B
        FP count of pairs found in the same partition in A but not in B
        FN count of pairs found in the same partition in B but not in A
        TN count of pairs in different partitions in both A and B
        == =============================================================

        Note that although the resulting confusion matrix has the form of a
        correlation table for two binary variables, it is not symmetric if the
        original partitions are not symmetric.

        """
        pairwise = self._pairwise_
        if pairwise is None:
            actual_positives = sum(nchoose2(b) for b in self.iter_row_totals())
            called_positives = sum(nchoose2(a) for a in self.iter_col_totals())
            TP = sum(nchoose2(cell) for cell in self.itervalues())
            FN = actual_positives - TP
            FP = called_positives - TP
            TN = nchoose2(self.grand_total) - TP - FP - FN
            pairwise = self._pairwise_ = ConfusionMatrix2.from_ccw(TP, FP, TN, FN)
        return pairwise

    def get_score(self, scoring_method, *args, **kwargs):
        """Evaluate specified scoring method
        """
        try:
            method = getattr(self, scoring_method)
        except AttributeError:
            method = getattr(self.pairwise_, scoring_method)
        return method(*args, **kwargs)

    def adjusted_rand_score(self):
        """Rand score (accuracy) corrected for chance

        This is a memory-efficient replacement for a similar Scikit-Learn
        function.
        """
        return self.pairwise_.kappa()


confmat2_type = namedtuple("ConfusionMatrix2", "TP FP TN FN")


class ConfusionMatrix2(ContingencyTable, OrderedCrossTab):
    """A confusion matrix (2x2 contingency table)

    For a binary variable (where one is measuring either presence vs absence of
    a particular feature), a confusion matrix where the ground truth levels are
    rows looks like this::

        >>> ConfusionMatrix2(TP=20, FN=31, FP=14, TN=156).to_array()
        array([[ 20,  31],
               [ 14, 156]])

    For a nominal variable, the negative class becomes a distinct label, and
    TP/FP/FN/TN terminology does not apply, although the algorithms should work
    the same way (with the obvious distinction that different assumptions will
    be made). For a convenient reference about some of the attributes and
    methods defined here see [1]_.

    Attributes
    ----------

    TP :
        True positive count
    FP :
        False positive count
    TN :
        True negative count
    FN :
        False negative count

    References
    ----------

    .. [1] `Wikipedia entry for Confusion Matrix
            <https://en.wikipedia.org/wiki/Confusion_matrix>`_
    """

    def __repr__(self):
        return "ConfusionMatrix2(rows=%s)" % repr(self.to_rows())

    def __init__(self, TP=None, FN=None, FP=None, TN=None, rows=None):
        if rows is None:
            rows = ((TP, FN), (FP, TN))
        ContingencyTable.__init__(self, rows=rows)

    @classmethod
    def from_sets(cls, set1, set2, universe_size=None):
        """Instantiate from two sets

        Accepts an optional universe_size parameter which allows us to take into
        account TN class and use probability-based similarity metrics.  Most of
        the time, however, set comparisons are performed ignoring this parameter
        and relying instead on non-probabilistic indices such as Jaccard's or
        Dice.
        """
        if not isinstance(set1, Set):
            set1 = set(set1)
        if not isinstance(set2, Set):
            set2 = set(set2)
        TP = len(set1 & set2)
        FP = len(set2) - TP
        FN = len(set1) - TP
        if universe_size is None:
            TN = 0
        else:
            TN = universe_size - TP - FP - FN
            if TN < 0:
                raise ValueError(
                    "universe_size must be at least as large as set union")
        return cls(TP, FN, FP, TN)

    @classmethod
    def from_random_counts(cls, low=0, high=100):
        """Instantiate from random values
        """
        return cls(*np.random.randint(low=low, high=high, size=(4,)))

    @classmethod
    def from_ccw(cls, TP, FP, TN, FN):
        """Instantiate from counter-clockwise form of TP FP TN FN
        """
        return cls(TP, FN, FP, TN)

    def to_ccw(self):
        """Convert to counter-clockwise form of TP FP TN FN
        """
        return confmat2_type(TP=self.TP, FP=self.FP, TN=self.TN, FN=self.FN)

    def get_score(self, scoring_method, *args, **kwargs):
        """Evaluate specified scoring method
        """
        method = getattr(self, scoring_method)
        return method(*args, **kwargs)

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
        """Accuracy (Rand Index)

        Synonyms: Simple Matching Coefficient, Rand Index
        """
        return _div(self.TP + self.TN, self.grand_total)

    def PPV(self):
        """Positive Predictive Value (Precision)

        Synonyms: precision, frequency of hits, post agreement, success ratio,
        correct alarm ratio
        """
        return _div(self.TP, self.TP + self.FP)

    def NPV(self):
        """Negative Predictive Value

        Synonyms: frequency of correct null forecasts
        """
        return _div(self.TN, self.TN + self.FN)

    def TPR(self):
        """True Positive Rate (Recall, Sensitivity)

        Synonyms: recall, sensitivity, hit rate, probability of detection,
        prefigurance
        """
        return _div(self.TP, self.TP + self.FN)

    def FPR(self):
        """False Positive Rate

        Synonyms: fallout
        """
        return _div(self.FP, self.TN + self.FP)

    def TNR(self):
        """True Negative Rate (Specificity)

        Synonyms: specificity
        """
        return _div(self.TN, self.FP + self.TN)

    def FNR(self):
        """False Negative Rate

        Synonyms: miss rate, frequency of misses
        """
        return _div(self.FN, self.TP + self.FN)

    def FDR(self):
        """False discovery rate

        Synonyms: false alarm ratio, probability of false alarm
        """
        return _div(self.FP, self.TP + self.FP)

    def FOR(self):
        """False omission rate

        Synonyms: detection failure ratio, miss ratio
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

        Defined as

        .. math::

            DOR = \\frac{PLL}{NLL}.

        See Also
        --------

        PLL, NLL

        """
        return _div(self.TP * self.TN, self.FP * self.FN)

    def fscore(self, beta=1.0):
        """F-score

        As beta tends to infinity, F-score will approach recall.  As beta tends
        to zero, F-score will approach precision. A similarity coefficient that
        uses a similar definition is called Dice coefficient.

        See Also
        --------
        dice_coeff
        """
        return harmonic_mean_weighted(self.precision(), self.recall(), beta ** 2)

    def dice_coeff(self):
        """Dice similarity (Nei-Li coefficient)

        This is the same as F1-score, but calculated slightly differently here.
        Note that Dice can be zero if total number of positives is zero, but
        F-score is undefined in that case (because recall is undefined).

        See Also
        --------
        fscore, jaccard_coeff, ochiai_coeff
        """
        (a, b), (c, _) = self.rows
        return _div(2 * a, 2 * a + b + c)

    def jaccard_coeff(self):
        """Jaccard similarity coefficient

        Jaccard coefficient has an interesting property in that in L-shaped
        matrices where either FP or FN are close to zero, its scale becomes
        equivalent to the scale of either recall or precision respectively.

        Synonyms: critical success index

        See Also
        --------
        dice_coeff, ochiai_coeff
        """
        (a, b), (c, _) = self.rows
        return _div(a, a + b + c)

    def ochiai_coeff(self):
        """Ochiai similarity coefficient (Fowlkes-Mallows)

        Gives cosine similarity for a 2x2 table. Also known as Fowlkes-Mallows
        Index in clustering evaluation.

        This similarity index has an interpretation that it is the geometric
        mean of the conditional probability of an element (in the case of
        pairwise clustering comparison, a pair of elements) belonging to the
        same cluster given that they belong to the same class [1]_.

        See Also
        --------
        jaccard_coeff, dice_coeff

        References
        ----------

        .. [1] `Ramirez, E. H., Brena, R., Magatti, D., & Stella, F. (2012).
               Topic model validation. Neurocomputing, 76(1), 125-133.
               <http://dx.doi.org/10.1016/j.neucom.2011.04.032>`_
        """
        (a, b), (c, _) = self.rows
        return _div(a, sqrt((a + b) * (a + c)))

    def sokal_sneath_coeff(self):
        """Sokal and Sneath similarity index

        In a 2x2 matrix

        .. math::

            \\begin{matrix} a & b \\\\ c & d \\end{matrix}

        Dice places more weight on :math:`a` component, Jaccard places equal
        weight on :math:`a` and :math:`b + c`, while Sokal and Sneath places
        more weight on :math:`b + c`.

        See Also
        --------
        dice_coeff, jaccard_coeff
        """
        (a, b), (c, _) = self.rows
        return _div(a, a + 2 * (b + c))

    def prevalence_index(self):
        """Prevalence

        In interrater agreement studies, prevalence is high when the proportion
        of agreements on the positive classification differs from that of the
        negative classification.  Example of a confusion matrix with high
        prevalence:

        .. math::

            \\begin{matrix} 3 & 27 \\\\ 28 & 132 \\end{matrix}

        In the example given, both raters agree that there are very few positive
        examples relative to the number of negatives. In other word, the
        negative rating is very prevalent.

        See Also
        --------

        bias_index
        """
        return _div(abs(self.TP - self.TN), self.grand_total)

    def frequency_bias(self):
        """Frequency bias

        How much more often is rater B is predicting TP
        """
        return _div(self.TP + self.FP, self.TP + self.FN)

    def bias_index(self):
        """Bias Index

        In interrater agreement studies, bias is the extent to which the raters
        disagree on the positive-negative ratio of the binary variable studied.
        Example of a confusion matrix with high bias:

        .. math::

            \\begin{matrix} 17 & 14 \\\\ 78 & 81 \\end{matrix}

        Note that the rater whose judgment is represented by rows (A) believes
        there are a lot more negative examples than positive ones, while the
        rater whose judgment is represented by columns (B) thinks the number of
        positives is roughly equal to the number of negatives. In other words,
        the rater A appears to be negatively biased.

        See Also
        --------

        prevalence_index
        """
        return _div(abs(self.FN - self.FP), self.grand_total)

    def informedness(self):
        """Informedness (recall corrected for chance)

        A complement to markedness. Can be thought of as recall corrected for
        chance. Alternative formulations:

        .. math::

            Informedness &= Sensitivity + Specificity - 1.0 \\\\
                         &= TPR - FPR

        In the case of ranked predictions, TPR can be plotted on the y-axis
        with FPR on the x-axis. The resulting plot is known as Receiver
        Operating Characteristic (ROC) curve [1]_. The delta between a point on
        the ROC curve and the diagonal is equal to the value of informedness at
        the given FPR threshold.

        This measure was first proposed for evaluating medical diagnostics
        tests in [2]_, and was also used in meteorology under the name "True
        Skill Score" [3]_.

        Synonyms: Youden's J, True Skill Score, Hannssen-Kuiper Score,
        Attributable Risk, DeltaP.

        See Also
        --------

        markedness

        References
        ----------

        .. [1] `Fawcett, T. (2006). An introduction to ROC analysis. Pattern
               recognition letters, 27(8), 861-874.
               <http://doi.org/10.1016/j.patrec.2005.10.010>`_

        .. [2] `Youden, W. J. (1950). Index for rating diagnostic tests. Cancer,
               3(1), 32-35.
               <http://www.ncbi.nlm.nih.gov/pubmed/15405679>`_

        .. [3] `Doswell III, C. A., Davies-Jones, R., & Keller, D. L. (1990). On
               summary measures of skill in rare event forecasting based on
               contingency tables. Weather and Forecasting, 5(4), 576-585.
               <http://journals.ametsoc.org/doi/abs/10.1175/1520-0434%281990%29005%3C0576%3AOSMOSI%3E2.0.CO%3B2>`_
        """
        p1, q1 = self.row_totals.values()
        return _div(self.covar(), p1 * q1)

    def markedness(self):
        """Markedness (precision corrected for chance)

        A complement to informedness. Can be thought of as precision corrected
        for chance. Alternative formulations:

        .. math::

            Markedness &= PPV + NPV - 1.0 \\\\
                       &= PPV - FOR

        In the case of ranked predictions, PPV can be plotted on the y-axis
        with FOR on the x-axis. The resulting plot is known as Relative
        Operating Level (ROL) curve [1]_. The delta between a point on the ROL
        curve and the diagonal is equal to the value of markedness at the given
        FOR threshold.

        Synonyms: DeltaP′

        See Also
        --------

        informedness

        References
        ----------

        .. [1] `Mason, S. J., & Graham, N. E. (2002). Areas beneath the
               relative operating characteristics (ROC) and relative
               operating levels (ROL) curves: Statistical significance
               and interpretation. Quarterly Journal of the Royal
               Meteorological Society, 128(584), 2145-2166.
               <https://doi.org/10.1256/003590002320603584>`_
        """
        p2, q2 = self.col_totals.values()
        return _div(self.covar(), p2 * q2)

    def kappa0(self):
        """One-sided component of ``kappa`` and MCC

        Roughly corresponds to precision
        """
        _, q1 = self.row_totals.values()
        p2, _ = self.col_totals.values()
        return _div(self.covar(), p2 * q1)

    def kappa1(self):
        """One-sided component of ``kappa`` and MCC

        Roughly corresponds to recall
        """
        p1, _ = self.row_totals.values()
        _, q2 = self.col_totals.values()
        return _div(self.covar(), p1 * q2)

    def loevinger_coeff(self):
        """Loevinger two-sided coefficient of homogeneity

        Given a clustering (numbers correspond to class labels, inner groups to
        clusters) with perfect homogeneity but imperfect completeness, Loevinger
        coefficient returns a perfect score on the corresponding pairwise
        co-association matrix::

            >>> clusters = [[0, 0], [0, 0, 0, 0], [1, 1, 1, 1]]
            >>> t = ClusteringMetrics.from_clusters(clusters)
            >>> t.pairwise_.loevinger_coeff()
            1.0

        At the same time, kappa and Matthews coefficients are 0.63 and 0.68,
        respectively. Being symmetrically defined, Loevinger coefficient will
        also return a perfect score in the dual (opposite) situation::

            >>> clusters = [[0, 2, 2, 0, 0, 0], [1, 1, 1, 1]]
            >>> t = ClusteringMetrics.from_clusters(clusters)
            >>> t.pairwise_.loevinger_coeff()
            1.0

        Loevinger's coefficient has a unique property: all relevant two-way
        correlation coefficients on a 2x2 table (including Kappa and Matthews'
        Correlation Coefficient) become Loevinger's coefficient after
        normalization by maximum value [1]_.

        References
        ----------

        .. [1] `Warrens, M. J. (2008). On association coefficients for 2x2
               tables and properties that do not depend on the marginal
               distributions.  Psychometrika, 73(4), 777-789.
               <https://doi.org/10.1007/s11336-008-9070-3>`_

        """
        p1, q1 = self.row_totals.values()
        p2, q2 = self.col_totals.values()
        return _div(self.covar(), min(p1 * q2, p2 * q1))

    def kappa(self):
        """Cohen's Kappa (Interrater Agreement)

        Kappa coefficient is best known in the psychology field where it was
        introduced to measure interrater agreement [1]_. It has also been used
        in replication studies [2]_, clustering evaluation [3]_, image
        segmentation [4]_, feature selection [5]_, and forecasting [56]_. The
        first derivation of this measure is in [7]_.

        Kappa can be derived by correcting Accuracy (Simple Matching
        Coefficient, Rand Index) for chance. Tbe general formula for chance
        correction of an association measure :math:`M` is:

        .. math::

            M_{adj} = \\frac{M - E(M)}{M_{max} - E(M)},

        where :math:`M_{max}` is the maximum value a measure :math:`M` can
        achieve, and :math:`E(M)` is the expected value of :math:`M` under
        statistical independence given fixed table margins.

        Kappa can be decomposed into a pair of components (regression
        coefficients), :math:`k_1` (recall-like) and :math:`k_0`
        (precision-like), of which it is a harmonic mean:

        .. math::

            k_1 = \\frac{cov}{p_1 q_2},

            k_0 = \\frac{cov}{p_2 q_1}.

        It is interesting to note that if one takes a geometric mean of the
        above two components, one obtains Matthews' Correlation Coefficient.
        The latter is also obtained from a geometric mean of informedness and
        markedness (which are similar to, but not the same, as :math:`k_1` and
        :math:`k_0`).  Unlike informedness and markedness, :math:`k_1` and
        :math:`k_0` don't have a lower bound.  For that reason, when
        characterizing one-way dependence in a 2x2 confusion matrix, it is
        arguably better to use use informedness and markedness.

        As 'd' approaches infinity, kappa turns into Dice coefficient
        (F-score).

        Synonyms: Adjusted Rand Index, Heidke Skill Score

        References
        ----------

        .. [1] `Cohen, J. (1960). A coefficient of agreement for nominal scales.
               Educational and psychological measurement, 20(1), 37-46.
               <https://doi.org/10.1177/001316446002000104>`_

        .. [2] `Arabie, P., Hubert, L. J., & De Soete, G. (1996). Clustering
               validation: results and implications for applied analyses (p.
               341).  World Scientific Pub Co Inc.
               <https://doi.org/10.1142/9789812832153_0010>`_

        .. [3] `Warrens, M. J. (2008). On the equivalence of Cohen's kappa and
               the Hubert-Arabie adjusted Rand index. Journal of Classification,
               25(2), 177-183.
               <https://doi.org/10.1007/s00357-008-9023-7>`_

        .. [4] `Briggman, K., Denk, W., Seung, S., Helmstaedter, M. N., &
               Turaga, S. C. (2009). Maximin affinity learning of image
               segmentation. In Advances in Neural Information Processing
               Systems (pp. 1865-1873).
               <http://books.nips.cc/papers/files/nips22/NIPS2009_0084.pdf>`_

        .. [5] `Santos, J. M., & Embrechts, M. (2009). On the use of the
               adjusted rand index as a metric for evaluating supervised
               classification. In Artificial neural networks - ICANN 2009 (pp.
               175-184).  Springer Berlin Heidelberg.
               <https://doi.org/10.1007/978-3-642-04277-5_18>`_

        .. [6] `Doswell III, C. A., Davies-Jones, R., & Keller, D. L. (1990). On
               summary measures of skill in rare event forecasting based on
               contingency tables. Weather and Forecasting, 5(4), 576-585.
               <http://journals.ametsoc.org/doi/abs/10.1175/1520-0434%281990%29005%3C0576%3AOSMOSI%3E2.0.CO%3B2>`_

        .. [7] `Heidke, Paul. "Berechnung des Erfolges und der Gute der
               Windstarkevorhersagen im Sturmwarnungsdienst." Geografiska
               Annaler (1926): 301-349.
               <http://www.jstor.org/stable/519729>`_

        """
        (a, b), (c, d) = self.rows
        p1 = a + b
        q1 = c + d
        p2 = a + c
        q2 = b + d
        n = p1 + q1
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
        """Maxwell & Pilliner's association index

        Another covariance-based association index corrected for chance. Turns
        into Dice coefficient (F-score) as 'd' approaches infinity.
        """
        (a, b), (c, d) = self.rows
        p1 = a + b
        q1 = c + d
        p2 = a + c
        q2 = b + d
        n = p1 + q1
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

        MCC is directly related to the Chi-square statistic. Its value is equal
        to the Chi-square value normalized by the maximum value Chi-Square
        can achieve with given margins (for a 2x2 table, the maximum Chi-square
        score is equal to the grand total N) transformed to correlation space by
        taking a square root.

        MCC is a also a geometric mean of informedness and markedness (the
        regression coefficients of the problem and its dual). As the value of
        'd' approaches infinity, MCC turns into Ochiai coefficient.

        Other names for MCC are Phi Coefficient and Yule's Q with correction for
        chance.
        """
        (a, b), (c, d) = self.rows
        p1 = a + b
        q1 = c + d
        p2 = a + c
        q2 = b + d
        n = p1 + q1
        if a == n or b == n or c == n or d == n:
            # only one cell is non-zero
            return np.nan
        elif p1 == 0 or p2 == 0 or q1 == 0 or q2 == 0:
            # one row or column is zero, another non-zero
            return 0.0
        else:
            # no more than one cell is zero
            return _div(self.covar(), sqrt(p1 * q1 * p2 * q2))

    def mi_corr1(self):
        """One-sided component of ``mi_corr``

        Roughly equivalent to informedness
        """
        h, _, _ = self.entropy_metrics()
        return copysign(1, self.covar()) * sqrt(h)

    def mi_corr0(self):
        """One-sided component of ``mi_corr``

        Roughly equivalent to markedness
        """
        _, c, _ = self.entropy_metrics()
        return copysign(1, self.covar()) * sqrt(c)

    def mi_corr(self):
        """Mutual information-based correlation

        The coefficient decomposes into regression coefficients defined
        according to fixed-margin tables. The ``mi_corr1`` coefficient, for
        example, is obtained by dividing the G-score by the maximum achievable
        value on a table with fixed true class counts (which here correspond to
        row totals).  The ``mi_corr0`` is its dual, defined by dividing the
        G-score by its maximum achievable value with fixed predicted label
        counts (here represented as column totals).
        """
        _, _, rsquare = self.entropy_metrics()
        return copysign(1, self.covar()) * sqrt(rsquare)

    def yule_q(self):
        """Yule's Q (association index)

        this index relates to the D odds ratio:

        .. math::

            Q = \\frac{DOR - 1}{DOR + 1}.

        """
        (a, b), (c, d) = self.rows
        return _div(self.covar(), a * d + b * c)

    def yule_y(self):
        """Yule's Y (colligation coefficient)

        The Y metric was used to produce a new association metric by adjusting
        for entropy in [1]_.

        References
        ----------

        .. [1] `Hasenclever, D., & Scholz, M. (2013). Comparing measures of
                association in 2x2 probability tables. arXiv preprint
                arXiv:1302.6161.
                <http://arxiv.org/pdf/1302.6161v1.pdf>`_

        """
        (a, b), (c, d) = self.rows
        ad = a * d
        bc = b * c
        return _div(sqrt(ad) - sqrt(bc),
                    sqrt(ad) + sqrt(bc))

    def covar(self):
        """Covariance (determinant of a 2x2 matrix)
        """
        (a, b), (c, d) = self.rows
        return a * d - b * c

    # various silly terminologies follow

    # information retrieval
    precision = PPV
    recall = TPR
    accuracy = ACC
    # fallout = FPR

    # clinical diagnostics
    sensitivity = TPR
    specificity = TNR
    # youden_j = informedness

    # sales/marketing
    # hit_rate = TPR
    # miss_rate = FNR

    # ecology
    # sm_coeff = ACC
    # phi_coeff = matthews_corr

    # meteorology
    # heidke_skill = kappa
    # true_skill = informedness

    # cluster analysis
    rand_index = ACC
    adjusted_rand_score = kappa
    fowlkes_mallows = ochiai_coeff


def mutual_info_score(labels_true, labels_pred):
    """Memory-efficient replacement for equivalently named Sklean function
    """
    ct = ContingencyTable.from_labels(labels_true, labels_pred)
    return ct.mutual_info_score()


def homogeneity_completeness_v_measure(labels_true, labels_pred):
    """Memory-efficient replacement for equivalently named Scikit-Learn function
    """
    ct = ContingencyTable.from_labels(labels_true, labels_pred)
    return ct.entropy_metrics()


def adjusted_rand_score(labels_true, labels_pred):
    """Rand score (accuracy) corrected for chance

    This is a memory-efficient replacement for the equivalently named
    Scikit-Learn function

    In a supplement to [1]_, the following example is given::

        >>> classes = [1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
        >>> clusters = [1, 2, 1, 2, 2, 3, 3, 3, 3, 3]
        >>> round(adjusted_rand_score(classes, clusters), 3)
        0.313

    References
    ----------

    .. [1] `Yeung, K. Y., & Ruzzo, W. L. (2001). Details of the adjusted Rand
            index and clustering algorithms, supplement to the paper "An empirical
            study on principal component analysis for clustering gene expression
            data". Bioinformatics, 17(9), 763-774.
            <http://faculty.washington.edu/kayee/pca/>`_

    """
    ct = ClusteringMetrics.from_labels(labels_true, labels_pred)
    return ct.adjusted_rand_score()


def adjusted_mutual_info_score(labels_true, labels_pred):
    """Adjusted Mutual Information between two clusterings

    This is a memory-efficient replacement for the equivalently named
    Scikit-Learn function.

    Perfect labelings are both homogeneous and complete, hence AMI has the
    perfect score of one::

        >>> adjusted_mutual_info_score([0, 0, 1, 1], [0, 0, 1, 1])
        1.0
        >>> adjusted_mutual_info_score([0, 0, 1, 1], [1, 1, 0, 0])
        1.0

    If classes members are completely split across different clusters, the
    assignment is utterly incomplete, hence AMI equals zero::

        >>> adjusted_mutual_info_score([0, 0, 0, 0], [0, 1, 2, 3])
        0.0

    """
    t = ClusteringMetrics.from_labels(labels_true, labels_pred)
    return t.adjusted_mutual_info_score()


def matthews_corr(*args, **kwargs):
    """Return MCC score for a 2x2 contingency table
    """
    return ConfusionMatrix2.from_ccw(*args, **kwargs).matthews_corr()


def cohen_kappa(*args, **kwargs):
    """Return Cohen's Kappa for a 2x2 contingency table
    """
    return ConfusionMatrix2.from_ccw(*args, **kwargs).kappa()
