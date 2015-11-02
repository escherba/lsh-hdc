import warnings
import numpy as np
from itertools import izip
from operator import itemgetter
from sklearn.metrics.ranking import auc, roc_curve
from pymaptools.iter import aggregate_tuples
from pymaptools.containers import labels_to_clusters


class RocCurve(object):

    """Receiver Operating Characteristic (ROC)

    ::

        >>> rc = RocCurve.from_binary([0, 0, 1, 1],
        ...                           [0.1, 0.4, 0.35, 0.8])
        >>> rc.auc_score()
        0.75
        >>> rc.max_informedness()
        0.5

    """
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
        """Replacement for Scikit-Learn's method

        If number of Y classes is other than two, a warning will be triggered
        but no exception thrown (the return value will be a NaN).  Also, we
        don't reorder arrays during ROC calculation since they are assumed to be
        in order.
        """
        return auc(self.fprs, self.tprs, reorder=False)

    def optimal_cutoff(self, scoring_method):
        """Calculate optimal cutoff point according to ``scoring_method`` lambda

        The scoring method must take two arguments: fpr and tpr.
        """
        max_index = np.NINF
        opt_pair = (np.nan, np.nan)
        for pair in izip(self.fprs, self.tprs):
            index = scoring_method(*pair)
            if index > max_index:
                opt_pair = pair
                max_index = index
        return opt_pair, max_index

    @staticmethod
    def _informedness(fpr, tpr):
        return tpr - fpr

    def max_informedness(self):
        """Calculates maximum value of Informedness on a ROC curve

        This is also known as Youden's J [1]_

        References
        ----------

        .. [1] `Wikipedia entry for Youden's J statistic
               <https://en.wikipedia.org/wiki/Youden%27s_J_statistic>`_
        """
        return self.optimal_cutoff(self._informedness)[1]


def roc_auc_score(y_true, y_score, sample_weight=None):
    """Replaces Scikit Learn implementation (for binary y_true vectors only)
    """
    return RocCurve.from_binary(y_true, y_score).auc_score()


class LiftCurve(object):

    """Area under Lift Curve (AUL) for cluster-size correlated classification

    The AUL measure here is similar to Gini coefficient of inequality [1]_
    except we plot the corresponding curve in the opposite direction (from
    "richest" to "poorest"), and do not subtract 0.5 from the resulting score.

    Assume that there is a large data set of mostly unique samples where a
    hidden binary variable is dependent on the number of similar samples that
    exist in the set (i.e. a sample is called positive if it has many neighbors)
    and that our goal is to label all samples in this set. It is easy to see
    that, given sparse enough data, if a clustering method relies on the same
    sample property on which the ground truth similarity space is defined, it
    will naturally separate the samples into two groups -- those found in
    clusters and containing mostly positives, and those found outside clusters
    and containing mostly negatives.  There would exist only one possible
    perfect clustering -- the one with a single, entirely homogeneous cluster C
    that covers all positives present in the data set. If one were to obtain
    such clustering, one could correctly label all positive samples in one step
    with the simple rule, *all positive samples belong to cluster C*. Under an
    imperfect clustering, on the other hand, the presence of the given sample in
    a cluster of size two or more implies the sample is only likely to be
    positive, with the confidence of the positive call monotonously increasing
    with the size of the cluster. Furthermore, under imperfect conditions, a
    possibility is introduced that a cluster could predominantly or even
    entirely consist of negatives, and one would have to perform additional work
    labeling samples. To minimize the amount of work performed, we would like
    the metric to penalize situations with many small clusters (even if they are
    perfectly homogeneous), with maximum penalty applied to the clustering where
    all clusters are of size one.

    The application that inspired the design of this metric was mining for
    positive spam examples in large data sets of short user-generated content.
    Given large enough data sets, spam content naturally forms clusters either
    because creative rewriting of every single individual spam message is too
    expensive for spammers to employ, or because, even if human or algorithmic
    rewriting is applied, one can still find features that link individual spam
    messages to their creator or to the product or service being promoted in the
    spam campaign. The finding was consistent with what is reported in
    literature [2]_.

    Given a clustering, we order the clusters from the largest one to the
    smallest one. We then plot a cumulative step function where the width of the
    bin under a given "step" is proportional to cluster size, and the height of
    the bin is proportional to the expected number of positive samples seen so
    far [3]_. After two-way normalization, a perfect clustering (i.e. where a
    single perfectly homogeneous cluster covers the entire set of positives)
    will have the AUL score of 1.0. A failure to cluster or a clustering based
    on a property completely unrelated with the ground truth labeling will have
    the AUL of 0.5. A perverse clustering, i.e. where predominantly negative
    samples fall into large clusters and positive ones remain unclustered or
    fall into smaller clusters will have the AUL somewhere between 0.0 and 0.5.

    A special treatment is necessary for cases where clusters are tied by size.
    If one were to treat tied clusters as a single group, one would obtain AUL
    of 1.0 when no clusters at all are present, which is against our desiderata.
    On the other hand, if one were to treat tied clusters entirely separately,
    one would obtain different results depending on the properties of the
    sorting algorithm, also an undesirable situation. Always placing "heavy"
    clusters (i.e. those containing more positives) towards the beginning or
    towards the end of the tied group will result in, respectively,
    overestimating or underestimating the true AUL. The solution here is to
    average the positive counts among all clusters in a tied group, and then
    walk through them one by one, with the stepwise cumulative function
    asymptotically approaching a diagonal from the group's bottom left corner to
    the top right one. This way, a complete absence of clustering (i.e. all
    clusters are of size one) will always result in AUL of 0.5, which is also
    the AUL for a random clustering uncorrelated with the ground truth labeling.

    .. [3] We take the expected number of positives and not the actual number
           seen so far as the vertical scale in order to penalize non-
           homogeneous clusters. Otherwise the y=1.0 ceiling would be reached
           early in the process even in very bad cases, for example when there
           is only one giant non- homogeneous cluster.

    References
    ----------

    .. [1] `Wikipedia entry for Gini coefficient of inequality
           <https://en.wikipedia.org/wiki/Gini_coefficient>`_

    .. [2] `Whissell, J. S., & Clarke, C. L. (2011, September). Clustering for
           semi-supervised spam filtering. In Proceedings of the 8th Annual
           Collaboration, Electronic messaging, Anti-Abuse and Spam Conference
           (pp.  125-134). ACM.
           <https://doi.org/10.1145/2030376.2030391>`_

    """

    def __init__(self, scores_true, scores_pred):
        # convert input to a series of tuples
        score_groups = izip(scores_pred, scores_true)

        # sort tuples by predicted score in descending order
        score_groups = sorted(score_groups, key=itemgetter(0), reverse=True)

        # group tuples by predicted score so as to handle ties correctly
        self.score_groups = list(aggregate_tuples(score_groups))

    @classmethod
    def from_clusters(cls, clusters):
        """Alternative interface for AUL metric
        """
        # take all non-empty clusters, score them by size and by number of
        # ground truth positives
        data = ((len(cluster), sum(bool(val) for val in cluster))
                for cluster in clusters if cluster)
        scores_pred, scores_true = zip(*data) or ([], [])
        return cls(scores_true, scores_pred)

    @classmethod
    def from_labels(cls, y_true, labels_pred):
        """Alternative interface for AUL metric
        """
        clusters = labels_to_clusters(y_true, labels_pred)
        return cls.from_clusters(clusters)

    def aul_score(self, plot=False):
        total_any = 0
        total_true = 0
        exp_vertical = 0
        aul = 0.0
        bin_height = 0.0

        if plot:
            xs, ys = [], []
            bin_right_edge = 0.0

        # second pass: iterate over each group of predicted scores of the same
        # size and calculate the AUL metric
        for pred_score, true_scores in self.score_groups:
            # number of clusters
            num_true_scores = len(true_scores)

            # sum total of positives in all clusters of given size
            group_height = sum(true_scores)

            total_true += group_height

            # cluster size x number of clusters of given size
            group_width = pred_score * num_true_scores

            total_any += group_width

            if pred_score > 1:
                # penalize non-homogeneous clusters simply by assuming that they
                # are homogeneous, in which case their expected vertical
                # contribution should be equal to their horizontal contribution.
                exp_vertical += group_width
            else:
                # clusters of size one are by definition homogeneous so their
                # expected vertical contribution equals sum total of any
                # remaining true positives.
                exp_vertical += group_height

            avg_true_score = group_height / float(num_true_scores)

            for _ in true_scores:

                bin_width = pred_score
                bin_height += avg_true_score
                aul += bin_height * bin_width

                if plot:
                    xs.append(bin_right_edge)
                    bin_right_edge += bin_width
                    xs.append(bin_right_edge)

                    ys.append(bin_height)
                    ys.append(bin_height)

        if total_true > total_any:
            warnings.warn(
                "Number of positives found (%d) exceeds total count of %d"
                % (total_true, total_any)
            )

        rect_area = exp_vertical * total_any

        # special case: since normalizing the AUL defines it as always smaller
        # than the bounding rectangle, when denominator in the expression below
        # is zero, the AUL score is also equal to zero.
        aul = 0.0 if rect_area == 0 else aul / rect_area

        if plot:
            xs = np.array(xs, dtype=float) / total_any
            ys = np.array(ys, dtype=float) / exp_vertical
            return aul, xs, ys
        else:
            return aul

    def plot(self):  # pragma: no cover
        """Shortcut to plot a lift chart (for aul_score debugging)
        """
        from matplotlib import pyplot as plt

        aul, xs, ys = self.aul_score(plot=True)
        fig, ax = plt.subplots()
        ax.plot(xs, ys, marker="o", linestyle='-')
        ax.fill([0.0] + list(xs) + [1.0], [0.0] + list(ys) + [0.0], 'b', alpha=0.2)
        ax.plot([0.0, 1.0], [0.0, 1.0], linestyle='--', color='grey')
        ax.plot([0.0, 1.0], [1.0, 1.0], linestyle='--', color='grey')
        ax.plot([1.0, 1.0], [0.0, 1.0], linestyle='--', color='grey')
        ax.set_xlim(xmin=0.0, xmax=1.1)
        ax.set_ylim(ymin=0.0, ymax=1.1)
        ax.set_xlabel("portion total")
        ax.set_ylabel("portion expected positive")
        ax.set_title("Lift Curve (AUL=%.3f)" % aul)
        fig.show()


def aul_score_from_clusters(clusters):
    return LiftCurve.from_clusters(clusters).aul_score()


def aul_score_from_labels(y_true, labels_pred):
    return LiftCurve.from_labels(y_true, labels_pred).aul_score()
