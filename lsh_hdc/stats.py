from collections import Counter
from functools import partial
from itertools import imap
from math import log, fabs

__author__ = 'escherba'


def median(xs):
    """
    :param xs: A list of numbers
    :type xs: list
    :return:  Median
    :rtype : float
    """
    xss = sorted(xs)
    length = len(xss)
    if not length % 2:
        return (xss[length / 2] + xss[length / 2 - 1]) / 2.0
    return xss[length / 2]


def entropy(N, n):
    """Calculate Shannon entropy given n, N

    :param N: sample count
    :type N: int
    :param n: number of bits
    :type n: int
    :return: (Information) entropy
    :rtype: float
    """
    n_ = float(n)
    if n_ > 0.0:
        ratio = n_ / float(N)
        return - ratio * log(ratio)
    else:
        return 0.0


def average(l):
    """Calculate arithmetic mean (average)

    :param l: a list of numbers
    :type l: list
    :returns: average
    :rtype: float
    """
    xs = list(l)
    return float(reduce(lambda x, y: x + y, xs)) / float(len(xs))


def sumsq(l):
    """Calculate sum of squares given a list

    :param l: a list of numbers
    :type l: list
    :returns: sum of squares
    :rtype: float
    """
    xs = list(l)
    avg = average(xs)
    return sum((el - avg) ** 2 for el in xs)


def weighted_median(values, weights):
    """Calculate a weighted median

    :param values: a vector of values
    :type values: list
    :param weights: a vector of weights
    :type weights: list
    :returns: value at index k s.t. the weights of all values v_i, i < k
              is < S/2 where S is the sum of all weights
    """
    sorted_v = sorted(zip(values, weights))
    if len(sorted_v) < 2:
        return values[0][0]
    k = 0
    w = sum(weights)
    w2 = w / 2
    for k, val in enumerate(sorted_v):
        w -= val[1]
        if w <= w2:
            break
    return sorted_v[k][0]


def mad(v):
    """Calculate median absolute deviation
    http://en.wikipedia.org/wiki/Median_absolute_deviation

    :param v: a list
    :type v: list
    """
    m = median(v)
    return median([fabs(x - m) for x in v])


class Summarizer(object):

    def add_object(self, *args, **kwargs):
        pass

    def get_summary(self):
        pass


class MADSummarizer(Summarizer):
    def __init__(self):
        self.weights = []
        self.mad_values = []

    def add_object(self, obj):
        """

        :param obj: a list of numbers
        :type obj: list
        """
        self.mad_values.append(mad(obj))
        self.weights.append(len(obj))

    def get_summary(self):
        """Calculate median absolute deviation ratio
        http://en.wikipedia.org/wiki/Median_absolute_deviation

        :rtype : float
        """
        return weighted_median(self.mad_values, self.weights)


class MADRatioSummarizer(Summarizer):
    def __init__(self):
        self.weights = []
        self.mad_values = []
        self.total_values = []

    def add_object(self, obj):
        """

        :param obj: a list of numbers
        :type obj: list
        """
        self.total_values.extend(obj)
        self.mad_values.append(mad(obj))
        self.weights.append(len(obj))

    def get_summary(self):
        """Calculate square of median absolute deviation ratio

        :rtype : float
        """
        return 1.0 - (weighted_median(self.mad_values, self.weights)
                      / mad(self.total_values)) ** 2


class VarianceSummarizer(Summarizer):
    def __init__(self):
        self.total_ss = 0.0
        self.N = 0

    def add_object(self, obj):
        """

        :param obj: a list of numbers
        :type obj: list
        """
        self.total_ss += sumsq(obj)
        self.N += len(obj)

    def get_summary(self):
        """Return (biased) estimator of weighted variance

        :return: weighted variance
        :rtype: float
        """
        return self.total_ss / float(self.N)


class ExplainedVarianceSummarizer(Summarizer):
    def __init__(self):
        self.residual = 0.0
        self.all = []

    def add_object(self, obj):
        """

        :param obj: a list of numbers
        :type obj: list
        """
        self.residual += sumsq(obj)
        self.all.extend(obj)

    def get_summary(self):
        """

        :return: Explained variance
        :rtype : float
        """
        try:
            result = 1.0 - self.residual / sumsq(self.all)
        except ZeroDivisionError:
            result = None
        return result


class UncertaintySummarizer(Summarizer):
    def __init__(self):
        self.multiverse = Counter()
        self.numerator = 0.0
        self.cluster_count = 0
        self.post_count = 0

    def add_object(self, obj, cluster_size):
        """

        :param obj: a mapping from keys to counts
        :type obj: collections.Counter
        """
        self.numerator += \
            sum(imap(partial(entropy, cluster_size), obj.values()))
        self.multiverse.update(obj)
        self.cluster_count += 1
        self.post_count += cluster_size

    def get_summary(self):
        """

        :returns: Theil index of uncertainty
        :rtype : float
        """
        try:
            denominator = float(self.cluster_count) * \
                sum(imap(partial(entropy, self.post_count),
                         self.multiverse.values()))
            result = 1.0 - self.numerator / denominator
        except ZeroDivisionError:
            result = None
        return result


class FeatureClusterSummarizer:

    def __init__(self):
        self.label2features = dict()

    def add_features(self, label, features):
        if label in self.label2features:
            raise RuntimeError("Duplicate label")
        self.label2features[label] = features

    def summarize_clusters(self, clusters):
        s = UncertaintySummarizer()
        for cluster in clusters:
            universe = Counter()
            for label in cluster:
                features = self.label2features[label]
                universe.update(features)
            s.add_object(universe, len(cluster))
        return s.get_summary()


class StatResult:
    def __init__(self):
        self.TP = 0
        self.FP = 0
        self.FN = 0
        self.TN = 0

    def get_recall(self, pretty=False):
        """
        :rtype : float, str
        """
        try:
            result = float(self.TP) / (self.TP + self.FN)
        except ZeroDivisionError:
            result = float('nan')
        return '{:.1%}'.format(result) if pretty else result

    def get_precision(self, pretty=False):
        """
        :rtype : float, str
        """
        try:
            result = float(self.TP) / (self.TP + self.FP)
        except ZeroDivisionError:
            result = float('nan')
        return '{:.1%}'.format(result) if pretty else result

    def __repr__(self):
        return str(dict(TP=self.TP, FP=self.FP, TN=self.TN, FN=self.FN))

    def dict(self):
        """
        :rtype : dict
        """
        return dict(TP=self.TP, FP=self.FP, TN=self.TN, FN=self.FN)


def get_stats(clusters, pred, threshold=3):
    """
    :param clusters: A list of clusters (list of lists)
    :type clusters: list
    :param pred: A predicate that acts on a label and returns
                 True or False
    :type pred: function
    :param threshold: Threshold at which we call a cluster
    :type threshold: 3
    :return: an instance of StatResult
    :rtype: StatResult
    """
    c = StatResult()
    for i, cluster in enumerate(clusters):
        if len(cluster) >= threshold:
            for label in cluster:
                if pred(label):
                    c.TP += 1  # TRUE POSITIVES
                else:
                    c.FP += 1  # FALSE POSITIVES
        else:
            for label in cluster:
                if pred(label):
                    c.FN += 1  # FALSE NEGATIVES
                else:
                    c.TN += 1  # TRUE NEGATIVES
    return c