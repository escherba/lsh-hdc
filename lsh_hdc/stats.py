from collections import Counter, defaultdict
from functools import partial
from operator import itemgetter
from itertools import imap, chain
from math import log, fabs, copysign

__author__ = 'escherba'


def safe_div(num, denom):
    """Divide numbers, returning inf when dividing by zero"""
    try:
        return num / denom
    except ZeroDivisionError:
        return copysign(float('inf'), num)


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
        return 1.0 - safe_div(self.residual, sumsq(self.all))


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
        denominator = float(self.cluster_count) * \
            sum(imap(partial(entropy, self.post_count),
                     self.multiverse.values()))
        return 1.0 - safe_div(self.numerator, denominator)


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
        self.meta = {}

    def get_recall(self, pretty=False):
        """
        :rtype : float, str
        """
        result = safe_div(float(self.TP), (self.TP + self.FN))
        return '{:.1%}'.format(result) if pretty else result

    def get_precision(self, pretty=False):
        """
        :rtype : float, str
        """
        result = safe_div(float(self.TP), (self.TP + self.FP))
        return '{:.1%}'.format(result) if pretty else result

    def get_f1_score(self, pretty=False):
        """
        :rtype : float, str
        """
        recall = self.get_recall()
        precis = self.get_precision()
        result = safe_div(2.0 * recall * precis, (recall + precis))
        return '{:.1%}'.format(result) if pretty else result

    def __repr__(self):
        result = self.dict()
        return repr((result, {'meta': self.meta}))

    def dict(self):
        """
        :rtype : dict
        """
        result = dict(TP=self.TP, FP=self.FP, TN=self.TN, FN=self.FN)
        return result

    def add(self, ground_positive, predicted_positive):
        """
        :param ground_positive: Ground truth
        :type ground_positive: bool
        :param predicted_positive: Predicted result
        :type predicted_positive: bool
        """
        if predicted_positive:
            if ground_positive:
                self.TP += 1
            else:
                self.FP += 1
        else:
            if ground_positive:
                self.FN += 1
            else:
                self.TN += 1


def describe_clusters(clusters, pred, threshold=3):
    """
    Describe a list of clusters of labels with a predicate function that
    takes a label and returns ground truth result for that label

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
    num_clusters = 0
    for cluster in clusters:
        num_clusters += 1
        predicted_positive = len(cluster) >= threshold
        for label in cluster:
            c.add(pred(label), predicted_positive)
    c.meta['num_clusters'] = num_clusters
    return c


def auc(xs, ys, reorder=False):
    """ Compute area under curve using trapesoidal rule"""
    tuples = zip(xs, ys)
    assert len(tuples) > 1
    if reorder:
        tuples.sort()
    a = 0.0
    x0, y0 = tuples[0]
    for x1, y1 in tuples[1:]:
        a += (x1 - x0) * (y1 + y0)
        x0, y0 = x1, y1
    return a * 0.5


def roc_auc(fpr, tpr, reorder=False):
    """ Compute area under ROC curve """
    return auc(
        chain([0.0], fpr, [1.0]),
        chain([0.0], tpr, [1.0]),
        reorder=reorder)


class ROCSummarizer:
    """ROC curve summarizer"""
    def __init__(self):
        self.tps = []
        self.fps = []
        self.tns = []
        self.fns = []

    def add(self, tp, fp, tn, fn):
        """

        :param tp: true positives
        :type tp: int
        :param fp: false positives
        :type fp: int
        :param tn: true negatives
        :type tn: int
        :param fn: false negatives
        :type fn: int
        """
        self.tps.append(tp)
        self.fps.append(fp)
        self.tns.append(tn)
        self.fns.append(fn)

    @staticmethod
    def _div(data):
        """
        :return: a "safe" result of division of x by x + y
        :rtype : float
        """
        x, y = data
        return safe_div(float(x), x + y)

    def get_tprs(self):
        """
        :return: a list of true positive rate (recall) values
        :rtype : list
        """
        return map(self._div, zip(self.tps, self.fns))

    def get_fprs(self):
        """
        :return: a list of false positive rate values
        :rtype : list
        """
        return map(self._div, zip(self.fps, self.tns))

    def get_precisions(self):
        """
        :return: a list of precision values
        :rtype : list
        """
        return map(self._div, zip(self.tps, self.fps))

    def get_points(self):
        """
        :return: a list of tuples (x, y)
        :rtype : list
        """
        return sorted(zip(self.get_fprs(), self.get_tprs()))

    def get_axes(self):
        """
        :return: tuple of [x], [y] lists (for plotting)
        :rtype : tuple
        """
        points = self.get_points()
        return map(itemgetter(0), points), map(itemgetter(1), points)

    def get_axes_pct(self):
        """
        :return: tuple of [x], [y] lists (for plotting)
        :rtype : tuple
        """
        points = self.get_points()
        return [100.0 * p[0] for p in points], [100.0 * p[1] for p in points]

    def get_auc_score(self):
        """
        :return: Area-under-the-curve (AUC) statistic
        :rtype : float
        """
        return roc_auc(*self.get_axes())


def get_roc_summary(iter, get_level, ground_pos):
    d = defaultdict(StatResult)
    for item in iter:
        sr = d[get_level(item)]
        sr.add(ground_pos(item), True)

    ref_pos, ref_neg = 0, 0
    for sr in d.values():
        ref_pos += sr.TP + sr.FN
        ref_neg += sr.TN + sr.FP

    sorted_keys = sorted(d.keys(), reverse=True)
    meta = Counter()
    roc = ROCSummarizer()
    for nc in sorted_keys:
        summ = d[nc]
        meta.update(summ.dict())
        tp, fp = meta['TP'], meta['FP']
        tn, fn = ref_neg - fp, ref_pos - tp
        roc.add(tp, fp, tn, fn)
    return roc
