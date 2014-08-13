from collections import Counter, defaultdict
from functools import partial
from operator import itemgetter
from itertools import imap, izip, chain
from math import log, fabs, copysign
from urllib import urlencode
import locale

__author__ = 'escherba'

locale.setlocale(locale.LC_ALL, 'en_US')


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
    """Calculate Shannon entropy given N, n

    :param N: sample count
    :type N: int
    :param n: number of bits
    :type n: int
    :return: (Information) entropy
    :rtype: float
    """
    assert n <= N
    if n > 0:
        ratio = float(n) / float(N)
        return - ratio * log(ratio)
    elif n == 0:
        return 0.0
    else:
        return float('nan')


def average(l):
    """Calculate arithmetic mean (average)

    :param l: a list of numbers
    :type l: list
    :returns: average
    :rtype: float
    """
    xs = list(l)
    return safe_div(float(reduce(lambda x, y: x + y, xs)), float(len(xs)))


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
        return 1.0 - safe_div(weighted_median(self.mad_values, self.weights),
                              mad(self.total_values)) ** 2


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
        return safe_div(self.total_ss, float(self.N))


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


def counts_entropy(counts):
    """Find entropy of a list of counts

    Assumes every entry in the list corresponds to a different class

    :param counts: list of counts, e.g. [10, 2, 3, 1]
    :type counts: list
    :return: entropy
    :rtype: float
    """
    fun = partial(entropy, sum(counts))
    return sum(fun(val) for val in counts)


def uncertainty_score(labels_true, labels_pred):
    """Uncertainty coefficient (Theil's U)

    :param labels_true: a list of true labels
    :type labels_true: collections.Iterable
    :param labels_pred: a list of predicted labels
    :type labels_pred: collections.Iterable
    :return: uncertainty coefficient
    :rtype: float

    This function is meant to be a plug-in replacement for SciKit-Learn's
    `normalized_mutual_info_score'. For more info, see:
    http://en.wikipedia.org/wiki/Uncertainty_coefficient

    This is an asymmetric coefficient. It is zero for non- informative cases
    where only one cluster is predicted:

    >>> from sklearn.metrics import normalized_mutual_info_score as NMI_score
    >>> labels_true = [0,1,1]
    >>> labels_pred = [1,1,1]
    >>> NMI_score(labels_true, labels_pred)
    5.5511151231257827e-07
    >>> uncertainty_score(labels_true, labels_pred)
    0.0

    This means that when clusters are perfectly homogeneous, regardless of the
    number of clusters, the index will be 1.0. It gives us a clean metric for
    cases when cluster number is fixed and all items are clustered. Otherwise
    using this metric will result in a preference for very small clusters.

    For an example given in
    http://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-clustering-1.html

    >>> labels_true = [0,0,0,0,0,1,0,1,1,1,2,1,0,0,2,2,2]
    >>> labels_pred = [0,0,0,0,0,0,1,1,1,1,1,1,2,2,2,2,2]
    >>> NMI_score(labels_true, labels_pred)
    0.36462479619424287
    >>> uncertainty_score(labels_true, labels_pred)
    0.3709496570219556

    """

    c = defaultdict(Counter)

    for p, t in izip(labels_pred, labels_true):
        c[p][t] += 1

    X_counter = Counter()
    XY_entropy = 0.0
    num_X_classes = 0

    for XY_counter in c.itervalues():
        X_counter.update(XY_counter)
        XY_entropy += counts_entropy(XY_counter.values())
        num_X_classes += 1

    X_entropy = counts_entropy(X_counter.values())
    XY_information = safe_div(XY_entropy, float(num_X_classes))
    return 1.0 - safe_div(XY_information, X_entropy)


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


class FeatureClusterSummarizer(object):

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


class StatResult(object):
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


def mplot_roc_curves(mplt, rocs, names, curve='roc', pct=False, auc=False):
    """Plot ROC curve with MATPLOTLIB
    :param mplt: matplotlib.pyplot module
    :type plt: module
    :param rocs: a list of ROCSummarizer instances
    :type rocs: list
    :param names: a list of labels for above instances
    :type names: list
    :param pct: whether to use percentage
    :type pct: bool
    :param auc: whether to show area under the curve
    :type auc: bool
    """

    transform = 'pct' if pct else None

    for i, (name, roc) in enumerate(izip(names, rocs)):
        if auc:
            auc_str = "AUC: {:.3f}".format(roc.get_auc_score())
            name = "{} ({})".format(i, auc_str) \
                if name is None \
                else "{} ({})".format(name, auc_str)
        else:
            if name is None:
                name = str(i)

        mplt.plot(*roc.get_axes(curve=curve, transform=transform), label=name)

    if pct:
        suffix = ' (%)'
        mult = 100.0
    else:
        suffix = ''
        mult = 1.0

    mplt.xlabel('False Positive Rate' + suffix)
    if curve == 'roc':
        mplt.title('ROC Curve')
        mplt.ylabel('Recall' + suffix)
        mplt.ylim([0.0, 1.0 * mult])
        mplt.xlim([0.0, 0.1 * mult])
        mplt.legend(loc='lower right')
    else:
        mplt.title('DET Curve')
        mplt.ylabel('False Negative Rate' + suffix)
        mplt.legend(loc='lower left')
    mplt.show()


def safe_log(d):
    if d > 0.0:
        return log(d)
    elif d == 0.0:
        return float('-inf')
    else:
        return float('nan')


def safe_logit(p):
    """Return logit transform (don't throw exceptions)"""
    return safe_log(safe_div(p, 1.0 - p))


# from scipy.special import erfinv
# def probit(p):
#     return sqrt(2.0) * erfinv(2.0 * p - 1.0)


class ROCSummarizer(object):
    """ROC curve summarizer

    Can return plot-ready series for ROC (Receiver Operating Characterstic)
    and DET (Detection Error Tradeoff) curves

    """

    VALID_CURVES = {'roc', 'det'}

    TRANSFORMS = {
        'logit': safe_logit,
        'log': safe_log,
        'pct': lambda x: 100.0 * x,
        'id': lambda x: x,
        # probit is harder to compute and practically the same as logit
    }

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

    def get_fnrs(self):
        """
        :return: a list of false negative rate values
        :rtype : list
        """
        return map(self._div, zip(self.fns, self.tps))

    def get_precisions(self):
        """
        :return: a list of precision values
        :rtype : list
        """
        return map(self._div, zip(self.tps, self.fps))

    def get_transform(self, transform='id'):
        try:
            return partial(map, self.TRANSFORMS[transform])
        except KeyError:
            raise ValueError("`transform' must be one of %s" %
                             self.TRANSFORMS.keys())

    def get_points(self, curve='roc', transform=None):
        """
        :return: a list of tuples (x, y)
        :rtype : list
        """

        if curve == 'roc':
            if transform is None:
                transform = 'pct'
            ft = self.get_transform(transform)
            pts = ft(self.get_fprs()), ft(self.get_tprs())
        elif curve == 'det':
            if transform is None:
                transform = 'logit'
            ft = self.get_transform(transform)
            pts = ft(self.get_fprs()), ft(self.get_fnrs())
        else:
            raise ValueError("`curve' must be one of %s" % self.VALID_CURVES)
        return sorted(zip(*pts))

    def get_axes(self, curve='roc', transform=None):
        """
        :return: tuple of [x], [y] lists (for plotting)
        :rtype : tuple
        """

        # set default transform
        if transform is None:
            if curve == 'roc':
                transform = 'id'
            elif curve == 'det':
                transform = 'logit'

        return zip(*self.get_points(curve=curve, transform=transform))

    def get_auc_score(self):
        """
        :return: Area-under-the-curve (AUC) statistic
        :rtype : float
        """
        return roc_auc(*self.get_axes())


def get_roc(d):
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


def get_roc_summaries(iterable, level_getters, ground_pos):
    ds = [defaultdict(StatResult) for x in level_getters]
    for item in iterable:
        ground_truth = ground_pos(item)
        for d, get_level in izip(ds, level_getters):
            d[get_level(item)].add(ground_truth, True)
    return map(get_roc, ds)


def int2str(i):
    return locale.format("%d", i, grouping=True)


class VennDiagram(object):

    PREFIX = '_'

    def __init__(self, pandas):
        self._pd = pandas
        self.venn = Counter()

    def add_fact(self, fact):
        """Add a fact (a list of key-value pairs)"""
        tuples = sorted(fact.iteritems(), key=itemgetter(0))
        self.venn[tuple(tuples)] += 1

    def get_dataframe(self, prefix=PREFIX):
        dicts = []
        for tuples, val in self.venn.iteritems():
            keys = [(prefix + k, v) for k, v in tuples]
            dicts.append(dict(chain(keys, [("count", val)])))

        return self._pd.DataFrame.from_dict(dicts)

    @staticmethod
    def get_google_venn2(df, columns):
        cols = columns[:2]
        A, B = cols
        counts = map(lambda x: x if isinstance(x, int) else
                     df.__getitem__(x)['count'].sum(),
                     [df[A], df[B], 0, df[A] & df[B], 0, 0, 0])
        return cols, counts

    @staticmethod
    def get_google_venn3(df, columns):
        cols = columns[:3]
        A, B, C = cols
        counts = map(lambda x: df.__getitem__(x)['count'].sum(),
                     [df[A], df[B], df[C], df[A] & df[B], df[A] & df[C],
                      df[B] & df[C], df[A] & df[B] & df[C]])
        return cols, counts

    def get_googlechart(self, df, columns, w=400, h=400):

        """return URL for Google Chart API
        :param df: data frame
        :type df: pandas.DataFrame
        :param columns: list of columns sorted by frequency
        :type columns: list
        """

        num_columns = len(columns)
        if num_columns >= 3:
            cols, counts = self.get_google_venn3(df, columns)
        elif num_columns == 2:
            cols, counts = self.get_google_venn2(df, columns)
        else:
            raise ValueError("Do not know how to create a Venn diagram"
                             "from {} sets".format(num_columns))

        prefix_len = len(self.PREFIX)
        lbls = '|'.join(col[prefix_len:] + ': ' + int2str(c)
                        for col, c in izip(cols, counts))

        mult = safe_div(100.0, float(max(counts)))
        norm = [round(float(c) * mult, 1) for c in counts]
        stri = ','.join(map(str, norm))
        return "https://chart.googleapis.com/chart?" + urlencode(dict(
            cht='v',
            chs='%dx%d' % (w, h),
            chd='t:' + stri,
            chdl=lbls,
            # chco='FF6342,ADDE63,63C6DE'
        ))

    def show(self, circles=None):

        df = self.get_dataframe(prefix=self.PREFIX)
        if circles is None:
            circle_sizes = [(c, df[df[c]]['count'].sum())
                            for c in df.columns[:-1]]
            circles = map(itemgetter(0), sorted(circle_sizes,
                                                key=itemgetter(1),
                                                reverse=True))
        else:
            circles = [self.PREFIX + lbl for lbl in circles]

        print
        print "Venn Diagram"
        print df.groupby(circles).sum()
        print
        print 'Link: ' + self.get_googlechart(df, circles)
        print

if __name__ == "__main__":
    import doctest
    doctest.testmod()
