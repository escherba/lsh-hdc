from math import log as log_nat
from collections import defaultdict, Counter, Mapping
from itertools import izip


def cond_entropy(counts, N):
    """Returns conditional entropy

    The parameter `counts` is expected to be an list or tuple-like iterable.
    For convenience, it can also be a dict/mapping type, in which case
    its values will be used to calculate entropy

    TODO: Cythonize this using NumPy's buffer interface for arrays
    """
    if isinstance(counts, Mapping):
        counts = counts.values()
    log_row_total = log_nat(sum(counts))
    # to avoid loss of precision, calculate 'log(a/b)' as 'log(a) - loh(b)'
    return -sum(c * (log_nat(c) - log_row_total) for c in counts if c != 0) / N


def harmonic_mean(x, y):
    """Harmonic mean of two numbers. Returns a float
    """
    # Since harmonic mean converges to arithmetic mean as x approaches y,
    # return the latter when x == y, which is numerically safer.
    return (x + y) / 2.0 if x == y else (2.0 * x * y) / (x + y)


def clustering_metrics(labels_true, labels_pred):
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

    >>> clustering_metrics([0, 0, 1, 1], [1, 1, 0, 0])
    (1.0, 1.0, 1.0)
    """
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
    H_C = cond_entropy(class_total, N)
    H_K = cond_entropy(kluster_total, N)
    H_CK = sum(cond_entropy(x, N) for x in klusters.itervalues())
    H_KC = sum(cond_entropy(x, N) for x in classes.itervalues())
    # The '<=' comparisons below both prevent division by zero errors
    # and guarantee that the scores are always non-negative.
    homogeneity = 0.0 if H_C <= H_CK else 1.0 - H_CK / H_C
    completeness = 0.0 if H_K <= H_KC else 1.0 - H_KC / H_K
    nmi_score = harmonic_mean(homogeneity, completeness)
    return homogeneity, completeness, nmi_score
