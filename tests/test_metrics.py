import warnings
import numpy as np
from numpy.testing import assert_array_almost_equal
from math import sqrt
from itertools import izip
from nose.tools import assert_almost_equal, assert_true, assert_equal
from lsh_hdc.metrics import adjusted_rand_score, \
    homogeneity_completeness_v_measure, centropy, \
    jaccard_similarity, ClusteringMetrics, \
    ConfusionMatrix2, geometric_mean, harmonic_mean, _div, cohen_kappa, \
    matthews_corr, expected_mutual_information, mutual_info_score, \
    adjusted_mutual_info_score


def check_with_nans(num1, num2, places=None, msg=None, delta=None, ensure_nans=True):
    nancheck_msg = "NaN check failed for '%s'" % msg
    if np.isnan(num1):
        if ensure_nans:
            assert_true(np.isnan(num2), msg=nancheck_msg)
        elif not np.isnan(num2):
            warnings.warn(nancheck_msg)
    elif np.isnan(num2):
        if ensure_nans:
            assert_true(np.isnan(num1), msg=nancheck_msg)
        elif not np.isnan(num1):
            warnings.warn(nancheck_msg)
    else:
        assert_almost_equal(num1, num2, places=places, msg=msg, delta=delta)


def _kappa(a, c, d, b):
    """An alternative implementation of Cohen's kappa (for testing)
    """
    n = a + b + c + d
    p1 = a + b
    p2 = a + c
    q1 = c + d
    q2 = b + d
    if a == n or b == n or c == n or d == n:
        # only one cell is non-zero
        return np.nan
    elif p1 == 0 or p2 == 0 or q1 == 0 or q2 == 0:
        # one row or column is zero, another non-zero
        return 0.0
    else:
        # no more than one cell is zero
        po = a + d
        pe = (p2 * p1 + q2 * q1) / float(n)
        return _div(po - pe, n - pe)


def _entropy_metrics(cm):
    """Given a ClusteringMetrics object, calculate three entropy-based metrics

    (Alternative implementation for testing)
    """
    H_C = centropy(cm.row_totals)
    H_K = centropy(cm.col_totals)
    H_CK = sum(centropy(col) for col in cm.iter_cols())
    H_KC = sum(centropy(row) for row in cm.iter_rows())
    # The '<=' comparisons below both prevent division by zero errors
    # and ensure that the scores are non-negative.
    homogeneity = 0.0 if H_C <= H_CK else (H_C - H_CK) / H_C
    completeness = 0.0 if H_K <= H_KC else (H_K - H_KC) / H_K
    nmi_score = harmonic_mean(homogeneity, completeness)
    return homogeneity, completeness, nmi_score


def _talburt_wang_index(labels_true, labels_pred):
    """Alt. implementation of Talburt-Wang index for testing
    """
    V = set()
    A = set()
    B = set()
    for pair in izip(labels_true, labels_pred):
        V.add(pair)
        A.add(pair[0])
        B.add(pair[1])
    prod = len(A) * len(B)
    return np.nan if prod == 0 else sqrt(prod) / len(V)


def uniform_labelings_scores(score_func, n_samples, k_range, n_runs=10,
                             seed=42):
    # Compute score for random uniform cluster labelings
    random_labels = np.random.RandomState(seed).random_integers
    scores = np.zeros((len(k_range), n_runs))
    for i, k in enumerate(k_range):
        for j in range(n_runs):
            labels_a = random_labels(low=0, high=k - 1, size=n_samples)
            labels_b = random_labels(low=0, high=k - 1, size=n_samples)
            scores[i, j] = score_func(labels_a, labels_b)
    return scores


def test_adjusted_mutual_info_score():
    # Compute the Adjusted Mutual Information and test against known values
    labels_a = np.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3])
    labels_b = np.array([1, 1, 1, 1, 2, 1, 2, 2, 2, 2, 3, 1, 3, 3, 3, 2, 2])

    # Mutual information
    mi_1 = mutual_info_score(labels_a, labels_b)
    assert_almost_equal(mi_1, 0.41022, 5)
    mi_2 = mutual_info_score(labels_b, labels_a)
    assert_almost_equal(mi_2, 0.41022, 5)

    # Expected mutual information
    cm = ClusteringMetrics.from_labels(labels_a, labels_b)
    row_totals = np.fromiter(cm.iter_row_totals(), dtype=np.int32)
    col_totals = np.fromiter(cm.iter_col_totals(), dtype=np.int32)
    emi_1 = expected_mutual_information(row_totals, col_totals)
    assert_almost_equal(emi_1 / cm.grand_total, 0.15042, 5)
    emi_2 = expected_mutual_information(col_totals, row_totals)
    assert_almost_equal(emi_2 / cm.grand_total, 0.15042, 5)

    # Adjusted mutual information (1)
    ami_1 = adjusted_mutual_info_score(labels_a, labels_b)
    assert_almost_equal(ami_1, 0.27502, 5)
    ami_2 = adjusted_mutual_info_score(labels_a, labels_b)
    assert_almost_equal(ami_2, 0.27502, 5)

    # Adjusted mutual information (2)
    ami_1 = adjusted_mutual_info_score([1, 1, 2, 2], [2, 2, 3, 3])
    assert_equal(ami_1, 1.0)
    ami_2 = adjusted_mutual_info_score([2, 2, 3, 3], [1, 1, 2, 2])
    assert_equal(ami_2, 1.0)

    # Test AMI with a very large array
    a110 = np.array([list(labels_a) * 110]).flatten()
    b110 = np.array([list(labels_b) * 110]).flatten()
    ami = adjusted_mutual_info_score(a110, b110)
    assert_almost_equal(ami, 0.37, 2)  # not accurate to more than 2 places


def test_jaccard_nan():
    """Returns NaN for empty set
    """
    sim = jaccard_similarity([], [])
    assert_true(np.isnan(sim))


def test_entropy_of_counts_zero():
    """Returns zero for empty set
    """
    val = centropy([])
    assert_almost_equal(val, 0.0000, 4)


def test_perfectly_good_clustering():
    """Perfect separation
    """
    h, c, v = homogeneity_completeness_v_measure([0, 0, 1, 1], [1, 1, 0, 0])
    assert_almost_equal(h, 1.00, 2)
    assert_almost_equal(c, 1.00, 2)
    assert_almost_equal(v, 1.00, 2)


def test_perfectly_bad_clustering():
    """No separation
    """
    h, c, v = homogeneity_completeness_v_measure([0, 0, 1, 1], [1, 1, 1, 1])
    assert_almost_equal(h, 0.00, 2)
    assert_almost_equal(c, 1.00, 2)
    assert_almost_equal(v, 0.00, 2)


def test_homogeneous_but_not_complete_labeling():
    """homogeneous but not complete clustering
    """
    h, c, v = homogeneity_completeness_v_measure(
        [0, 0, 0, 1, 1, 1],
        [0, 0, 0, 1, 2, 2])
    assert_almost_equal(h, 1.00, 2)
    assert_almost_equal(c, 0.69, 2)
    assert_almost_equal(v, 0.81, 2)


def test_complete_but_not_homogeneous_labeling():
    """complete but not homogeneous clustering
    """
    h, c, v = homogeneity_completeness_v_measure(
        [0, 0, 1, 1, 2, 2],
        [0, 0, 1, 1, 1, 1])
    assert_almost_equal(h, 0.58, 2)
    assert_almost_equal(c, 1.00, 2)
    assert_almost_equal(v, 0.73, 2)


def test_not_complete_and_not_homogeneous_labeling():
    """neither complete nor homogeneous but not so bad either
    """
    h, c, v = homogeneity_completeness_v_measure(
        [0, 0, 0, 1, 1, 1],
        [0, 1, 0, 1, 2, 2])
    assert_almost_equal(h, 0.67, 2)
    assert_almost_equal(c, 0.42, 2)
    assert_almost_equal(v, 0.52, 2)


def test_non_consecutive_labels_std():
    """regression tests for labels with gaps
    """
    h, c, v = homogeneity_completeness_v_measure(
        [0, 0, 0, 2, 2, 2],
        [0, 1, 0, 1, 2, 2])
    assert_almost_equal(h, 0.67, 2)
    assert_almost_equal(c, 0.42, 2)
    assert_almost_equal(v, 0.52, 2)

    h, c, v = homogeneity_completeness_v_measure(
        [0, 0, 0, 1, 1, 1],
        [0, 4, 0, 4, 2, 2])
    assert_almost_equal(h, 0.67, 2)
    assert_almost_equal(c, 0.42, 2)
    assert_almost_equal(v, 0.52, 2)


def test_ari_nan():
    """Returns NaN for empty lists
    """
    ari = adjusted_rand_score([], [])
    assert_true(np.isnan(ari))


def test_non_consecutive_labels_ari():
    """regression tests for labels with gaps
    """
    ari_1 = adjusted_rand_score([0, 0, 0, 1, 1, 1], [0, 1, 0, 1, 2, 2])
    ari_2 = adjusted_rand_score([0, 0, 0, 1, 1, 1], [0, 4, 0, 4, 2, 2])
    assert_almost_equal(ari_1, 0.24, 2)
    assert_almost_equal(ari_2, 0.24, 2)


def test_IR_example():
    """Test example from IR book by Manning et al.

    The example gives 3 clusters and 17 points total. It is described on
    http://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-clustering-1.html
    """
    ltrue = (0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2)
    lpred = (0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 2, 2, 1, 2, 2, 2)
    cm = ClusteringMetrics.from_labels(ltrue, lpred)

    # test centropy metrics
    h, c, v = cm.entropy_metrics()
    assert_almost_equal(h, 0.371468, 6)
    assert_almost_equal(c, 0.357908, 6)
    assert_almost_equal(v, 0.364562, 6)

    assert_almost_equal(cm.vi_distance(),     1.366,    3)
    assert_almost_equal(cm.chisq_score(),     9.017647, 6)
    assert_almost_equal(cm.g_score(),        13.325845, 6)

    assert_almost_equal(cm.adjusted_jaccard_coeff(),       0.217138, 6)
    assert_almost_equal(cm.adjusted_sokal_sneath_coeff(),  0.128675, 6)

    # test metrics that are based on pairwise co-association matrix
    coassoc = cm.coassoc_

    assert_almost_equal(coassoc.chisq_score(),         8.063241, 6)
    assert_almost_equal(coassoc.g_score(),             7.804221, 6)

    assert_almost_equal(coassoc.jaccard_coeff(),       0.312500, 6)
    assert_almost_equal(coassoc.ochiai_coeff(),        0.476731, 6)
    assert_almost_equal(coassoc.dice_coeff(),          0.476190, 6)
    assert_almost_equal(coassoc.sokal_sneath_coeff(),  0.185185, 6)

    assert_almost_equal(coassoc.kappa(),               0.242915, 6)
    assert_almost_equal(coassoc.rand_index(),          0.676471, 6)
    assert_almost_equal(coassoc.precision(),           0.500000, 6)
    assert_almost_equal(coassoc.recall(),              0.454545, 6)

    exp_tw = _talburt_wang_index(ltrue, lpred)
    act_tw = cm.talburt_wang_index()
    assert_almost_equal(exp_tw, act_tw, 6)


def test_adjustment_for_chance():
    """Check that adjusted scores are almost zero on random labels
    """
    n_clusters_range = [2, 10, 50, 90]
    n_samples = 100
    n_runs = 10

    scores = uniform_labelings_scores(
        adjusted_rand_score, n_samples, n_clusters_range, n_runs)

    max_abs_scores = np.abs(scores).max(axis=1)
    assert_array_almost_equal(max_abs_scores, [0.02, 0.03, 0.03, 0.02], 2)


def test_twoway_confusion_1():
    """Finley's tornado data
    http://www.cawcr.gov.au/projects/verification/Finley/Finley_Tornados.html
    """
    cm = ConfusionMatrix2.from_ccw(28, 72, 2680, 23)

    assert_almost_equal(cm.g_score(),       126.1, 1)
    assert_almost_equal(cm.chisq_score(),   397.9, 1)

    assert_almost_equal(cm.mi_corr(),       0.429, 3)
    assert_almost_equal(cm.mi_corr1(),      0.497, 3)
    assert_almost_equal(cm.mi_corr0(),      0.382, 3)

    assert_almost_equal(cm.matthews_corr(), 0.377, 3)
    assert_almost_equal(cm.informedness(),  0.523, 3)
    assert_almost_equal(cm.markedness(),    0.271, 3)

    assert_almost_equal(cm.kappa(),         0.355, 3)
    assert_almost_equal(cm.kappa1(),        0.532, 3)
    assert_almost_equal(cm.kappa0(),        0.267, 3)


def test_twoway_confusion_2():
    """Finley's tornado data (listed in Goodman and Kruskal)
    """
    cm = ConfusionMatrix2.from_ccw(11, 14, 906, 3)

    assert_almost_equal(cm.g_score(),       70.83, 2)
    assert_almost_equal(cm.chisq_score(),   314.3, 1)

    assert_almost_equal(cm.mi_corr(),       0.614, 3)
    assert_almost_equal(cm.mi_corr1(),      0.698, 3)
    assert_almost_equal(cm.mi_corr0(),      0.555, 3)

    assert_almost_equal(cm.matthews_corr(), 0.580, 3)
    assert_almost_equal(cm.informedness(),  0.770, 3)
    assert_almost_equal(cm.markedness(),    0.437, 3)

    assert_almost_equal(cm.kappa(),         0.556, 3)
    assert_almost_equal(cm.kappa1(),        0.780, 3)
    assert_almost_equal(cm.kappa0(),        0.431, 3)


def test_negative_correlation():
    """Some metrics should have negative sign
    """
    cm = ConfusionMatrix2.from_ccw(10, 120, 8, 300)
    assert_almost_equal(cm.g_score(),        384.52, 2)
    assert_almost_equal(cm.chisq_score(),    355.70, 2)

    assert_almost_equal(cm.mi_corr(),       -0.8510, 4)
    assert_almost_equal(cm.mi_corr1(),      -0.8524, 4)
    assert_almost_equal(cm.mi_corr0(),      -0.8496, 4)
    assert_almost_equal(cm.matthews_corr(), -0.9012, 4)
    assert_almost_equal(cm.informedness(),  -0.9052, 4)
    assert_almost_equal(cm.markedness(),    -0.8971, 4)
    assert_almost_equal(cm.kappa(),         -0.6407, 4)
    inform, marked = cm.informedness(), cm.markedness()
    expected_matt = geometric_mean(inform, marked)
    assert_almost_equal(expected_matt, cm.matthews_corr(), 6)


def test_twoway_confusion_phi():
    cm = ConfusionMatrix2.from_ccw(116, 21, 18, 21)
    assert_almost_equal(cm.matthews_corr(), 0.31, 2)
    assert_almost_equal(cm.yule_q(), 0.6512, 4)
    assert_almost_equal(cm.DOR(),    4.7347, 4)

    cm = ConfusionMatrix2.from_ccw(35, 60, 41, 9)
    assert_almost_equal(cm.chisq_score(), 5.50, 2)


def test_0000():
    m = (0, 0, 0, 0)
    cm = ConfusionMatrix2.from_ccw(*m)
    assert_almost_equal(cm.chisq_score(), 0.0, 4)
    assert_almost_equal(cm.g_score(), 0.0, 4)
    assert_true(np.isnan(cm.matthews_corr()))
    assert_true(np.isnan(cm.mp_corr()))
    assert_true(np.isnan(cm.kappa()))

    assert_true(np.isnan(cm.loevinger_coeff()))
    assert_true(np.isnan(cm.yule_q()))
    assert_true(np.isnan(cm.yule_y()))

    assert_true(np.isnan(_kappa(*m)))


def test_1000():
    m = (1, 0, 0, 0)
    cm = ConfusionMatrix2.from_ccw(*m)
    assert_almost_equal(cm.chisq_score(), 0.0, 4)
    assert_almost_equal(cm.g_score(), 0.0, 4)

    h, c, v = cm.entropy_metrics()
    assert_almost_equal(h, 1.0, 4)
    assert_almost_equal(c, 1.0, 4)
    assert_almost_equal(v, 1.0, 4)

    assert_true(np.isnan(cm.matthews_corr()))
    assert_true(np.isnan(cm.mp_corr()))
    assert_true(np.isnan(cm.kappa()))

    assert_true(np.isnan(cm.loevinger_coeff()))
    assert_true(np.isnan(cm.yule_q()))
    assert_true(np.isnan(cm.yule_y()))

    assert_true(np.isnan(_kappa(*m)))


def test_0100():
    m = (0, 1, 0, 0)
    cm = ConfusionMatrix2.from_ccw(*m)
    assert_almost_equal(cm.chisq_score(), 0.0, 4)
    assert_almost_equal(cm.g_score(), 0.0, 4)

    h, c, v = cm.entropy_metrics()
    assert_almost_equal(h, 1.0, 4)
    assert_almost_equal(c, 1.0, 4)
    assert_almost_equal(v, 1.0, 4)

    assert_true(np.isnan(cm.matthews_corr()))
    assert_true(np.isnan(cm.mp_corr()))
    assert_true(np.isnan(cm.kappa()))

    assert_true(np.isnan(cm.loevinger_coeff()))
    assert_true(np.isnan(cm.yule_q()))
    assert_true(np.isnan(cm.yule_y()))

    assert_true(np.isnan(_kappa(*m)))


def test_0010():
    m = (0, 0, 1, 0)
    cm = ConfusionMatrix2.from_ccw(*m)
    assert_almost_equal(cm.chisq_score(), 0.0, 4)
    assert_almost_equal(cm.g_score(), 0.0, 4)

    h, c, v = cm.entropy_metrics()
    assert_almost_equal(h, 1.0, 4)
    assert_almost_equal(c, 1.0, 4)
    assert_almost_equal(v, 1.0, 4)

    assert_true(np.isnan(cm.matthews_corr()))
    assert_true(np.isnan(cm.mp_corr()))
    assert_true(np.isnan(cm.kappa()))

    assert_true(np.isnan(cm.loevinger_coeff()))
    assert_true(np.isnan(cm.yule_q()))
    assert_true(np.isnan(cm.yule_y()))

    assert_true(np.isnan(_kappa(*m)))


def test_0001():
    m = (0, 0, 0, 1)
    cm = ConfusionMatrix2.from_ccw(*m)
    assert_almost_equal(cm.chisq_score(), 0.0, 4)
    assert_almost_equal(cm.g_score(), 0.0, 4)

    h, c, v = cm.entropy_metrics()
    assert_almost_equal(h, 1.0, 4)
    assert_almost_equal(c, 1.0, 4)
    assert_almost_equal(v, 1.0, 4)

    assert_true(np.isnan(cm.matthews_corr()))
    assert_true(np.isnan(cm.mp_corr()))
    assert_true(np.isnan(cm.kappa()))

    assert_true(np.isnan(cm.loevinger_coeff()))
    assert_true(np.isnan(cm.yule_q()))
    assert_true(np.isnan(cm.yule_y()))

    assert_true(np.isnan(_kappa(*m)))


def test_1010():
    m = (1, 0, 1, 0)
    cm = ConfusionMatrix2.from_ccw(*m)
    assert_almost_equal(cm.chisq_score(), 2.0, 4)
    assert_almost_equal(cm.g_score(), 2.7726, 4)
    assert_almost_equal(cm.matthews_corr(), 1.0, 4)
    assert_almost_equal(cm.mp_corr(), 1.0, 4)
    assert_almost_equal(cm.kappa(), 1.0, 4)

    assert_almost_equal(cm.loevinger_coeff(), 1.0, 4)
    assert_almost_equal(cm.yule_q(), 1.0, 4)
    assert_almost_equal(cm.yule_y(), 1.0, 4)

    assert_almost_equal(_kappa(*m), 1.0, 4)


def test_1100():
    m = (1, 1, 0, 0)
    cm = ConfusionMatrix2.from_ccw(*m)
    assert_almost_equal(cm.chisq_score(), 0.0, 4)
    assert_almost_equal(cm.g_score(), 0.0, 4)
    assert_almost_equal(cm.matthews_corr(), 0.0, 4)
    assert_almost_equal(cm.mp_corr(), 0.0, 4)
    assert_almost_equal(cm.kappa(), 0.0, 4)

    assert_true(np.isnan(cm.loevinger_coeff()))
    assert_true(np.isnan(cm.yule_q()))
    assert_true(np.isnan(cm.yule_y()))

    assert_almost_equal(_kappa(*m), 0.0, 4)


def test_0011():
    m = (0, 0, 1, 1)
    cm = ConfusionMatrix2.from_ccw(*m)
    assert_almost_equal(cm.chisq_score(), 0.0, 4)
    assert_almost_equal(cm.g_score(), 0.0, 4)
    assert_almost_equal(cm.matthews_corr(), 0.0, 4)
    assert_almost_equal(cm.mp_corr(), 0.0, 4)
    assert_almost_equal(cm.kappa(), 0.0, 4)

    assert_true(np.isnan(cm.loevinger_coeff()))
    assert_true(np.isnan(cm.yule_q()))
    assert_true(np.isnan(cm.yule_y()))

    assert_almost_equal(_kappa(*m), 0.0, 4)


def test_0101():
    m = (0, 1, 0, 1)
    cm = ConfusionMatrix2.from_ccw(*m)
    assert_almost_equal(cm.chisq_score(), 2.0, 4)
    assert_almost_equal(cm.g_score(), 2.7726, 4)
    assert_almost_equal(cm.matthews_corr(), -1.0, 4)
    assert_almost_equal(cm.mp_corr(), -1.0, 4)
    assert_almost_equal(cm.kappa(), -1.0, 4)

    assert_almost_equal(cm.loevinger_coeff(), -1.0, 4)
    assert_almost_equal(cm.yule_q(), -1.0, 4)
    assert_almost_equal(cm.yule_y(), -1.0, 4)

    assert_almost_equal(_kappa(*m), -1.0, 4)


def test_1001():
    m = (1, 0, 0, 1)
    cm = ConfusionMatrix2.from_ccw(*m)
    assert_almost_equal(cm.chisq_score(), 0.0, 4)
    assert_almost_equal(cm.g_score(), 0.0, 4)
    assert_almost_equal(cm.matthews_corr(), 0.0, 4)
    assert_almost_equal(cm.mp_corr(), 0.0, 4)
    assert_almost_equal(cm.kappa(), 0.0, 4)

    assert_true(np.isnan(cm.loevinger_coeff()))
    assert_true(np.isnan(cm.yule_q()))
    assert_true(np.isnan(cm.yule_y()))

    assert_almost_equal(_kappa(*m), 0.0, 4)


def test_0110():
    m = (0, 1, 1, 0)
    cm = ConfusionMatrix2.from_ccw(*m)
    assert_almost_equal(cm.chisq_score(), 0.0, 4)
    assert_almost_equal(cm.g_score(), 0.0, 4)
    assert_almost_equal(cm.matthews_corr(), 0.0, 4)
    assert_almost_equal(cm.mp_corr(), 0.0, 4)
    assert_almost_equal(cm.kappa(), 0.0, 4)

    assert_true(np.isnan(cm.loevinger_coeff()))
    assert_true(np.isnan(cm.yule_q()))
    assert_true(np.isnan(cm.yule_y()))

    assert_almost_equal(_kappa(*m), 0.0, 4)


def test_0111():
    m = (0, 1, 1, 1)
    cm = ConfusionMatrix2.from_ccw(*m)
    assert_almost_equal(cm.chisq_score(), 0.75, 4)
    assert_almost_equal(cm.g_score(), 1.0465, 4)
    assert_almost_equal(cm.matthews_corr(), -0.5, 4)
    assert_almost_equal(cm.mp_corr(), -0.5, 4)
    assert_almost_equal(cm.kappa(), -0.5, 4)
    assert_almost_equal(_kappa(*m), -0.5, 4)


def test_1011():
    m = (1, 0, 1, 1)
    cm = ConfusionMatrix2.from_ccw(*m)
    assert_almost_equal(cm.chisq_score(), 0.75, 4)
    assert_almost_equal(cm.g_score(), 1.0465, 4)
    assert_almost_equal(cm.matthews_corr(), 0.5, 4)
    assert_almost_equal(cm.mp_corr(), 0.5, 4)
    assert_almost_equal(cm.kappa(), 0.4, 4)
    assert_almost_equal(_kappa(*m), 0.4, 4)


def test_1101():
    m = (1, 1, 0, 1)
    cm = ConfusionMatrix2.from_ccw(*m)
    assert_almost_equal(cm.chisq_score(), 0.75, 4)
    assert_almost_equal(cm.g_score(), 1.0465, 4)
    assert_almost_equal(cm.matthews_corr(), -0.5, 4)
    assert_almost_equal(cm.mp_corr(), -0.5, 4)
    assert_almost_equal(cm.kappa(), -0.5, 4)
    assert_almost_equal(_kappa(*m), -0.5, 4)


def test_1110():
    m = (1, 1, 1, 0)
    cm = ConfusionMatrix2.from_ccw(*m)
    assert_almost_equal(cm.chisq_score(), 0.75, 4)
    assert_almost_equal(cm.g_score(), 1.0465, 4)
    assert_almost_equal(cm.matthews_corr(), 0.5, 4)
    assert_almost_equal(cm.mp_corr(), 0.5, 4)
    assert_almost_equal(cm.kappa(), 0.4, 4)
    assert_almost_equal(_kappa(*m), 0.4, 4)


def test_1111():
    m = (1, 1, 1, 1)
    cm = ConfusionMatrix2.from_ccw(*m)
    assert_almost_equal(cm.chisq_score(), 0.0, 4)
    assert_almost_equal(cm.g_score(), 0.0, 4)
    assert_almost_equal(cm.matthews_corr(), 0.0, 4)
    assert_almost_equal(cm.mp_corr(), 0.0, 4)
    assert_almost_equal(cm.kappa(), 0.0, 4)
    assert_almost_equal(_kappa(*m), 0.0, 4)


def test_kappa_precalculated():
    # from literature
    assert_almost_equal(cohen_kappa(22, 4, 11, 2),
                        0.67, 2)
    assert_almost_equal(matthews_corr(22, 4, 11, 2),
                        0.67, 2)
    assert_almost_equal(cohen_kappa(147, 10, 62, 3),
                        0.86, 2)
    assert_almost_equal(matthews_corr(147, 10, 62, 3),
                        0.87, 2)
    # numeric stability cases
    assert_almost_equal(cohen_kappa(69, 1, 3, 11),
                        0.280000, 6)
    assert_almost_equal(matthews_corr(69, 1, 3, 11),
                        0.350000, 6)
    assert_almost_equal(cohen_kappa(1, 2, 96, 5),
                        0.191111, 6)
    assert_almost_equal(matthews_corr(1, 2, 96, 5),
                        0.203746, 6)
