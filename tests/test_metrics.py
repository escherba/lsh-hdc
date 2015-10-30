import random
import numpy as np
import warnings
from itertools import chain
from pymaptools.sample import discrete_sample, random_seed
from lsh_hdc.metrics import RocCurve, adjusted_rand_score, \
    homogeneity_completeness_v_measure, centropy, \
    jaccard_similarity, clustering_aul_score, ClusteringMetrics, \
    ConfusionMatrix2, geometric_mean, harmonic_mean, _div, cohen_kappa, \
    matthews_corr
from numpy.testing import assert_array_almost_equal
from nose.tools import assert_almost_equal, assert_true, assert_equal


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


def _auc(fpr, tpr, reorder=False):
    """Compute area under ROC curve

    This is a simple alternative implementation for testing.
    For production tasks, use Sklearn's implementation.
    """

    def generic_auc(xs, ys, reorder=False):
        """Compute area under a curve using trapesoidal rule
        """
        tuples = zip(xs, ys)
        if not tuples:
            return float('nan')
        if reorder:
            tuples.sort()
        a = 0.0
        x0, y0 = tuples[0]
        for x1, y1 in tuples[1:]:
            a += (x1 - x0) * (y1 + y0)
            x0, y0 = x1, y1
        return a * 0.5

    return generic_auc(
        chain([0.0], fpr, [1.0]),
        chain([0.0], tpr, [1.0]),
        reorder=reorder)


def simulate_predictions(n=100, seed=None):
    """simulate classifier predictions for data size of n
    """
    if seed is None:
        seed = random_seed()
    random.seed(seed)
    probas = [random.random() for _ in xrange(n)]
    classes = [discrete_sample({0: (1 - p), 1: p}) for p in probas]
    return classes, probas


def test_roc_curve():
    # Test Area under Receiver Operating Characteristic (ROC) curve
    for _ in range(10):
        y_true, probas_pred = simulate_predictions(1000, seed=random_seed())
        rc = RocCurve.from_binary(y_true, probas_pred)
        expected_auc = _auc(rc.fprs, rc.tprs)
        score = rc.auc_score()
        assert_almost_equal(expected_auc, score, 2)


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
    assert_almost_equal(c, 0.00, 2)
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

    assert_almost_equal(cm.chisq_score(),     9.017647, 6)
    assert_almost_equal(cm.g_score(),        13.325845, 6)

    # test metrics that are based on pairwise co-association matrix
    coassoc = cm.coassoc_

    assert_almost_equal(coassoc.chisq_score(),   8.063241, 6)
    assert_almost_equal(coassoc.g_score(),       7.804221, 6)

    assert_almost_equal(coassoc.jaccard_coeff(), 0.312500, 6)
    assert_almost_equal(coassoc.ochiai_coeff(),  0.476731, 6)
    assert_almost_equal(coassoc.fscore(),        0.476190, 6)
    assert_almost_equal(coassoc.dice_coeff(),    0.476190, 6)

    assert_almost_equal(coassoc.rand_index(),    0.676471, 6)
    assert_almost_equal(coassoc.precision(),     0.500000, 6)
    assert_almost_equal(coassoc.recall(),        0.454545, 6)


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


def test_clustering_aul_empty():
    """Test empty clustering first
    """
    score = clustering_aul_score([], bool)
    assert_true(np.isnan(score))


def test_clustering_aul_perfect():
    """Test empty clustering first
    """
    clusters = [[1, 1, 1, 1, 1], [0], [0]]
    score = clustering_aul_score(clusters, bool)
    assert_almost_equal(score, 1.0, 4)


def test_clustering_aul_precalculated():
    """Test empty clustering first
    """
    clusters = [[1, 1, 1], [1, 1], [0], [0]]
    score = clustering_aul_score(clusters, bool)
    assert_almost_equal(score, 0.8286, 4)


def test_twoway_confusion_ll():
    """Example from McDonald's G-test for independence
    http://www.biostathandbook.com/gtestind.html
    """
    cm = ConfusionMatrix2.from_ccw(4758, 8840, 76, 30)
    assert_almost_equal(cm.g_score(),       2.14, 2)
    mi_info, mi_mark, mi_corr = cm.mutinf_signed()
    assert_almost_equal(mi_corr,            0.0150, 4)
    assert_almost_equal(mi_info,            0.0110, 4)
    assert_almost_equal(mi_mark,            0.0415, 4)
    assert_almost_equal(cm.chisq_score(),   2.07, 2)
    assert_almost_equal(cm.matthews_corr(), 0.0123, 4)
    assert_almost_equal(cm.informedness(),  0.0023, 4)
    assert_almost_equal(cm.markedness(),    0.0669, 4)
    assert_almost_equal(cm.kappa(),         0.0016, 4)


def test_negative_correlation():
    """Some metrics should have negative sign
    """
    cm = ConfusionMatrix2.from_ccw(10, 120, 8, 300)
    assert_almost_equal(cm.g_score(),        384.52, 2)
    mi_info, mi_mark, mi_corr = cm.mutinf_signed()
    assert_almost_equal(mi_corr,            -0.8510, 4)
    assert_almost_equal(mi_info,            -0.8524, 4)
    assert_almost_equal(mi_mark,            -0.8496, 4)
    assert_almost_equal(cm.chisq_score(),    355.70, 2)
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
    """
    """
    m = (0, 0, 0, 0)
    cm = ConfusionMatrix2.from_ccw(*m)
    assert_almost_equal(cm.chisq_score(), 0.0, 4)
    assert_almost_equal(cm.g_score(), 0.0, 4)
    assert_true(np.isnan(cm.matthews_corr()))
    assert_true(np.isnan(cm.kappa()))
    assert_true(np.isnan(_kappa(*m)))


def test_1000():
    """
    """
    m = (1, 0, 0, 0)
    cm = ConfusionMatrix2.from_ccw(*m)
    assert_almost_equal(cm.chisq_score(), 0.0, 4)
    assert_almost_equal(cm.g_score(), 0.0, 4)
    assert_true(np.isnan(cm.matthews_corr()))
    assert_true(np.isnan(cm.kappa()))
    assert_true(np.isnan(_kappa(*m)))


def test_0100():
    """
    """
    m = (0, 1, 0, 0)
    cm = ConfusionMatrix2.from_ccw(*m)
    assert_almost_equal(cm.chisq_score(), 0.0, 4)
    assert_almost_equal(cm.g_score(), 0.0, 4)
    assert_true(np.isnan(cm.matthews_corr()))
    assert_true(np.isnan(cm.kappa()))
    assert_true(np.isnan(_kappa(*m)))


def test_0010():
    """
    """
    m = (0, 0, 1, 0)
    cm = ConfusionMatrix2.from_ccw(*m)
    assert_almost_equal(cm.chisq_score(), 0.0, 4)
    assert_almost_equal(cm.g_score(), 0.0, 4)
    assert_true(np.isnan(cm.matthews_corr()))
    assert_true(np.isnan(cm.kappa()))
    assert_true(np.isnan(_kappa(*m)))


def test_0001():
    """
    """
    m = (0, 0, 0, 1)
    cm = ConfusionMatrix2.from_ccw(*m)
    assert_almost_equal(cm.chisq_score(), 0.0, 4)
    assert_almost_equal(cm.g_score(), 0.0, 4)
    assert_true(np.isnan(cm.matthews_corr()))
    assert_true(np.isnan(cm.kappa()))
    assert_true(np.isnan(_kappa(*m)))


def test_1010():
    """
    """
    m = (1, 0, 1, 0)
    cm = ConfusionMatrix2.from_ccw(*m)
    assert_almost_equal(cm.chisq_score(), 2.0, 4)
    assert_almost_equal(cm.g_score(), 2.7726, 4)
    assert_almost_equal(cm.matthews_corr(), 1.0, 4)
    assert_almost_equal(cm.kappa(), 1.0, 4)
    assert_almost_equal(_kappa(*m), 1.0, 4)


def test_1100():
    """
    """
    m = (1, 1, 0, 0)
    cm = ConfusionMatrix2.from_ccw(*m)
    assert_almost_equal(cm.chisq_score(), 0.0, 4)
    assert_almost_equal(cm.g_score(), 0.0, 4)
    assert_almost_equal(cm.matthews_corr(), 0.0, 4)
    assert_almost_equal(cm.kappa(), 0.0, 4)
    assert_almost_equal(_kappa(*m), 0.0, 4)


def test_0011():
    """
    """
    m = (0, 0, 1, 1)
    cm = ConfusionMatrix2.from_ccw(*m)
    assert_almost_equal(cm.chisq_score(), 0.0, 4)
    assert_almost_equal(cm.g_score(), 0.0, 4)
    assert_almost_equal(cm.matthews_corr(), 0.0, 4)
    assert_almost_equal(cm.kappa(), 0.0, 4)
    assert_almost_equal(_kappa(*m), 0.0, 4)


def test_0101():
    """
    """
    m = (0, 1, 0, 1)
    cm = ConfusionMatrix2.from_ccw(*m)
    assert_almost_equal(cm.chisq_score(), 2.0, 4)
    assert_almost_equal(cm.g_score(), 2.7726, 4)
    assert_almost_equal(cm.matthews_corr(), -1.0, 4)
    assert_almost_equal(cm.kappa(), -1.0, 4)
    assert_almost_equal(_kappa(*m), -1.0, 4)


def test_0111():
    """
    """
    m = (0, 1, 1, 1)
    cm = ConfusionMatrix2.from_ccw(*m)
    assert_almost_equal(cm.chisq_score(), 0.75, 4)
    assert_almost_equal(cm.g_score(), 1.0465, 4)
    assert_almost_equal(cm.matthews_corr(), -0.5, 4)
    assert_almost_equal(cm.kappa(), -0.5, 4)
    assert_almost_equal(_kappa(*m), -0.5, 4)


def test_1011():
    """
    """
    m = (1, 0, 1, 1)
    cm = ConfusionMatrix2.from_ccw(*m)
    assert_almost_equal(cm.chisq_score(), 0.75, 4)
    assert_almost_equal(cm.g_score(), 1.0465, 4)
    assert_almost_equal(cm.matthews_corr(), 0.5, 4)
    assert_almost_equal(cm.kappa(), 0.4, 4)
    assert_almost_equal(_kappa(*m), 0.4, 4)


def test_1101():
    """
    """
    m = (1, 1, 0, 1)
    cm = ConfusionMatrix2.from_ccw(*m)
    assert_almost_equal(cm.chisq_score(), 0.75, 4)
    assert_almost_equal(cm.g_score(), 1.0465, 4)
    assert_almost_equal(cm.matthews_corr(), -0.5, 4)
    assert_almost_equal(cm.kappa(), -0.5, 4)
    assert_almost_equal(_kappa(*m), -0.5, 4)


def test_1110():
    """
    """
    m = (1, 1, 1, 0)
    cm = ConfusionMatrix2.from_ccw(*m)
    assert_almost_equal(cm.chisq_score(), 0.75, 4)
    assert_almost_equal(cm.g_score(), 1.0465, 4)
    assert_almost_equal(cm.matthews_corr(), 0.5, 4)
    assert_almost_equal(cm.kappa(), 0.4, 4)
    assert_almost_equal(_kappa(*m), 0.4, 4)


def test_1111():
    """
    """
    m = (1, 1, 1, 1)
    cm = ConfusionMatrix2.from_ccw(*m)
    assert_almost_equal(cm.chisq_score(), 0.0, 4)
    assert_almost_equal(cm.g_score(), 0.0, 4)
    assert_almost_equal(cm.matthews_corr(), 0.0, 4)
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


def test_clusterings_randomize():
    """Alternative implementations should coincide for a random sample
    """

    for _ in xrange(1000):
        ltrue = np.random.randint(low=0, high=5, size=(20,))
        lpred = np.random.randint(low=0, high=5, size=(20,))
        cm = ClusteringMetrics.from_labels(ltrue, lpred)

        for m1, m2 in zip(cm.entropy_metrics(), _entropy_metrics(cm)):
            check_with_nans(m1, m2, 4)


def test_2x2_invariants():
    """Alternative implementations should coincide for a random sample
    """

    for _ in xrange(10000):
        cm = ConfusionMatrix2.from_random_counts(low=0, high=10)
        cells_ccw = cm.to_ccw()

        # check dogfood
        assert_equal(
            cm.to_ccw(),
            ConfusionMatrix2.from_ccw(*cm.to_ccw()).to_ccw(),
            msg="must be able to convert to tuple and create from tuple")

        # check informedness
        actual_info = cm.informedness()
        expected_info_1 = cm.sensitivity() + cm.specificity() - 1.0
        expected_info_2 = cm.TPR() - cm.FPR()
        check_with_nans(actual_info, expected_info_1, 4, ensure_nans=False)
        check_with_nans(actual_info, expected_info_2, 4, ensure_nans=False)

        # check markedness
        actual_mark = cm.markedness()
        expected_mark = cm.PPV() + cm.NPV() - 1.0
        check_with_nans(actual_mark, expected_mark, 4, ensure_nans=False,
                        msg="Markedness must be equal to expected")

        # check matthews corr coeff
        actual_mcc = cm.matthews_corr()
        expected_mcc = geometric_mean(actual_info, actual_mark)
        check_with_nans(actual_mcc, expected_mcc, 4, ensure_nans=False,
                        msg="MCC1 and MCC 2 must be the same")

        # check kappa implementations
        actual_kappa = cm.kappa()
        expected_kappa = _kappa(*cells_ccw)
        check_with_nans(cm.kappa(), expected_kappa, 4,
                        msg="kappas must be equal")

        # check odds ratio implementation
        actual_odds_ratio = cm.DOR()
        expected_odds_ratio = _div(cm.PLL(), cm.NLL())
        check_with_nans(actual_odds_ratio, expected_odds_ratio, 4, ensure_nans=False,
                        msg="DOR must be equal")

        # check F-score and Dice
        expected_f = harmonic_mean(cm.precision(), cm.recall())
        actual_f = cm.fscore()
        check_with_nans(expected_f, actual_f, 6,
                        msg="Fscore must be equal to expected")
        check_with_nans(expected_f, cm.dice_coeff(), 6, ensure_nans=False,
                        msg="Fscore must be equal to Dice")
