import numpy as np
from itertools import chain
from pymaptools.containers import clusters_to_labels
from lsh_hdc.ranking import RocCurve, LiftCurve, \
    aul_score_from_clusters, aul_score_from_labels, roc_auc_score
from nose.tools import assert_almost_equal
from pymaptools.sample import discrete_sample, random_seed
from lsh_hdc.monte_carlo.predictions import simulate_clustering
from sklearn.metrics.ranking import roc_auc_score as auc_sklearn


def simulate_predictions(n=100, seed=None):
    """simulate classifier predictions for data size of n
    """
    if seed is None:
        seed = random_seed()
    np.random.seed(seed % (2 ** 32))
    probas = np.random.random(n)
    classes = [discrete_sample({0: (1 - p), 1: p}) for p in probas]
    return classes, probas


def test_simulated():
    """Two different implementations of aul_score should return same numbers
    """

    # test lots of small examples
    for _ in xrange(100):
        clusters = simulate_clustering(galpha=1, gbeta=2, nclusters=4,
                                       pos_ratio=0.5, sim_size=20)
        lc = LiftCurve.from_clusters(clusters)
        expected_score = lc.aul_score(plot=True)[0]
        actual_score = lc.aul_score(plot=False)
        assert_almost_equal(expected_score, actual_score, 4)

    # test a few large ones too
    for _ in xrange(10):
        clusters = simulate_clustering()
        lc = LiftCurve.from_clusters(clusters)
        expected_score = lc.aul_score(plot=True)[0]
        actual_score = lc.aul_score(plot=False)
        assert_almost_equal(expected_score, actual_score, 4)


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


def test_roc_simulated():
    # Test Area under Receiver Operating Characteristic (ROC) curve
    for _ in range(10):
        y_true, probas_pred = simulate_predictions(1000, seed=random_seed())
        rc = RocCurve.from_labels(y_true, probas_pred)
        auc_expected1 = _auc(rc.fprs, rc.tprs)
        auc_expected2 = auc_sklearn(y_true, probas_pred)
        auc_actual = roc_auc_score(y_true, probas_pred)
        assert_almost_equal(auc_expected1, auc_actual, 3)
        assert_almost_equal(auc_expected2, auc_actual, 3)


def test_sample_empty():
    """Empty clusterings have AUL=0.0
    """
    clusters = []
    score1 = aul_score_from_labels(*clusters_to_labels(clusters))
    score2 = aul_score_from_clusters(clusters)
    assert_almost_equal(score1, 0.0, 4)
    assert_almost_equal(score2, 0.0, 4)


def test_sample_perfect():
    """Perfect clustering
    """
    clusters = [[1, 1, 1, 1, 1], [0], [0]]

    aul1 = aul_score_from_labels(*clusters_to_labels(clusters))
    aul2 = aul_score_from_clusters(clusters)
    assert_almost_equal(aul1, 1.0, 4)
    assert_almost_equal(aul2, 1.0, 4)

    auc = RocCurve.from_clusters(clusters).auc_score()
    assert_almost_equal(auc, 1.0, 4)


def test_sample_class1_nh():
    """Same as in ``test_sample_perfect`` but class 1 not homogeneous
    """
    clusters = [[1, 1, 1], [1, 1], [0], [0]]

    aul1 = aul_score_from_labels(*clusters_to_labels(clusters))
    aul2 = aul_score_from_clusters(clusters)
    assert_almost_equal(aul1, 0.8286, 4)
    assert_almost_equal(aul2, 0.8286, 4)

    auc = RocCurve.from_clusters(clusters).auc_score()
    assert_almost_equal(auc, 1.0, 4)


def test_sample_cluster0_nh():
    """Same as in ``test_sample_perfect`` but cluster 0 not homogeneous
    """
    clusters = [[1, 1, 1, 1, 0], [0], [0]]

    aul1 = aul_score_from_labels(*clusters_to_labels(clusters))
    aul2 = aul_score_from_clusters(clusters)
    assert_almost_equal(aul1, 0.8, 4)
    assert_almost_equal(aul2, 0.8, 4)

    auc = RocCurve.from_clusters(clusters).auc_score()
    assert_almost_equal(auc, 0.8333, 4)


def test_sample_cluster0_c0():
    """Similar to ``test_sample_perfect`` but have a cluster of class 0
    """
    clusters = [[1, 1, 1, 1], [0, 0], [0]]

    aul1 = aul_score_from_labels(*clusters_to_labels(clusters))
    aul2 = aul_score_from_clusters(clusters)
    assert_almost_equal(aul1, 0.6667, 4)
    assert_almost_equal(aul2, 0.6667, 4)

    auc = RocCurve.from_clusters(clusters).auc_score()
    assert_almost_equal(auc, 1.0, 4)


def test_sample_neg_class1():
    """Similar to ``test_sample_perfect`` but have a negative of class 1
    """
    clusters = [[1, 1, 1, 1, 1], [1], [0]]

    aul1 = aul_score_from_labels(*clusters_to_labels(clusters))
    aul2 = aul_score_from_clusters(clusters)
    assert_almost_equal(aul1, 0.8690, 4)
    assert_almost_equal(aul2, 0.8690, 4)

    auc = RocCurve.from_clusters(clusters).auc_score()
    assert_almost_equal(auc, 0.9167, 4)


def test_sample_bad():
    """Bad clustering should score poorly
    """
    clusters = [[1, 1, 0, 0], [0]]

    aul1 = aul_score_from_labels(*clusters_to_labels(clusters))
    aul2 = aul_score_from_clusters(clusters)
    assert_almost_equal(aul1, 0.5, 4)
    assert_almost_equal(aul2, 0.5, 4)

    auc = RocCurve.from_clusters(clusters).auc_score()
    assert_almost_equal(auc, 0.6667, 4)


def test_sample_perverse():
    """Perverese cases are 0.0 < AUL < 0.5
    """
    clusters = [[1], [0, 0]]

    aul1 = aul_score_from_labels(*clusters_to_labels(clusters))
    aul2 = aul_score_from_clusters(clusters)
    assert_almost_equal(aul1, 0.1111, 4)
    assert_almost_equal(aul2, 0.1111, 4)

    auc = RocCurve.from_clusters(clusters).auc_score()
    assert_almost_equal(auc, 0.0, 4)
