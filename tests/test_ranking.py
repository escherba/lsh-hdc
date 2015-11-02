import numpy as np
from itertools import chain
from pymaptools.containers import clusters_to_labels
from lsh_hdc.ranking import RocCurve, \
    aul_score_from_clusters, aul_score_from_labels
from nose.tools import assert_almost_equal
from pymaptools.sample import discrete_sample, random_seed


def simulate_predictions(n=100, seed=None):
    """simulate classifier predictions for data size of n
    """
    if seed is None:
        seed = random_seed()
    np.random.seed(seed % (2 ** 32))
    probas = [np.random.random() for _ in xrange(n)]
    classes = [discrete_sample({0: (1 - p), 1: p}) for p in probas]
    return classes, probas



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



def test_roc_curve():
    # Test Area under Receiver Operating Characteristic (ROC) curve
    for _ in range(10):
        y_true, probas_pred = simulate_predictions(1000, seed=random_seed())
        rc = RocCurve.from_binary(y_true, probas_pred)
        expected_auc = _auc(rc.fprs, rc.tprs)
        score = rc.auc_score()
        assert_almost_equal(expected_auc, score, 2)


def test_clustering_aul_empty():
    """Empty clusterings have AUL=0.0
    """
    clusters = []
    score1 = aul_score_from_labels(*clusters_to_labels(clusters))
    score2 = aul_score_from_clusters(clusters)
    assert_almost_equal(score1, 0.0, 4)
    assert_almost_equal(score2, 0.0, 4)


def test_clustering_aul_perfect():
    """Perfect clusterings have AUL=1.0
    """
    clusters = [[1, 1, 1, 1, 1], [0], [0]]
    score1 = aul_score_from_labels(*clusters_to_labels(clusters))
    score2 = aul_score_from_clusters(clusters)
    assert_almost_equal(score1, 1.0, 4)
    assert_almost_equal(score2, 1.0, 4)


def test_clustering_aul_bad():
    """Bad clusterings have have AUL=0.5
    """
    clusters = [[1, 1, 0, 0], [0]]
    score1 = aul_score_from_labels(*clusters_to_labels(clusters))
    score2 = aul_score_from_clusters(clusters)
    assert_almost_equal(score1, 0.5, 4)
    assert_almost_equal(score2, 0.5, 4)


def test_clustering_aul_perverse():
    """Perverese cases are 0.0 < AUL < 0.5
    """
    clusters = [[1], [0, 0]]
    score1 = aul_score_from_labels(*clusters_to_labels(clusters))
    score2 = aul_score_from_clusters(clusters)
    assert_almost_equal(score1, 0.1111, 4)
    assert_almost_equal(score2, 0.1111, 4)


def test_clustering_aul_precalculated():
    """A decent clustering should have a high score`
    """
    clusters = [[1, 1, 1], [1, 1], [0], [0]]
    score1 = aul_score_from_labels(*clusters_to_labels(clusters))
    score2 = aul_score_from_clusters(clusters)
    assert_almost_equal(score1, 0.8286, 4)
    assert_almost_equal(score2, 0.8286, 4)


