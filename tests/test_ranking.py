import numpy as np
from itertools import chain
from pymaptools.containers import clusters_to_labels
from lsh_hdc.ranking import RocCurve, LiftCurve, \
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


def simulate_clustering(galpha=2, gbeta=10, nclusters=20, pos_ratio=0.2, err_pos=0.1,
                        err_neg=0.02):

    csizes = map(int, np.random.gamma(galpha, gbeta, nclusters))
    npos = sum(csizes)
    nneg = int(npos * ((1.0 - pos_ratio) / pos_ratio))
    clusters = []
    for csize in csizes:
        cluster = np.random.choice(2, csize, p=[err_pos, 1.0 - err_pos])
        clusters.append(list(cluster))
    negs = np.random.choice(2, nneg, p=[1.0 - err_neg, err_neg])
    clusters.extend([[x] for x in negs])
    return clusters


def test_aul_simulated():
    """Two different implementations of aul_score should return same numbers
    """

    # test lots of small examples
    for _ in xrange(100):
        clusters = simulate_clustering(galpha=1, gbeta=2, nclusters=4,
                                       pos_ratio=0.5)
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


