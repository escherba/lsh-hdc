"""
Randomized tests for invariant properties of some clustering metrics
"""

import numpy as np
import warnings
from lsh_hdc.metrics import ClusteringMetrics,  ConfusionMatrix2, \
    geometric_mean, harmonic_mean, _div, \
    adjusted_rand_score, mutual_info_score, \
    adjusted_mutual_info_score
from numpy.testing import assert_array_almost_equal
from nose.tools import assert_almost_equal, assert_true, assert_equal
from sklearn.metrics import \
    homogeneity_completeness_v_measure as sklearn_hcv, \
    adjusted_rand_score as sklearn_ari, \
    mutual_info_score as sklearn_mi, \
    adjusted_mutual_info_score as sklearn_ami


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


def test_RxC_invariants():
    """Alternative implementations should coincide for RxC matrices
    """
    for _ in xrange(1000):
        ltrue = np.random.randint(low=0, high=5, size=(20,))
        lpred = np.random.randint(low=0, high=5, size=(20,))
        cm = ClusteringMetrics.from_labels(ltrue, lpred)

        # homogeneity, completeness, V-measure
        expected_hcv = sklearn_hcv(ltrue, lpred)
        actual_hcv = cm.entropy_metrics()
        assert_array_almost_equal(actual_hcv, expected_hcv)

        # mutual information score
        expected_mi = sklearn_mi(ltrue, lpred)
        actual_mi = mutual_info_score(ltrue, lpred)
        assert_array_almost_equal(actual_mi, expected_mi)

        # adjusted mutual information
        expected_ami = sklearn_ami(ltrue, lpred)
        actual_ami = adjusted_mutual_info_score(ltrue, lpred)
        assert_array_almost_equal(actual_ami, expected_ami)

        # adjusted rand index
        expected_ari = sklearn_ari(ltrue, lpred)
        actual_ari = adjusted_rand_score(ltrue, lpred)
        assert_array_almost_equal(actual_ari, expected_ari)


def test_2x2_invariants():
    """Alternative implementations should coincide for 2x2 matrices
    """

    for _ in xrange(1000):
        cm = ConfusionMatrix2.from_random_counts(low=0, high=10)

        # object idempotency
        assert_equal(
            cm.to_ccw(),
            ConfusionMatrix2.from_ccw(*cm.to_ccw()).to_ccw(),
            msg="must be able to convert to tuple and create from tuple")

        # informedness
        actual_info = cm.informedness()
        expected_info_1 = cm.sensitivity() + cm.specificity() - 1.0
        expected_info_2 = cm.TPR() - cm.FPR()
        check_with_nans(actual_info, expected_info_1, 4, ensure_nans=False)
        check_with_nans(actual_info, expected_info_2, 4, ensure_nans=False)

        # markedness
        actual_mark = cm.markedness()
        expected_mark = cm.PPV() + cm.NPV() - 1.0
        check_with_nans(actual_mark, expected_mark, 4, ensure_nans=False,
                        msg="Markedness must be equal to expected")

        # matthews corr coeff
        actual_mcc = cm.matthews_corr()
        expected_mcc = geometric_mean(actual_info, actual_mark)
        check_with_nans(actual_mcc, expected_mcc, 4, ensure_nans=False,
                        msg="MCC1 and MCC 2 must be the same")

        # odds ratio and Yule's Q
        actual_odds_ratio = cm.DOR()
        actual_yule_q = cm.yule_q()
        expected_yule_q = _div(actual_odds_ratio - 1.0, actual_odds_ratio + 1.0)
        expected_odds_ratio = _div(cm.PLL(), cm.NLL())
        check_with_nans(actual_odds_ratio, expected_odds_ratio, 4, ensure_nans=False,
                        msg="DOR must be equal to PLL/NLL")
        check_with_nans(actual_yule_q, expected_yule_q, 4, ensure_nans=False,
                        msg="Yule's Q must be equal to (DOR-1)/(DOR+1)")

        # F-score and Dice
        expected_f = harmonic_mean(cm.precision(), cm.recall())
        actual_f = cm.fscore()
        check_with_nans(expected_f, actual_f, 6,
                        msg="Fscore must be equal to expected")
        check_with_nans(expected_f, cm.dice_coeff(), 6, ensure_nans=False,
                        msg="Fscore must be equal to Dice")

        # association coefficients (1)
        dice = cm.dice_coeff()
        expected_jaccard = _div(dice, 2.0 - dice)
        actual_jaccard = cm.jaccard_coeff()
        check_with_nans(actual_jaccard, expected_jaccard, 6, ensure_nans=False,
                        msg="Jaccard coeff must match expected value")

        # association coefficients (2)
        jaccard = cm.jaccard_coeff()
        expected_ss2 = _div(jaccard, 2.0 - jaccard)
        actual_ss2 = cm.sokal_sneath_coeff()
        check_with_nans(actual_ss2, expected_ss2, 6, ensure_nans=False,
                        msg="SS2 coeff must match expected value")

        # association coefficients (3)
        gl = cm.gower_legendre_coeff()
        expected_acc = _div(gl, 2.0 - gl)
        actual_acc = cm.accuracy()
        check_with_nans(actual_acc, expected_acc, 6, ensure_nans=False,
                        msg="Accuracy coeff must match expected value")

        # association coefficients (4)
        expected_rt = _div(actual_acc, 2.0 - actual_acc)
        actual_rt = cm.rogers_tanimoto_coeff()
        check_with_nans(actual_rt, expected_rt, 6, ensure_nans=False,
                        msg="Rogers-Tanimoto coeff must match expected value")