import unittest
from math import sqrt
from utils import randset, sigsim
from lsh import MinHashSignature, jaccard_sim


class TestSig(unittest.TestCase):
    def test_signature_length(self):
        """Signatures should have correct dimension"""
        dim = 100
        mh = MinHashSignature(dim)
        self.assertEqual(dim, len(mh.get_signature(randset())))

    def test_consistent_signature(self):
        """Signatures should be consistent"""
        mh = MinHashSignature(100)
        s = randset()
        self.assertEqual(mh.get_signature(s), mh.get_signature(s))

    def test_signature_similarity(self):
        """The probability that two sets' signatures match at some index
        are equal is equal to the Jaccard similarity between the two"""
        dim = 100
        n_tests = 100
        expected_error = 1.0 / sqrt(dim)  # Expected error is O(1/sqrt(dim))
        mh = MinHashSignature(dim)
        err = 0.0

        for test in range(n_tests):
            # Create random sets and their signatures
            sets = (randset(), randset())
            sigs = map(mh.get_signature, sets)

            # Calculate true Jaccard similarity, and sim of signatures
            jsim = jaccard_sim(*sets)
            ssim = sigsim(*sigs, dim=dim)

            # Accumulate error
            err += abs(jsim - ssim)

        # Over n_tests large, we should be within upper bound of expected error
        avg_err = err / n_tests
        self.assertGreaterEqual(
            expected_error,
            avg_err,
            msg="Accuracy test failed. (avg error: %f)" % avg_err)


if __name__ == '__main__':
    unittest.main()