# -*- coding: utf-8 -*-
import unittest
import random
from cityhash import CityHash128
from lsh.utils import randset, sigsim, randstr
from lsh import MinHashSignature, SimHashSignature, \
    jaccard_sim, hamming_ndist, Shingler, RegexTokenizer, \
    from_bitstring, bitstring_padded


class TestSig(unittest.TestCase):
    def test_shingler(self):

        s = Shingler(span=5, skip=1, unique=True)
        shingles = s.get_shingles("abracadabra")
        self.assertIn(("d", "b", "a"), shingles)

        t = Shingler(span=5, skip=1, unique=False)
        shingles = t.get_shingles("abracadabra")
        self.assertEqual(("a", "r", "c"), shingles[0])

    def test_word_shingler(self):

        s = Shingler(span=5, skip=1, unique=True, tokenizer=RegexTokenizer())
        shingles = s.get_shingles("the quick brown fox jumps over a lazy dog")
        self.assertIn(("jumps", "a", "dog"), shingles)

        t = Shingler(span=5, skip=1, unique=False, tokenizer=RegexTokenizer())
        shingles = t.get_shingles("the quick brown fox jumps over a lazy dog")
        self.assertEqual(("the", "brown", "jumps"), shingles[0])

    def test_signature_length(self):
        """Signatures should have correct dimension"""
        mh = MinHashSignature(10 * 10)
        self.assertEqual(100, len(mh.get_signature(randset())))

    def test_consistent_signature(self):
        """Signatures should be consistent"""
        mh = MinHashSignature(10 * 10)
        s = randset()
        self.assertEqual(mh.get_signature(s), mh.get_signature(s))

    def test_simhash64(self):
        """Simhash signature of an empty string should be zero
        and unicode and regular strings should give the same
        simhash signatures
        """
        sh = SimHashSignature(64)

        sig1 = sh.get_signature("")
        sig2 = sh.get_signature(u"")
        self.assertEqual(sig1, 0)
        self.assertEqual(sig1, sig2)

        sig3 = sh.get_signature("abracadabra")
        sig4 = sh.get_signature(u"abracadabra")
        self.assertEqual(sig3, sig4)

        str1 = "♡♥❤❥"
        str2 = u"♡♥❤❥"
        sig5 = sh.get_signature(str1)
        sig6 = sh.get_signature(str2)
        self.assertNotEqual(sig5, sig6)

    def test_simhash128(self):
        """Simhash signature of an empty string should be zero
        and unicode and regular strings should give the same
        simhash signatures
        """
        sh = SimHashSignature(128)

        sig1 = sh.get_signature("")
        sig2 = sh.get_signature(u"")
        self.assertEqual(sig1, 0)
        self.assertEqual(sig1, sig2)

        sig3 = sh.get_signature("abracadabra")
        sig4 = sh.get_signature(u"abracadabra")
        self.assertEqual(sig3, sig4)

        str1 = "♡♥❤❥"
        str2 = u"♡♥❤❥"
        sig5 = sh.get_signature(str1)
        sig6 = sh.get_signature(str2)
        self.assertNotEqual(sig5, sig6)

    def test_simhash128_addition(self):
        """Test that our technique of combining two numbers works"""
        for _ in range(100):
            rstr = randstr(random.randint(0, 10))
            a, b = CityHash128(rstr)
            num1 = from_bitstring(bitstring_padded(64, b) + bitstring_padded(64, a))
            num2 = (1 << 64) * a + b
            self.assertEqual(num1, num2)

    def test_simhash_similarity(self):
        """Signatures should be consistent"""
        sh = SimHashSignature(64)

        sig1 = sh.get_signature("abracadabra")
        sig2 = sh.get_signature("")
        dist = hamming_ndist(sig1, sig2)
        self.assertEqual(dist, 25)

        sig1 = sh.get_signature("abracadabra")
        sig2 = sh.get_signature("abracadabra")
        dist = hamming_ndist(sig1, sig2)
        self.assertEqual(dist, 0)

        sig1 = sh.get_signature("abracadabra")
        sig2 = sh.get_signature("abracdabra")
        dist = hamming_ndist(sig1, sig2)
        self.assertEqual(dist, 6)

    def test_simhash_feature_weights(self):
        """Features should be weighted and should contribute to
        similarity/distance
        """
        sh = SimHashSignature(64)

        sig1 = sh.get_signature("abracadabra")
        sig2 = sh.get_signature("abracdabra")
        dist = hamming_ndist(sig1, sig2)
        self.assertEqual(dist, 6)

        sig1 = sh.get_signature("abracadabra", ("cats", 0))
        sig2 = sh.get_signature("abracdabra", ("dogs", 0))
        dist = hamming_ndist(sig1, sig2)
        self.assertEqual(dist, 6)

        sig1 = sh.get_signature("abracadabra", ("cats", 0))
        sig2 = sh.get_signature("abracadabra", ("dogs", 0))
        dist = hamming_ndist(sig1, sig2)
        self.assertEqual(dist, 0)

        sig1 = sh.get_signature("abracadabra", ("ca", 4))
        sig2 = sh.get_signature("abracadabra", ("do", 4))
        dist = hamming_ndist(sig1, sig2)
        self.assertEqual(dist, 0)

        sig1 = sh.get_signature("abracadabra", ("ca", 5))
        sig2 = sh.get_signature("abracadabra", ("do", 5))
        dist = hamming_ndist(sig1, sig2)
        self.assertEqual(dist, 7)

        sig1 = sh.get_signature("abracadabra", ("cats", 200))
        sig2 = sh.get_signature("abracadabra", ("dogs", 200))
        dist = hamming_ndist(sig1, sig2)
        self.assertEqual(dist, 17)

        sig1 = sh.get_signature("abracadabra", ("cats", 10))
        sig2 = sh.get_signature("abracadabra", ("cats", 10))
        dist = hamming_ndist(sig1, sig2)
        self.assertEqual(dist, 0)

    def test_signature_similarity(self):
        """The probability that two sets' signatures match at some index
        are equal is equal to the Jaccard similarity between the two"""
        n_tests = 100
        expected_error = 1.0 / 10  # Expected error is O(1/sqrt(dim))
        mh = MinHashSignature(10 * 10)
        err = 0.0

        for test in range(n_tests):
            # Create random sets and their signatures
            sets = (randset(), randset())
            sigs = map(mh.get_signature, sets)

            # Calculate true Jaccard similarity, and sim of signatures
            jsim = jaccard_sim(*sets)
            ssim = sigsim(*sigs, dim=100)

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