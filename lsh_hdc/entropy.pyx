# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

from libc.math cimport exp, log
from scipy.special import gammaln
from collections import Mapping, Iterator
import numpy as np
cimport numpy as np
cimport cython
from lsh_hdc.ext cimport lgamma

np.import_array()


cpdef ndarray_from_iter(iterable, dtype=None, contiguous=False):
    """Create NumPy arrays from different object types

    In addition to standard ``np.asarray`` casting functionality, this function
    handles conversion from the following types: ``collections.Mapping``,
    ``collections.Iterator``.

    If the input object is an instance of ``collections.Mapping``, assumes that
    we are interesting in creating a NumPy array from the values.
    """
    if isinstance(iterable, Iterator):
        arr = np.fromiter(iterable, dtype=dtype)
        if contiguous:
            arr = np.ascontiguousarray(arr, dtype=dtype)
    elif isinstance(iterable, Mapping):
        arr = np.fromiter(iterable.itervalues(), dtype=dtype)
        if contiguous:
            arr = np.ascontiguousarray(arr, dtype=dtype)
    elif contiguous:
        arr = np.ascontiguousarray(iterable, dtype=dtype)
    else:
        arr = np.asarray(iterable, dtype=dtype)
    return arr


cpdef nchoose2(np.int64_t n):
    """Binomial coefficient for k=2

    Scipy has ``scipy.special.binom`` and ``scipy.misc.comb``, however on
    individual (non-vectorized) ops used in memory-constrained stream
    computation, a simple definition below is faster. It is possible to get the
    best of both worlds by writing a generator that returns NumPy arrays of
    limited size and then calling a vectorized n-choose-2 function on those,
    however the current way is fast enough for computing coincidence matrices
    (turns out memory was the bottleneck, not raw computation speed).
    """
    return (n * (n - 1LL)) >> 1LL


cpdef centropy(counts):
    """Entropy of an iterable of counts

    Assumes every entry in the list belongs to a different class. The resulting
    value is *not* normalized by N. Also note that the entropy value is
    calculated using natural base, which may not be what you want, so you may
    need to normalized it with log(base).

    The 'counts' parameter is expected to be an list or tuple-like iterable.
    For convenience, it can also be a dict/mapping type, in which case its
    values will be used to calculate entropy.

    """
    # The Cython version of this method is 50x faster on large arrays than pure
    # CPython implementation. The speed-up is primarily due to the ``cdef
    # np.int64_t c`` definition.

    cdef np.int64_t c, n
    cdef np.float64_t sum_c_logn_c, result

    if isinstance(counts, Mapping):
        counts = counts.itervalues()

    n = 0LL
    sum_c_logn_c = 0.0
    for c in counts:
        if c != 0LL:
            n += c
            sum_c_logn_c += c * log(c)
    result = 0.0 if n == 0LL else n * log(n) - sum_c_logn_c
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef emi_from_margins(np.ndarray[np.int64_t, ndim=1, mode='c'] a,
                       np.ndarray[np.int64_t, ndim=1, mode='c'] b):
    """Calculate Expected Mutual Information given margins of RxC table

    The resulting value is *not* normalized by N.

    License: BSD 3 clause

    .. codeauthor:: Robert Layton <robertlayton@gmail.com>
    .. codeauthor:: Corey Lynch <coreylynch9@gmail.com>
    .. codeauthor:: Eugene Scherba <escherba@gmail.com>

    """
    # (Eugene Scherba, 10/2/2015): I modified this function so as to move all
    # heavy operations inside the main loop. Specifically, the code creating RxC
    # arrays that previously resulted in O(n^2) memory requirements was removed
    # and/or rewritten. Although there may have been marginal benefits from
    # preparing data for the main loop using vectorized NumPy operations, the
    # main speed bottleneck of this code has been the three-tier loop. So it
    # makes sense to just move all operations inside there to avoid O(n^2)
    # memory explosions on large data.

    cdef Py_ssize_t R, C, i, j, nij

    cdef np.int64_t N, N_1, max_ab, ai, bj, ai_1, bj_1, ai_bj, N_ai_bj_1

    cdef np.ndarray[np.float64_t, ndim=1, mode='c'] \
        log_a, log_b, log_Nnij, nijs, gln_ai_Nai_N, gln_b_Nb, gln_nij

    cdef np.float64_t emi, log_ai, log_ab_outer_ij, outer_sum, \
                      gln_ai_Nai_Ni, term2, term3

    log_a = np.log(a)
    log_b = np.log(b)

    R = len(a)
    C = len(b)

    N = np.sum(a)
    if N != np.sum(b):
        raise ValueError("Sums of row and column margins must be equal")

    # There are three major terms to the EMI equation, which are multiplied to
    # and then summed over varying nij values.

    # While nijs[0] will never be used, having it simplifies the indexing.
    max_ab = max(<np.int64_t>np.max(a), <np.int64_t>np.max(b))
    nijs = np.arange(0LL, max_ab + 1LL, dtype=np.float64)
    nijs[0] = 1.0  # Stops divide by zero warnings. As its not used, no issue.

    # term2 is log((N*nij) / (a a b)) == log(N * nij) - log(a * b)
    # term2 uses log(N * nij)
    log_Nnij = log(N) + np.log(nijs)

    # term3 is large, and involved many factorials. Calculate these in log
    # space to stop overflows.
    N_1 = N + 1LL
    gln_ai_Nai_N = gammaln(a + 1LL) + gammaln(N_1 - a) - gammaln(N_1)
    gln_b_Nb = gammaln(b + 1LL) + gammaln(N_1 - b)
    gln_nij = gammaln(nijs + 1.0)

    # emi itself is a summation over the various values.
    emi = 0.0
    for i in xrange(R):
        ai = a[i]
        ai_1 = ai + 1LL
        log_ai = log_a[i]
        gln_ai_Nai_Ni = gln_ai_Nai_N[i]
        for j in xrange(C):
            bj = b[j]
            bj_1 = bj + 1LL

            log_ab_outer_ij = log_ai + log_b[j]
            outer_sum = gln_ai_Nai_Ni + gln_b_Nb[j]

            ai_bj = ai + bj
            N_ai_bj_1 = N_1 - ai_bj

            for nij in xrange(max(1LL, ai_bj - N), min(ai_1, bj_1)):
                term2 = log_Nnij[nij] - log_ab_outer_ij
                # Numerators are positive, denominators are negative.
                term3 = exp(outer_sum
                    - gln_nij[nij] - lgamma(ai_1 - nij)
                    - lgamma(bj_1 - nij)
                    - lgamma(nij + N_ai_bj_1)
                )
                emi += (nijs[nij] * term2 * term3)
    return emi
