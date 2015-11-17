# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

from libc.math cimport exp, log
from scipy.special import gammaln
from collections import Mapping, Iterator
import numpy as np
cimport numpy as np
cimport cython

from libc.stdlib cimport malloc, free

np.import_array()


cdef extern from "gamma.h":
    cdef np.float64_t sklearn_lgamma(np.float64_t x)


cdef extern from "assignmentoptimal_dbl.h":
    void assignmentoptimal_dbl(np.float64_t *assignment, np.float64_t *cost, np.float64_t *distMatrixIn, Py_ssize_t nOfRows, Py_ssize_t nOfColumns)


cdef extern from "assignmentoptimal_lng.h":
    void assignmentoptimal_lng(np.int64_t *assignment, np.int64_t *cost, np.int64_t *distMatrixIn, Py_ssize_t nOfRows, Py_ssize_t nOfColumns)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.int64_t assignment_cost_lng(array2d, maximize=False):
    """Assignment cost of a weighted bipartite matching (int64 version)

    Uses Kuhn-Munkres (Hungarian) algorithm to find an optimal matching between
    two partitions and returns the cost of the matching in approx. O(n^3) time.
    """
    cdef Py_ssize_t n, m

    cdef np.ndarray[np.int64_t, ndim=2, mode='fortran'] contig = \
        np.asfortranarray(array2d, dtype=np.int64)

    if maximize:
        contig = -contig

    n = contig.shape[0]
    m = contig.shape[1]

    cdef np.int64_t* assignment = <np.int64_t*> malloc(n*sizeof(np.int64_t))
    cdef np.int64_t score = 0

    assignmentoptimal_lng(assignment, &score, &contig[0, 0], n, m)

    free(assignment)

    if maximize:
        score = -score

    return score


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.float64_t assignment_cost_dbl(array2d, maximize=False):
    """Assignment cost of a weighted bipartite matching (float64 version)

    Uses Kuhn-Munkres (Hungarian) algorithm to find an optimal matching between
    two partitions and returns the cost of the matching in approx. O(n^3) time.
    """
    cdef Py_ssize_t n, m

    cdef np.ndarray[np.float64_t, ndim=2, mode='fortran'] contig = \
        np.asfortranarray(array2d, dtype=np.float64)

    if maximize:
        contig = -contig

    n = contig.shape[0]
    m = contig.shape[1]

    cdef np.float64_t* assignment = <np.float64_t*> malloc(n*sizeof(np.float64_t))
    cdef np.float64_t score = 0

    assignmentoptimal_dbl(assignment, &score, &contig[0, 0], n, m)

    free(assignment)

    if maximize:
        score = -score

    return score


cpdef np.float64_t lgamma(np.float64_t x):
    """Log of gamma function for scalar double x

    This is a scalar-only replacement for ``scipy.special.gammaln``. On scalar
    values, this method is ~10x faster than the corresponding SciPy one. On
    large arrays, however, even when vectorized using ``np.vectorize``, this
    method is slower than the SciPy one, so use ``gammaln`` in those cases.

    This function is borrowed verbatim from Scikit-Learn.
    """
    if x <= 0.0:
        raise ValueError("x must be strictly positive, got %f" % x)
    return sklearn_lgamma(x)


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


cpdef np.int64_t nchoose2(np.int64_t n) nogil:
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


cpdef np.float64_t fentropy(freqs):
    """Entropy of an iterable of frequencies

    Assumes every entry in the list belongs to a different class. The resulting
    value is *not* normalized by N. Also note that the entropy value is
    calculated using natural base, which may not be what you want, so you may
    need to normalized it with log(base).

    The 'freqs' parameter is expected to be an list or tuple-like iterable.
    For convenience, it can also be a dict/mapping type, in which case its
    values will be used to calculate entropy.

    """

    # The Cython version of this method is upt to 50x faster on large arrays
    # than pure Python implementation. The speed-up is almost solely due to
    # cdef-ing ``f`` variable.

    cdef np.float64_t f, s, sum_f_logn_f

    if isinstance(freqs, Mapping):
        freqs = freqs.itervalues()

    s = 0.0
    sum_f_logn_f = 0.0
    for f in freqs:
        if f != 0.0:
            s += f
            sum_f_logn_f += f * log(f)

    # ensure non-negative
    return max(0.0, s * log(s) - sum_f_logn_f)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.float64_t emi_from_margins(
    np.ndarray[np.int64_t, ndim=1, mode='c'] a,
    np.ndarray[np.int64_t, ndim=1, mode='c'] b):
    """Calculate Expected Mutual Information given margins of RxC table

    For the sake of numeric precision, the resulting value is *not* normalized
    by N.

    License: BSD 3 clause

    .. codeauthor:: Robert Layton <robertlayton@gmail.com>
    .. codeauthor:: Corey Lynch <coreylynch9@gmail.com>
    .. codeauthor:: Eugene Scherba <escherba@gmail.com>

    """
    # List of changes (Eugene Scherba, 10/2/2015):
    #
    # 1) Removed/rewritten the lines that were creating RxC intermediate NumPy
    # arrays which resulted in the O(n^2) memory requirement. Instead, the
    # intermediate values are now calculated inside the loop, which may be
    # slightly less efficient for small data sizes, but has huge advantages for
    # large or even moderately sized data. This change reduces the memory
    # requirements of this code from O(n^2) to O(n).
    #
    # 2) Removed normalization by N from the calculation. It is actually not
    # needed to normalize by N if we also don't normalize the input MI value
    # (in the calculation of the adjusted score which is ``MI - E(MI) / MI_max
    # - E(MI)``, the N value cancels out). Not normalizing the EMI calculation
    # by N avoids having to perform lots of tiny floating point increments to
    # the EMI aggregate value and thus improves numeric accuracy, especially
    # for small values of EMI.
    #
    # 3) The inner loop now directly calls ``sklearn_lgamma`` instead of relying
    # on the ``lgamma`` wrapper, resulting in 15-20% speed improvement.  The
    # wrapper is unnecessary in the inner loop as the loop parameters guarantee
    # that the values passed to the log-gamma function can never be negative.
    #

    cdef Py_ssize_t R, C, i, j, nij

    cdef np.int64_t N, N1, N3, max_ab, ai_1, bj_1, N3_ai_1, N3_ai_bj_1

    cdef np.float64_t emi, log_ai, log_ab_outer_ij, outer_sum, gln_ai_Nai_Ni

    cdef np.ndarray[np.float64_t, ndim=1, mode='c'] \
        log_a, log_b, log_Nnij, nijs, gln_ai_Nai_N, gln_b_Nb, gln_nij

    cdef np.ndarray[np.int64_t, ndim=1, mode='c'] \
        a1, b1

    log_a = np.log(a)
    log_b = np.log(b)

    R = len(a)
    C = len(b)

    N = np.sum(a)
    if N != np.sum(b):
        raise ValueError("Sums of row and column margins must be equal")

    # There are three major terms to the EMI equation, which are multiplied to
    # and then summed over varying nij values.

    # term1 is nijs.
    # While nijs[0] will never be used, having it simplifies the indexing.
    max_ab = max(<np.int64_t>np.max(a), <np.int64_t>np.max(b))
    nijs = np.arange(0LL, max_ab + 1LL, dtype=np.float64)
    nijs[0] = 1.0  # Stops divide by zero warnings. As its not used, no issue.

    # term2 is log((N*nij) / (a a b)) == log(N * nij) - log(a * b)
    # term2 uses log(N * nij)
    log_Nnij = np.log(N) + np.log(nijs)

    # term3 is large, and involved many factorials. Calculate these in log
    # space to stop overflows.
    N1 = N + 1LL
    N3 = N + 3LL

    a1 = a + 1LL
    b1 = b + 1LL
    gln_ai_Nai_N = gammaln(a1) + gammaln(N1 - a) - gammaln(N1)
    gln_b_Nb = gammaln(b1) + gammaln(N1 - b)
    gln_nij = gammaln(nijs + 1.0)

    # emi itself is a summation over the various values.
    emi = 0.0
    for i in xrange(R):
        ai_1 = a1[i]
        log_ai = log_a[i]
        gln_ai_Nai_Ni = gln_ai_Nai_N[i]
        N3_ai_1 = N3 - ai_1
        for j in xrange(C):
            bj_1 = b1[j]
            log_ab_outer_ij = log_ai + log_b[j]
            outer_sum = gln_ai_Nai_Ni + gln_b_Nb[j]
            N3_ai_bj_1 = N3_ai_1 - bj_1

            for nij in xrange(max(1LL, 1LL - N3_ai_bj_1), min(ai_1, bj_1)):
                # Numerators are positive, denominators are negative.
                emi += (
                    <np.float64_t>nij                    # term1
                    * (log_Nnij[nij] - log_ab_outer_ij)  # term2
                    * exp(outer_sum                      # term3
                        - gln_nij[nij]
                        - sklearn_lgamma(ai_1 - nij)
                        - sklearn_lgamma(bj_1 - nij)
                        - sklearn_lgamma(nij + N3_ai_bj_1)
                          )
                )
    return emi
