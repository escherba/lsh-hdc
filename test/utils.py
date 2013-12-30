"""
utils.py

Testing utilities
"""

import random
import operator
from itertools import imap


def first(it):
    """return first element of an iterable"""
    return it[0]


def second(it):
    """return second element of an iterable"""
    return it[1]


def last(it):
    """return last element of an iterable"""
    return it[-1]


def gapply(n, func, *args, **kwargs):
    """Apply a generating function n times to the argument list"""
    for _ in xrange(n):
        yield func(*args, **kwargs)


def lapply(*args, **kwargs):
    """Same as gapply except treturn a list"""
    return list(gapply(*args, **kwargs))


def randset():
    """Return a random set.  These values of n and k have wide-ranging
    similarities between pairs.
    """
    n = random.choice(range(5, 20))
    k = 10
    return tuple(set(gapply(n, random.choice, range(k))))


def sigsim(x, y, dim):
    """Return the similarity of the two signatures"""
    return float(sum(imap(operator.eq, x, y))) / float(dim)


def uniq_rev_index(d):
    """ (key -> [ value ]) -> (value -> key)

    Convert key -> [ value ] mapping to
    value -> key. Assume unique values
    (no value assigned to more than one key)
    """

    result = {}
    for key, values in d.iteritems():
        for value in values:
            if value in result:
                raise KeyError
            else:
                result[value] = key
    return result


def sort_by_length(els, reverse=True):
    """Given a list of els, sort its elements by len()
    in descending order. Returns a generator
    """
    return imap(first,
                sorted(((s, len(s)) for s in els),
                       key=operator.itemgetter(1), reverse=reverse))

