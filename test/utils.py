"""
utils.py

Testing utilities
"""

import random
import operator
import json
from itertools import imap


def read_json_file(file_path):
    with open(file_path, 'r') as json_file:
        for line in json_file:
            yield json.loads(line)


def getpropval(obj):
    """

    :return: a generator of properties and their values
    """
    return ((p, val) for p, val in ((p, getattr(obj, p)) for p in dir(obj))
            if not callable(val) and p[0] != '_')


class JsonRepr:

    def as_dict(self):
        """

        :rtype : dict
        """
        return dict(getpropval(self))

    def __repr__(self):
        """

        :rtype : str
        """
        return json.dumps(self.as_dict())

    def assign(self, o):
        for k, v in getpropval(o):
            setattr(self, k, v)


def first(it):
    """
    :returns: first element of an iterable"""
    return it[0]


def second(it):
    """
    :returns: second element of an iterable"""
    return it[1]


def last(it):
    """
    :returns: last element of an iterable"""
    return it[-1]


def gapply(n, func, *args, **kwargs):
    """Apply a generating function n times to the argument list

    :rtype: collections.iterable
    """
    for _ in xrange(n):
        yield func(*args, **kwargs)


def lapply(*args, **kwargs):
    """Same as gapply except treturn a list

    :rtype: list
    """
    return list(gapply(*args, **kwargs))


def randset():
    """Return a random set.  These values of n and k have wide-ranging
    similarities between pairs.

    :rtype: tuple
    """
    n = random.choice(range(5, 20))
    k = 10
    return tuple(set(gapply(n, random.choice, range(k))))


def sigsim(x, y, dim):
    """Return the similarity of the two signatures"""
    return float(sum(imap(operator.eq, x, y))) / float(dim)


def sort_by_length(els, reverse=True):
    """Given a list of els, sort its elements by len()
    in descending order. Returns a generator

    :rtype: collections.iterable
    """
    return imap(first,
                sorted(((s, len(s)) for s in els),
                       key=operator.itemgetter(1), reverse=reverse))

