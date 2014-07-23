"""
utils.py

Various utilities
"""

import random
import operator
import json
import string
from itertools import imap


def totuple(a):
    """convert possible scalar to tuple"""
    return a if type(a) == tuple else (a,)


def tsorted(a):
    """Sort a tuple"""
    return tuple(sorted(a))


def read_json_file(file_path):
    """Open a JSON file and read contents
    :param file_path: path to file
    :type file_path: str
    :returns: iterator of JSON objects
    :rtype: collections.iterable
    """
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

    :param n: number of times to apply a function
    :type n: integer
    :param func: a function to apply
    :type func: instancemethod
    :rtype: collections.iterable
    """
    for _ in xrange(n):
        yield func(*args, **kwargs)


def lapply(n, func, *args, **kwargs):
    """Same as gapply, except returns a list

    :param n: number of times to apply a function
    :type n: integer
    :param func: a function to apply
    :type func: instancemethod
    :rtype: list
    """
    return list(gapply(n, func, *args, **kwargs))


def randset():
    """Return a random set.  These values of n and k have wide-ranging
    similarities between pairs.

    :returns: a list of integers
    :rtype: tuple
    """
    n = random.choice(range(5, 20))
    k = 10
    return tuple(set(gapply(n, random.choice, range(k))))


alphabet = string.ascii_lowercase + string.ascii_uppercase + string.digits


def randstr(n):
    """Return a random string of length n"""
    return ''.join(random.choice(alphabet) for _ in xrange(n))


def sigsim(x, y, dim):
    """Return the similarity of the two signatures
    :param x: signature 1
    :type x: object
    :param y: signature 2
    :type y: object
    :param dim: number of dimensions
    :type dim: int
    :returns: similarity between two signatures
    :rtype: float
    """
    return float(sum(imap(operator.eq, x, y))) / float(dim)


def sort_by_length(els, reverse=True):
    """Given a list of els, sort its elements by len()
    in descending order. Returns a generator

    :param els: input list
    :type els: list
    :param reverse: Whether to reverse a list
    :type reverse: bool
    :rtype: collections.iterable
    """
    return imap(first,
                sorted(((s, len(s)) for s in els),
                       key=operator.itemgetter(1), reverse=reverse))
