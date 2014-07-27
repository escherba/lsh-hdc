# -*- coding: utf-8 -*-
import re
import random
import operator
import json
import string
from itertools import imap
from abc import abstractmethod
from HTMLParser import HTMLParser


class MLStripper(HTMLParser):
    def __init__(self):
        self.reset()
        self.fed = []

    def handle_data(self, d):
        self.fed.append(d)

    def handle_entityref(self, name):
        # Ignore HTML entities (already unescaped)
        self.fed.append(u'&' + name)

    def get_data(self):
        return ''.join(self.fed)


def clean_html(html):
    """Remove HTML markup from the given string."""
    html = re.sub(r"(?s)<!--(.*?)-->[\n]?", "\\1", html)
    html = re.sub(r"<!--", "", html)
    if html == '':
        return ''
    s = MLStripper()
    s.feed(html)
    return s.get_data().strip()


class Normalizer(object):
    """Abstract tokenizer interface"""

    @abstractmethod
    def normalize(self, text):
        """Tokenize text"""


class HTMLNormalizer(Normalizer):

    normalize_map = {k: None for k in (
        range(ord(u'\x00'), ord(u'\x08') + 1) +
        range(ord(u'\x0b'), ord(u'\x0c') + 1) +
        range(ord(u'\x0e'), ord(u'\x1f') + 1) +
        range(ord(u'\x7f'), ord(u'\x9f') + 1) +
        [ord(u'\uffff')] +
        [ord(u'\xad')] +
        range(ord(u'\u17b4'), ord(u'\u17b5') + 1) +
        range(ord(u'\u200b'), ord(u'\u200f') + 1) +
        range(ord(u'\u202a'), ord(u'\u202d') + 1) +
        range(ord(u'\u2060'), ord(u'\u2064') + 1) +
        range(ord(u'\u206a'), ord(u'\u206f') + 1) +
        [ord(u'\ufeff')]
    )}

    def __init__(self, lowercase=True):
        self.lowercase = lowercase
        self.html_parser = HTMLParser()

    def normalize(self, soup):
        """
        :param text: Input text
        :return: str, unicode
        :return: normalized text
        :rtype: str, unicode
        """
        html_parser = self.html_parser
        unescaped_soup = html_parser.unescape(html_parser.unescape(soup))

        text = clean_html(unescaped_soup)

        cleaned = text if text == '' else text.translate(self.normalize_map)

        return cleaned.lower() if self.lowercase else cleaned


class Tokenizer(object):
    """Abstract tokenizer interface"""

    @abstractmethod
    def tokenize(self, text):
        """Tokenize text"""


class RegexTokenizer(Tokenizer):
    def __init__(self, pattern=None):
        if pattern is None:
            """
            pattern = ur'(?u)\w+'
            pattern = ur'(?:\B[#@$£€¥₩฿])?(?u)\w+(?:[%\+]\B)?'
            pattern = ur'''
                        (?:                # Either URL
                        http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+
                        |                  # or
                        (?:\B[#@$£€¥₩฿])?  # (preceded by optional pound-, at-, or currency signs)
                        (?u)\w+            # a Unicode word
                        (?:[%\+]\B)?       # optionally followed by percentage or plus signs
                        )
                        '''
            """
            pattern = ur'(?u)\w+'
        self.r = re.compile(pattern, (re.VERBOSE | re.UNICODE))

    def tokenize(self, text):
        return self.r.findall(text)


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
