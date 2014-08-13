# -*- coding: utf-8 -*-
import regex as re
import random
import operator
import json
import string
from collections import namedtuple
from functools import partial
from itertools import imap
from pymaptools.utils import read_text_file
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

# Content of URLComponents:
# 0                          1          2       3      4        5
# [('http://t.co:80/path#a', 'http://', 't.co', ':80', '/path', '#a')]
URLComponents = namedtuple('URLComponents',
                           'string scheme authority port path anchor')


class URLFinder(object):
    # also see IETF spec regex:
    # r'(([^\s:/?#]+):)(//([^\s/?#]*))?([^\s?#]*)(\\?([^\s#]*))?(#([^\s#]*))'
    # http://www.ietf.org/rfc/rfc3986.txt

    # for list of valid TLDs, see:
    # http://data.iana.org/TLD/tlds-alpha-by-domain.txt

    # For examples of similar patterns:
    # http://daringfireball.net/2010/07/improved_regex_for_matching_urls
    # https://gist.github.com/uogbuji/705383
    # https://gist.github.com/gruber/249502
    # http://alanstorm.com/url_regex_explained
    # http://mathiasbynens.be/demo/url-regex

    RE_URL_FINDER = re.compile(
        ur"""
    (
        ((?:https?|ftp):\/\/)       # scheme
        (                           # begin authority
            (?:
                (?:
                    [a-z0-9]        # first char of domain component, no hyphen
                    [a-z0-9-]*      # middle of domain component
                    (?<!-)          # last char cannot be a hyphen
                    \.
                )+
                (?:%(tlds)s)\b      # top-level domain
            )
            |                       # or
            (?:(?:[0-9]{1,3}\.){3}[0-9]{1,3})    # IP address
        )                           # end authority
            (:[0-9]+)?              # optional port
        (
            # (?:\/%(qvalid)s*/?)*                  path
            (?:                     # One or more:
                [^\s\(\)<>]+        # Run of non-space, non-()<>
                |                                       # or
                \((?:[^\s()<>]+|(?:\([^\s()<>]+\)))*\)  # balanced parens,
                                                        # up to 2 levels
            )*
            (?:\?(?:%(qvalid)s+=%(qvalid)s+&)*  # GET parameters
            %(qvalid)s+=%(qvalid)s+)?           # last GET parameter
        )
        (
            \#(?u)[^\(\)\s\#%%\[\]\{\}\\"<>]*
        )?                                      # optional anchor
        (?<!(?u)[\s`\!\[\]{};:'".,<>\?«»“”‘’])  # avoid punctuation at the end
    )
    """ %
        dict(
            qvalid=ur"[a-z0-9$-_.+!*'(),%]",
            tlds='|'.join(read_text_file(__name__, 'tlds-alpha-by-domain.txt'))
        ),
        re.IGNORECASE | re.VERBOSE | re.UNICODE
    )

    URL_SHORTENERS = set(read_text_file(__name__, 'url_shortener_domains.txt'))

    def find_urls(self, s, overlapped=True):
        urls = []
        for url_ in re.findall(self.RE_URL_FINDER, s, overlapped=overlapped):
            url = URLComponents(*url_)
            # prevent run-on URLs
            if len(urls) > 0 and urls[-1].string.endswith(url.string) and \
                    not urls[-1].string == url.string:
                prev_urls = re.findall(self.RE_URL_FINDER,
                                       urls[-1].string[:-len(url.string)],
                                       overlapped=False)
                if len(prev_urls) > 0:
                    urls[-1] = URLComponents(*prev_urls[0])
            urls.append(url)
        return urls

    def is_shortener(self, url):
        return url.authority in self.URL_SHORTENERS

    def find_shortener_urls(self, text):
        """Find all urls and return those whose domain is one of the
        known URL shorteners"""
        for url in self.find_urls(text):
            if self.is_shortener(url):
                yield url


class HTMLNormalizer(Normalizer):

    # map zero-width characters to nothing
    TRANSLATE_MAP = {k: None for k in (
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

    # translate all Unicode spaces to regular spaces
    _normalize_whitespace = partial(re.compile(ur'(?u)\s+').sub, u' ')

    def __init__(self, lowercase=True, normalize_shortener_urls=True):
        self.lowercase = lowercase
        self.html_parser = HTMLParser()
        self.url_finder = URLFinder()
        self.normalize_shortener_urls = normalize_shortener_urls

    def normalize(self, text, input_encoding='UTF-8'):
        """
        :param text: Input text
        :rtype text: str, unicode
        :return: normalized text
        :rtype: unicode
        """

        # translate is only available for Unicode strings so we convert
        # plain text to Unicode
        if not isinstance(text, unicode):
            text = text.decode(input_encoding)

        html_parser = self.html_parser
        text = html_parser.unescape(html_parser.unescape(text))
        text = clean_html(text)

        if text != u'':
            text = text.translate(self.TRANSLATE_MAP)
            text = self._normalize_whitespace(text)
            if self.lowercase:
                text = text.lower()

        if self.normalize_shortener_urls:
            for url in self.url_finder.find_shortener_urls(text):
                authority_tok = (url.authority + url.port).replace(u'.', u'_')
                replacement = \
                    u' ' + url.scheme + authority_tok + \
                    u'/' + authority_tok + '_PATH '
                text = text.replace(url.string, replacement)

        return text


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


class JsonRepr(object):

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
