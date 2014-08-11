#!/usr/bin/env python

import fileinput
from lsh_hdc.utils import URLFinder
from itertools import chain

import re
from functools import partial


class URLFinder2(object):
    # also see IETF spec regex:
    # r'(([^\s:/?#]+):)(//([^\s/?#]*))?([^\s?#]*)(\\?([^\s#]*))?(#([^\s#]*))'
    # http://www.ietf.org/rfc/rfc3986.txt

    # for list of valid TLDs, see:
    # http://data.iana.org/TLD/tlds-alpha-by-domain.txt

    _find_urls = partial(re.findall, re.compile(
        'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
        re.IGNORECASE | re.UNICODE
    ))

    def find_urls(self, text):
        for t in self._find_urls(text):
            yield t


class URLFinder3(object):
    # also see IETF spec regex:
    # r'(([^\s:/?#]+):)(//([^\s/?#]*))?([^\s?#]*)(\\?([^\s#]*))?(#([^\s#]*))'
    # http://www.ietf.org/rfc/rfc3986.txt

    # for list of valid TLDs, see:
    # http://data.iana.org/TLD/tlds-alpha-by-domain.txt

    _find_urls = partial(re.findall, re.compile(
        ur'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?\xab\xbb\u201c\u201d\u2018\u2019]))',
        re.IGNORECASE | re.UNICODE
    ))

    def find_urls(self, text):
        for t in self._find_urls(text):
            yield t


uf1 = URLFinder()
uf2 = URLFinder2()
uf3 = URLFinder2()


def main():

    for line in fileinput.input():
        urls1 = list(uf1.find_urls(line))
        urls2 = list(uf2.find_urls(line))
        urls3 = list(uf3.find_urls(line))
        print line
        if len(list(chain(urls1, urls2, urls3))) > 0:
            for u in urls1:
                print "1", u.string
            for u in urls2:
                print "2", u
            for u in urls3:
                print "3", u
            print

if __name__ == '__main__':
    main()
