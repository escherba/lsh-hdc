import regex as re


class RegexTokenizer(object):

    def __init__(self, pattern=u'\\w+', ignore_case=False):
        flags = re.UNICODE
        if ignore_case:
            flags |= re.IGNORECASE
        self.tokenize = re.compile(pattern, flags).findall
