__author__ = 'escherba'

import re
from logging import getLogger

LOG = getLogger(__name__)


class ContentFilter(object):

    def __init__(self):
        self.word_regex = re.compile(ur'(?u)\w+', re.UNICODE)

    def match_instagram(self, obj):
        """Instagram rule: drop objects with empty content
        and matching id
        """
        content = obj[u'content']
        post_id = obj[u'post_id']
        return len(list(self.word_regex.findall(content))) == 0 and \
            post_id.endswith(u'@instagram.com')

    def accept(self, obj):
        """Process an input object according to our rules"""

        if self.match_instagram(obj):
            LOG.info("[bulk] %s (accept)", obj[u'post_id'])
            #Statsd.incr("model.bulk.filter.instagram", 1)
            return True

        # TODO: add more rules

        return False
