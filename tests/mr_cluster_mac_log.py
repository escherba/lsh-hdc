#!/usr/bin/env python2

import yaml
import json
from mrjob import protocol as mrjob_protocol
from mrjob.job import MRJob
from operator import itemgetter
from lsh_hdc.cluster import HDClustering
from content_rules import ContentFilter
from pkg_resources import resource_string


mac_cfg = yaml.load(resource_string(__name__, 'data/mac.yaml'))
hdc = HDClustering(cfg=mac_cfg['model'],
                   content_filter=ContentFilter(),
                   get_body=itemgetter('content'),
                   get_label=itemgetter('post_id'),
                   get_prefix=itemgetter('user_id')
                   )


class MRCluster(MRJob):

    INPUT_PROTOCOL = mrjob_protocol.RawValueProtocol
    OUTPUT_PROTOCOL = mrjob_protocol.JSONValueProtocol

    def mapper(self, _, data):
        obj = json.loads(data)['object']
        for x in hdc.mapper(obj):
            yield x

    def reducer(self, key, val):
        result = hdc.reducer((key, val))
        if len(result[1]) > 1:
            yield result


mrcluster = MRCluster()
mrcluster.run()
