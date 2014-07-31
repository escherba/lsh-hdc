#!/usr/bin/env python2

import yaml
from mrjob import protocol as mrjob_protocol
from mrjob.job import MRJob
from operator import itemgetter
from lsh_hdc.cluster import HDClustering
from content_rules import ContentFilter
from pkg_resources import resource_string


class MRCluster(MRJob):

    INPUT_PROTOCOL = mrjob_protocol.JSONValueProtocol
    OUTPUT_PROTOCOL = mrjob_protocol.JSONValueProtocol

    def __init__(self, hdc=None, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        self.hdc = hdc

    def mapper(self, _, data):
        obj = data['object']
        for pair in hdc.mapper(obj):
            yield pair

    def combiner(self, key, vals):
        ckey, cvals = hdc.reducer(key, vals)
        for cval in cvals:
            yield ckey, cval

    def reducer(self, key, vals):
        rkey, rvals = hdc.reducer(key, vals)
        if len(rvals) > 1:
            yield rkey, rvals


if __name__ == '__main__':
    mac_cfg = yaml.load(resource_string(__name__, 'data/mac.yaml'))
    hdc = HDClustering(cfg=mac_cfg['model'],
                       content_filter=ContentFilter(),
                       get_body=itemgetter('content'),
                       get_label=itemgetter('post_id'),
                       get_prefix=itemgetter('user_id')
                       )

    mrcluster = MRCluster(hdc=hdc)
    mrcluster.run()
