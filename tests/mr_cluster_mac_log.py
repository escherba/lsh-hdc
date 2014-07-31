#!/usr/bin/env python2

import os
import yaml
from mrjob import protocol as mrjob_protocol
from mrjob.job import MRJob
from mrjob.step import MRStep
from operator import itemgetter
from collections import Counter
from lsh_hdc.cluster import HDClustering
from content_rules import ContentFilter

import sys

fn = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data/mac.yaml'))
sys.stderr.write("--> " + fn + "\n")
with open(fn, 'r') as fh:
    mac_cfg = yaml.load(fh)


class MRCluster(MRJob):

    INPUT_PROTOCOL = mrjob_protocol.JSONValueProtocol
    INTERNAL_PROTOCOL = mrjob_protocol.JSONProtocol
    OUTPUT_PROTOCOL = mrjob_protocol.JSONProtocol

    def __init__(self, hdc=None, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        self.hdc = hdc

    def lsh_mapper(self, _, data):
        obj = data['object']
        for pair in hdc.mapper(obj):
            yield pair

    def lsh_combiner(self, key, vals):
        ckey, cvals = hdc.reducer(key, vals)
        for cval in cvals:
            yield ckey, cval

    def lsh_reducer(self, key, vals):
        rkey, rvals = hdc.reducer(key, vals)
        if len(rvals) > 1:
            yield rkey, rvals

    def ab_mapper(self, _, data):
        sorted_data = sorted(data)
        key = sorted_data[0][0]
        yield key, sorted_data

    def cluster_reducer(self, key, vals):
        uniq = set()
        for val in vals:
            uniq.add(tuple(tuple(v) for v in val))
        for val in uniq:
            yield key, val

    def sketch_mapper(self, _, data):

        # find most commonly occuring sketch and pick one associated
        # with alphabetically lowest id

        c = Counter()
        cluster = list(data)
        for label, sketch in cluster:
            c[sketch] += 1
        label2sketch = dict(cluster)
        label_c = [(label, c[sketch]) for label, sketch in cluster]
        sorted_data = sorted(label_c, key=lambda x: (-x[1], x[0]))
        output = [(label, label2sketch[label]) for label, cnt in sorted_data]
        key = output[0][1]
        yield key, output

    def steps(self):
        return [
            MRStep(mapper=self.lsh_mapper,
                   combiner=self.lsh_combiner,
                   reducer=self.lsh_reducer),
            MRStep(mapper=self.sketch_mapper,
                   reducer=self.cluster_reducer)
        ]


if __name__ == '__main__':
    hdc = HDClustering(cfg=mac_cfg['model'],
                       content_filter=ContentFilter(),
                       get_body=itemgetter('content'),
                       get_label=itemgetter('post_id'),
                       get_prefix=itemgetter('user_id')
                       )

    mrcluster = MRCluster(hdc=hdc)
    mrcluster.run()
