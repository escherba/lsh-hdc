#!/usr/bin/env python2

import yaml
from pymaptools import UnionFind
from mrjob import protocol as mr_protocol
from mrjob.job import MRJob
from mrjob.step import MRStep
from operator import itemgetter
from collections import Counter
from lsh_hdc.cluster import HDClustering
from content_rules import ContentFilter
from pkg_resources import resource_filename

with open(resource_filename(__name__, 'mac.yaml'), 'r') as fh:
    mac_cfg = yaml.load(fh)

hdc = HDClustering(cfg=mac_cfg['model'],
                   content_filter=ContentFilter(),
                   get_body=itemgetter('content'),
                   get_label=itemgetter('post_id'),
                   get_prefix=itemgetter('user_id'))


class MRCluster(MRJob):

    INPUT_PROTOCOL = mr_protocol.JSONValueProtocol
    INTERNAL_PROTOCOL = mr_protocol.JSONProtocol
    OUTPUT_PROTOCOL = mr_protocol.JSONValueProtocol

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

        # filter out all groupings of size one or smaller
        if len(rvals) > 1:
            yield rkey, rvals

    def ab_mapper(self, lsh, data):
        sorted_data = sorted(data)  # sort on labels

        # key is label + sketch pair
        for t in sorted_data:
            lbl = t[0]
            val = filter(lambda x: x[0] != lbl, sorted_data)
            yield t, (lsh, val)

    def uniq_reducer(self, key, vals):
        uniq = set()
        for lsh, tuples in vals:
            uniq.add(tuple(tuple(t) for t in tuples))
        for val in uniq:
            yield key, val

    def cluster_mapper(self, key, data):

        # emit tuple -> (None, remaining_tuples) for each tuple in
        # clustered groups
        if key is None:
            clustered_sketches = list(data)
            if len(clustered_sketches) > 1:
                for item in clustered_sketches:
                    lbl = item[0]
                    val = filter(lambda x: x[0] != lbl, clustered_sketches)
                    yield item, (None, val)
        else:
            yield key, data

    def cluster_combiner(self, key, vals):

        clustered_sketches = dict()
        for val in vals:
            lsh, tuples = val
            if lsh is None:
                clustered_sketches.update(tuples)
            else:
                yield key, val
        if len(clustered_sketches) > 0:
            yield key, (None, clustered_sketches.items())

    def cluster_reducer(self, key, vals):

        sketch_dist = hdc.sketch_dist_fn
        max_dist = hdc.max_dist
        min_support = hdc.min_support

        label, sketch = key
        unclustered_counter = Counter()
        unclustered_sketches = dict()
        clustered_sketches = {label: sketch}
        unclustered = []
        for val in vals:
            lsh, tuples = val
            if lsh is None:
                clustered_sketches.update(tuples)
            else:
                unclustered_counter.update(t[0] for t in tuples)
                unclustered_sketches.update(tuples)
                unclustered.append(val)

        # from unclustered labels, obtain new labels to cluster
        is_close = lambda t: \
            unclustered_counter[t[0]] >= min_support and \
            sketch_dist(sketch, t[1]) <= max_dist
        filtered_labels = set(t[0] for t in unclustered_sketches.iteritems()
                              if is_close(t))

        # merge new labels with already linked clusters
        clustered_sketches.update((lbl, unclustered_sketches[lbl])
                                  for lbl in filtered_labels)

        if len(clustered_sketches) > 1:
            yield None, clustered_sketches.items()

        # remove all the clustered labels from the unclustered groups and
        # emit unclustered groups on a rotated key
        for lsh, tuples in unclustered:
            remaining = filter(lambda t: t[0] not in clustered_sketches,
                               tuples)
            if len(remaining) > 1:
                yield remaining[0], (lsh, remaining[1:])

    def union_mapper(self, key, val):
        """ emit only 'good' clusters """
        if key is None:
            yield key, val

    def union_reducer(self, key, vals):
        """ find connected components """
        uf = UnionFind()
        for tuples in vals:
            uf.union(*[t[0] for t in tuples])
        for i, s in enumerate(uf.sets()):
            yield i, s

    def steps(self):
        return [
            MRStep(mapper=self.lsh_mapper,
                   combiner=self.lsh_combiner,
                   reducer=self.lsh_reducer),
            MRStep(mapper=self.ab_mapper,
                   combiner=self.cluster_combiner,
                   reducer=self.cluster_reducer),
            MRStep(mapper=self.cluster_mapper,
                   combiner=self.cluster_combiner,
                   reducer=self.cluster_reducer),
            MRStep(mapper=self.cluster_mapper,
                   combiner=self.cluster_combiner,
                   reducer=self.cluster_reducer),
            MRStep(mapper=self.cluster_mapper,
                   combiner=self.cluster_combiner,
                   reducer=self.cluster_reducer),
            MRStep(mapper=self.union_mapper,
                   reducer=self.union_reducer)
        ]


if __name__ == '__main__':
    MRCluster.run()
