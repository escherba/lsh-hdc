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

with open(resource_filename(__name__, 'mac-a0.yaml'), 'r') as fh:
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

    def cluster_mapper(self, key, data):

        if key is None:  # good cluster
            items = list(data)
            if len(items) > 1:
                parent = min(items)  # minimal element becomes parent
                parent_label = parent[0]
                for child in items:
                    if child[0] != parent_label:
                        yield child, (None, parent)
        else:
            yield key, data

    def cluster_combiner(self, key, vals):

        children = dict()
        for val in vals:
            lsh, child = val
            if lsh is None:  # good cluster
                children[child[0]] = child[1]
            else:
                yield key, val
        if len(children) > 0:
            for child in children.iteritems():
                yield child, (None, key)  # key becomes parent

    def cluster_reducer(self, key, vals):

        sketch_dist = hdc.sketch_dist_fn
        max_dist = hdc.max_dist
        min_support = hdc.min_support
        logical_op = hdc.sketch_operator

        parent, sketch = key
        unclustered_counter = Counter()
        unclustered_sketches = dict()
        parents = {parent: sketch}
        unclustered = []
        for lsh, val in vals:
            if lsh is None:
                parents[val[0]] = val[1]
            else:
                unclustered_counter.update(t[0] for t in val)
                unclustered_sketches.update(val)
                unclustered.append((lsh, val))

        # from unclustered labels, obtain new labels to cluster
        is_close = lambda t: \
            logical_op(unclustered_counter[t[0]] >= min_support,
                       sketch_dist(sketch, t[1]) <= max_dist)
        filtered_labels = set(t[0] for t in unclustered_sketches.iteritems()
                              if is_close(t))

        # merge new labels with already linked clusters
        parents.update((lbl, unclustered_sketches[lbl])
                       for lbl in filtered_labels)

        if len(parents) > 1:
            self.increment_counter('clustered', 'count', 1)
            yield None, parents.items()

        # remove all the clustered labels from the unclustered groups and
        # emit unclustered groups on a rotated key
        for lsh, tuples in unclustered:
            remaining = filter(lambda t: t[0] not in parents,
                               tuples)
            if len(remaining) > 1:
                self.increment_counter('unclustered', 'count', 1)
                yield remaining[0], (lsh, remaining[1:])

    def union_mapper(self, key, val):
        """ emit only 'good' clusters """
        if key is None:
            yield key, val

    def union_reducer(self, key, vals):
        """ find connected components """
        uf = UnionFind()
        for elements in vals:
            uf.union(*[el[0] for el in elements])
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
