#!/usr/bin/env python2

import yaml
import sys
import argparse
import json
from itertools import imap
from operator import itemgetter
from lsh_hdc.cluster import HDClustering
from content_rules import ContentFilter

parser = argparse.ArgumentParser()

parser.add_argument('--config', required=True, type=argparse.FileType('r'),
                    help="YAML clustering config")
parser.add_argument('infile', nargs='?', type=argparse.FileType('r'),
                    default=sys.stdin, help="input file in MAC log format")

args = parser.parse_args()


mac_cfg = yaml.load(args.config)

data = []
positives = set()
all_objects = dict()
for json_obj in imap(json.loads, args.infile):
    obj = json_obj['object']
    data.append(obj)
    id_ = obj['post_id']
    all_objects[id_] = obj
    imp_section = json_obj.get('impermium', {}) or {}
    imp_result = imp_section.get('result', {}) or {}
    imp_tags = imp_result.get('tag_details', {}) or {}
    if 'bulk' in imp_tags:
        positives.add(id_)

hdc = HDClustering(cfg=mac_cfg['model'],
                   content_filter=ContentFilter(),
                   get_body=itemgetter('content'),
                   get_label=itemgetter('post_id'),
                   get_prefix=itemgetter('user_id')
                   )
clusters = hdc.clusters_from_iter(data)

count_clusters = sorted([(len(c), c) for c in clusters], reverse=True)
for i, (l, cluster) in enumerate(count_clusters):
    if len(cluster) > 1:
        print "===== Cluster %d (%d comments) =======" % (i, l)
        for id_ in cluster:
            print all_objects[id_]['content'].rstrip()
        print

'''

Various stats

num_clusters = len([x for x in clusters if len(x) > 1])
print "Found %d clusters" % num_clusters
print "Points not clustered: %d" % (len(data) - num_clusters)

is_label_positive = lambda lbl: lbl in positives
results = dict(stats=get_stats(clusters, is_label_positive))

c = results['stats']
recall = c.get_recall()
precision = c.get_precision()
print json.dumps(dict(
    stats=c.dict(),
    ratios=dict(
        precision=precision,
        recall=recall
    )
))
'''
