#!/usr/bin/env python2

import sys
import json
import matplotlib.pyplot
from os.path import basename
from argparse import ArgumentParser, FileType
from itertools import imap
from collections import Counter
from functools import partial
from funcy import compose
from lsh_hdc.stats import get_roc_summaries, mplot_roc_curves

parser = ArgumentParser()
parser.add_argument('--clusters', nargs='+', type=str, required=True,
                    help="File(s) containig clusterings")
parser.add_argument('--labels', nargs='*', type=str, required=False,
                    default=[], help='Labels for clusterings')
parser.add_argument('--ground', required=False, type=FileType('r'),
                    default=sys.stdin,
                    help="Impermium-tagged file for comparison")
args = parser.parse_args()

if args.labels:
    assert len(args.labels) == len(args.clusters)
    names = args.labels
else:
    names = map(basename, args.clusters)


def has_tag(tag, json_obj):
    imp_section = json_obj.get('impermium', {}) or {}
    imp_result = imp_section.get('result', {}) or {}
    imp_tags = imp_result.get('tag_details', {}) or {}
    return tag in imp_tags


def get_label(obj):
    return obj['object']['post_id']


is_bulk = partial(has_tag, 'bulk')

# consuming ground truth set from stdin -- can use multiple clusterings

# memoize cluster set
all_neighbors = []
for filename in args.clusters:
    with open(filename, 'r') as clusters:
        neighbors = Counter()
        all_neighbors.append(neighbors)
        for cluster in imap(compose(set, json.loads), clusters):
            cluster_size = len(cluster)
            if cluster_size > 1:
                num_neighbors = cluster_size - 1
                for label in cluster:
                    neighbors[label] += num_neighbors

rocs = get_roc_summaries(
    imap(json.loads, args.ground),
    level_getters=[compose(n.__getitem__, get_label)
                   for n in all_neighbors],
    ground_pos=is_bulk)

result = json.dumps(dict(
    metrics=dict(
        auc_scores={name: roc.get_auc_score()
                    for name, roc in zip(names, rocs)},
        items_clustered=dict(zip(names, map(len, all_neighbors)))
    )
))

print result

mplot_roc_curves(matplotlib.pyplot, rocs, names, auc=True)
