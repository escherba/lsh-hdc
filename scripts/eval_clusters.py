#!/usr/bin/env python2

import sys
import json
import matplotlib.pyplot as plt
from os.path import basename
from argparse import ArgumentParser, FileType
from itertools import imap, izip
from collections import Counter
from functools import partial
from funcy import compose
from lsh_hdc.stats import get_roc_summaries

parser = ArgumentParser()
parser.add_argument('--clusters', nargs="+", required=False,
                    type=str, help="File(s) containig clusterings")
parser.add_argument('--ground', required=False, type=FileType('r'),
                    default=sys.stdin,
                    help="Impermium-tagged file for comparison")
args = parser.parse_args()

cluster_f = [sys.stdin] if args.clusters is None else args.clusters


def has_tag(tag, json_obj):
    imp_section = json_obj.get('impermium', {}) or {}
    imp_result = imp_section.get('result', {}) or {}
    imp_tags = imp_result.get('tag_details', {}) or {}
    return tag in imp_tags


def get_label(obj):
    return obj['object']['post_id']


def plot_roc_curves(rocs, names):
    for cf, roc in izip(names, rocs):
        plt.plot(*roc.get_axes(), label=cf)
    plt.ylabel('Recall')
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 0.2])
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.legend(loc='lower right')
    plt.show()

is_bulk = partial(has_tag, 'bulk')

# consuming ground truth set from stdin -- can use multiple clusterings

# memoize cluster set
all_neighbors = []
for cf in cluster_f:
    with open(cf, 'r') as clusters:
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

print json.dumps(dict(
    metrics=dict(
        auc_scores=[roc.get_auc_score() for roc in rocs],
    )
))
plot_roc_curves(rocs, imap(basename, args.clusters))
