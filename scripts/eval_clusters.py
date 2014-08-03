#!/usr/bin/env python2

import sys
import json
import matplotlib.pyplot as plt
from argparse import ArgumentParser, FileType
from itertools import imap
from collections import Counter
from functools import partial
from funcy import compose
from lsh_hdc.stats import describe_clusters, get_roc_summary

parser = ArgumentParser()
parser.add_argument('--clusters', required=False, type=FileType('r'),
                    help="List of clusters")
parser.add_argument('--imperm', required=False, type=FileType('r'),
                    help="Impermium-tagged file for comparison")
parser.add_argument('infile', nargs='?', type=FileType('r'),
                    default=sys.stdin, help="input file")
args = parser.parse_args()

cluster_f = args.infile \
    if args.clusters is None \
    else args.clusters

imperm_f = args.infile \
    if args.imperm is None \
    else args.imperm


def has_tag(tag, json_obj):
    imp_section = json_obj.get('impermium', {}) or {}
    imp_result = imp_section.get('result', {}) or {}
    imp_tags = imp_result.get('tag_details', {}) or {}
    return tag in imp_tags


def get_label(obj):
    return obj['object']['post_id']

is_bulk = partial(has_tag, 'bulk')

clusters = imap(json.loads, cluster_f)
if cluster_f == args.infile:

    # memoize test set
    imperm_positives = set()
    for json_obj in imap(json.loads, imperm_f):
        if is_bulk(json_obj):
            imperm_positives.add(get_label(json_obj))

    r = describe_clusters(clusters, lambda lbl: lbl in imperm_positives)

    print json.dumps(dict(
        num_clusters=r.meta['num_clusters'],
        confusion_matrix=r.dict(),
        metrics=dict(
            f1_score=r.get_f1_score(),
            precision=r.get_precision(),
            recall=r.get_recall()
        )
    ))
elif imperm_f == args.infile:

    # memoize cluster set
    neighbors = Counter()
    for cluster in imap(set, clusters):
        cluster_size = len(cluster)
        if cluster_size > 1:
            num_neighbors = cluster_size - 1
            for label in cluster:
                neighbors[label] += num_neighbors

    roc = get_roc_summary(
        imap(json.loads, imperm_f),
        get_level=compose(neighbors.__getitem__, get_label),
        ground_pos=is_bulk)

    print json.dumps(dict(
        metrics=dict(
            auc=roc.get_auc_score(),
        )
    ))
    plt.plot(*roc.get_axes())
    plt.ylabel('Recall')
    plt.xlabel('False Positive Rate')
    plt.show()
