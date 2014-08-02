#!/usr/bin/env python2

import sys
import argparse
import json
from itertools import imap
from functools import partial
from lsh_hdc.stats import get_stats

parser = argparse.ArgumentParser()
parser.add_argument('--imperm', required=True, type=argparse.FileType('r'),
                    help="Impermium-tagged file for comparison")
parser.add_argument('infile', nargs='?', type=argparse.FileType('r'),
                    default=sys.stdin, help="input file in MAC log format")
args = parser.parse_args()


def has_tag(tag, json_obj):
    imp_section = json_obj.get('impermium', {}) or {}
    imp_result = imp_section.get('result', {}) or {}
    imp_tags = imp_result.get('tag_details', {}) or {}
    return tag in imp_tags

is_bulk = partial(has_tag, 'bulk')

positives = set()
for json_obj in imap(json.loads, args.imperm):
    id_ = json_obj['object']['post_id']
    if is_bulk(json_obj):
        positives.add(id_)

# get stats
clusters = imap(json.loads, args.infile)
results = dict(stats=get_stats(clusters,
                               lambda lbl: lbl in positives))

c = results['stats']
print json.dumps(dict(
    stats=c.dict(),
    num_clusters=c.meta['num_clusters'],
    ratios=dict(
        f1_score=c.get_f1_score(),
        precision=c.get_precision(),
        recall=c.get_recall()
    )
))
