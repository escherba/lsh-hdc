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

for cluster in hdc.clusters_from_iter(data):
    if len(cluster) > 1:
        print json.dumps(cluster)
