#!/usr/bin/env python2

import json
from sys import argv


def read_json(filename):
    with open(filename, 'r') as fh:
        for line in fh:
            yield json.loads(line)


all_objs = {obj['_id']: obj for obj in read_json(argv[1])}

for value in all_objs.itervalues():
    print json.dumps(value)
