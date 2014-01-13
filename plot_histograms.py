#!/usr/bin/env python2

from itertools import islice

import matplotlib.pyplot as plt

from lsh.utils import read_json_file


for o in islice(read_json_file('examples/times.txt'), 10):
    times = o.get('timestamps', [])
    if times:
        med = int(o['timestamps_median'])
        times_adj = map(lambda t: (float(t) - float(med)) / 3600.0, times)
        plt.hist(times_adj, bins=20)
        plt.show()
