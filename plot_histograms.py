#!/usr/bin/env python2

from test.utils import read_json_file
from itertools import islice
import matplotlib.pyplot as plt

for o in islice(read_json_file('times.txt'), 10):
    times = o.get('times', [])
    if times:
        med = int(o['o_med'])
        times_adj = map(lambda t: (float(t) - float(med)) / 3600.0, times)
        plt.hist(times_adj, bins=20)
        plt.show()
