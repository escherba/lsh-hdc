"""

To benchmark the proposed implementation:

::

    ipython scripts/ami_benchmark.py -- --implementation proposed


To benchmark Scikit-Learn implementation:

::

    ipython scripts/ami_benchmark.py -- --implementation sklearn


"""

import os
import numpy as np
import sys
import cPickle as pickle
import argparse
from IPython import get_ipython


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--implementation', type=str, choices=['sklearn', 'proposed', 'oo'],
                        default='proposed', help='which implementation to benchmark')
    parser.add_argument('--num', type=int, default=3, help='how many tests to run')
    namespace = parser.parse_args(args)
    return namespace


ARGS = parse_args()
if ARGS.implementation == 'proposed':
    from lsh_hdc.metrics import adjusted_mutual_info_score
elif ARGS.implementation == 'oo':
    from lsh_hdc.metrics import ClusteringMetrics
elif ARGS.implementation == 'sklearn':
    from sklearn.metrics.cluster import adjusted_mutual_info_score
else:
    raise argparse.ArgumentError('Unknown value for --implementation')


ipython = get_ipython()
if ipython is None:
    print "No IPython"
    sys.exit(0)


PATH = "out.pickle"

if os.path.exists(PATH):
    print "Loading from pickle file"
    with open(PATH, 'r') as fh:
        ltrue, lpred = pickle.load(fh)
else:
    ltrue = np.random.randint(low=0, high=500, size=(20000,))
    lpred = np.random.randint(low=0, high=500, size=(20000,))
    print "Saving to pickle file"
    with open(PATH, 'w') as fh:
        pickle.dump((ltrue, lpred), fh, protocol=pickle.HIGHEST_PROTOCOL)


if ARGS.implementation == 'oo':
    cm = ClusteringMetrics.from_labels(ltrue, lpred)
    line = "cm.adjusted_mutual_info()"
else:
    line = "adjusted_mutual_info_score(ltrue, lpred)"


print "Sanity check:"
print "\tAMI = {}".format(eval(line))

for idx in xrange(ARGS.num, start=1):
    print "Running test {}/{}...".format(idx, ARGS.num)
    ipython.magic("timeit " + line)
