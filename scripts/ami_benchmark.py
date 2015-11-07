"""

To benchmark the proposed implementation:

::

    ipython scripts/ami_benchmark.py -- --implementation proposed


To benchmark Scikit-Learn implementation:

::

    ipython scripts/ami_benchmark.py -- --implementation sklearn


"""

import os
import sys
import argparse
import numpy as np
import cPickle as pickle
from IPython import get_ipython


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--implementation', type=str, choices=['sklearn', 'proposed', 'oo'],
                        default='proposed', help='which implementation to benchmark')
    parser.add_argument('--num', type=int, default=3, help='how many tests to run')
    parser.add_argument('--max_classes', type=int, default=500, help='maximum number of classes')
    parser.add_argument('--max_clusters', type=int, default=500, help='maximum number of clusters')
    parser.add_argument('--num_labels', type=int, default=20000, help='sample size (number of labels)')
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
    ltrue = np.random.randint(low=0, high=ARGS.max_classes, size=(ARGS.num_labels,))
    lpred = np.random.randint(low=0, high=ARGS.max_clusters, size=(ARGS.num_labels,))
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

for idx in xrange(ARGS.num):
    print "Running test {}/{}...".format(idx + 1, ARGS.num)
    ipython.magic("timeit " + line)
