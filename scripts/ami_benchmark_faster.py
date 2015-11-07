import os
import numpy as np
import sys
import cPickle as pickle
from IPython import get_ipython
from lsh_hdc.metrics import adjusted_mutual_info_score
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


print "Sanity check:"
print "    AMI = {}".format(adjusted_mutual_info_score(ltrue, lpred))

print "Running test 1/3..."
ipython.magic("timeit adjusted_mutual_info_score(ltrue, lpred)")
print "Running test 2/3..."
ipython.magic("timeit adjusted_mutual_info_score(ltrue, lpred)")
print "Running test 3/3..."
ipython.magic("timeit adjusted_mutual_info_score(ltrue, lpred)")
