#!/usr/bin/env python

from math import sqrt
from tests.test_files import TestFiles
import json


def prime_numbers(mlim=100):
    """
    Prime number generator
    """
    for num in range(2, mlim + 1):
        if all(num % i != 0 for i in range(2, int(sqrt(num)) + 1)):
            yield num


counter = 0
for prime in prime_numbers(100000):
    if not (counter % 10):
        print "Working with prime {}".format(prime)
        results = TestFiles.run_simulated_manually(
            'data/simulated.txt',
            universe_size=prime
        )
        c = results['stats']
        ti = results['uindex']
        print json.dumps(dict(
            stats=c.dict(),
            ratios=dict(
                precision=c.get_precision(),
                recall=c.get_recall()
            ),
            ti=ti
        ))
    counter += 1
