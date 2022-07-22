import math
import accupy
import numpy as np
import dace

from util import generate_split_set
from summation import naive_summation

def test_naive_accuracy():
    N_RUNS = 100
    NS = [10, 100, 1000, 10000, 100000]
    TINY_PERCENTSS = [.5, 1.0, 10.0, 50.0, 90.0, 99.0, 99.5]

    print('======================================')
    print('Testing naive sum accuracy')
    print('--------------------------------------')
    for tiny_percent in TINY_PERCENTSS:
        print('Testing with ' + str(tiny_percent) + '% small numbers')
        for n in NS:
            print('  For n=' + str(n))
            errors = []
            for _ in range(N_RUNS):
                dataset = generate_split_set(n, tiny_percent)
                sorted = np.sort(dataset)
                naive = naive_summation(sorted)
                accurate = math.fsum(sorted)
                err = abs(naive - accurate)
                errors.append(err)
            print('    Median error: ' + str(np.median(errors)))
    print('======================================')

if __name__=='__main__':
    pass
    #test_naive_accuracy()
