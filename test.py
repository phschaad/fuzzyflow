import numpy as np
import dace

from util import generate_dataset, generate_split_set

def naive_summation(data):
    res = 0.0
    for i in range(len(data)):
        res += data[i]
    return res

def kahanSum(fa):
    sum = 0.0
    c = 0.0
    for f in fa:
        y = f - c
        t = sum + y
        c = (t - sum) - y
        sum = t
    return sum

serial_sdfg = dace.SDFG.from_file('_dacegraphs/summation.sdfg')
parallel_sdfg = dace.SDFG.from_file('_dacegraphs/parallelized_summation.sdfg')

'''
sorted_data, _, _ = generate_dataset(N, TINY_PERCENT)

print(sorted_data)

#res_serial = []
#res_parallel = 0.0
res_serial = serial_sdfg(data=sorted_data, N=N)
res_parallel = parallel_sdfg(data=sorted_data, N=N)
res_naive = naive_summation(sorted_data)
res_numpy = np.sum(sorted_data)
res_khan = kahanSum(sorted_data)
res_khanReduced = kahanSumReduced(sorted_data)

print(res_serial[0])
print(res_parallel[0])
print(res_naive)
print(res_numpy)
print(res_khan)
print(res_khanReduced)

err = abs(res_serial[0] - res_parallel[0])
np_err = abs(res_serial[0] - res_numpy)
naive_err = abs(res_serial[0] - res_naive)
naive_v_khan_err = abs(res_naive - res_khan)

print('Error: ' + str(err))
print('NP-Error: ' + str(np_err))
print('Naive-Error: ' + str(naive_err))
print('Khan vs. Naive Error: ' + str(naive_v_khan_err))
print('Kahan orig v reduced: ' + str(abs(res_khan - res_khanReduced)))
'''

def test_naive_kahan():
    n = 1000
    tiny_perc = 10.0
    runs = 100
    errors = []
    for i in range(runs):
        dataset = generate_split_set(n, tiny_perc)
        sorted = np.sort(dataset)
        print(sorted)
        naive = naive_summation(sorted)
        kahan = kahanSum(sorted)
        err = abs(naive - kahan)
        errors.append(err)
    print('Median error: ' + str(np.median(errors)))

if __name__=='__main__':
    N = 1000
    TINY_PERCENT = 10.0
    test_naive_kahan()
    pass
