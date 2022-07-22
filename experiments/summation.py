import dace

N = dace.symbol('N')

@dace.program
def naive_summation(data: dace.float64[N]):
    res = 0.0
    for i in range(N):
        res += data[i]
    return res
