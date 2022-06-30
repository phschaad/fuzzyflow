import dace

N = dace.symbol('N')

@dace.program
def summation(data: dace.float64[N]):
    res = 0.0
    for i in range(N):
        res += data[i]
    return res


@dace.program
def kahanSum(data: dace.float64[N]):
    sum = 0.0
    # Variable to store the error
    c = 0.0
    for i in range(N):
        oldsum = sum
        sum = oldsum + (data[i] - c)
        c = ((oldsum + (data[i] - c)) - oldsum) - (data[i] - c)
    return sum


kahanSum.to_sdfg().save('_dacegraphs/kahan.sdfg')
