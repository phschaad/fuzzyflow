import dace

N = dace.symbol('N')
M = dace.symbol('M')

@dace.program
def prog(A: dace.float32[N, M], B: dace.float32[N, M], C: dace.float32[N, M]):
    for i, j in dace.map[0:N, 0:M]:
        tmp1 = dace.define_local_scalar(dace.float32)
        tmp2 = dace.define_local_scalar(dace.float32)
        tmp3 = dace.define_local_scalar(dace.float32)
        tmp4 = dace.define_local_scalar(dace.float32)
        tmp5 = dace.define_local_scalar(dace.float32)
        tmp6 = dace.define_local_scalar(dace.float32)

        with dace.tasklet:
            in1 << A[i, j]
            in2 << B[i, j]
            out = in1 + in2
            out >> tmp1

        with dace.tasklet:
            in1 << tmp1
            out1 = 3 * in1
            out2 = in1 / 2
            out3 = out1 - 50
            out4 = max(0, out3)
            out1 >> tmp2
            out2 >> tmp3
            out3 >> tmp4
            out4 >> tmp5

        with dace.tasklet:
            in1 << tmp2
            in2 << tmp4
            out1 = in1 + in2
            out1 >> tmp6

        with dace.tasklet:
            in1 << tmp3
            in2 << tmp5
            in3 << tmp6
            out1 = in1 + in2 + in3
            out1 >> C[i, j]


#prog.to_sdfg().view()
#
#sdfg = dace.SDFG('mergesdfg')
#sdfg.add_array('A', [N], dace.float32)
#sdfg.add_array('B', [N], dace.float32)
#sdfg.add_array('C', [N], dace.float32)
#state = sdfg.add_state('rootstate')
#state.add_mapped_tasklet(
#    'addall', dict(i='0:N'), dict(in1=dace.Memlet('A[i]')), 'out1 = in1 + 4',
#    dict(out1=dace.Memlet('B[i]')), external_edges=True
#)
#state.add_mapped_tasklet(
#    'multall', dict(i='0:N'), dict(in1=dace.Memlet('B[i]')), 'out1 = in1 * 2',
#    dict(out1=dace.Memlet('C[i]')), external_edges=True
#)

prog.to_sdfg().save('volume_test.sdfg')
