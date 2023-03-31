import os
import re
from dace.sdfg import SDFG
from dace.transformation.optimizer import SDFGOptimizer

import fuzzyflow as ff


IGNORE_LIST = [
    'ElementWise*',
    'FPGA*',
    'GPU*',
    'MPI*',
    'ReductionNOperation',
    'OuterProductOperation',
    'CopyToDevice',
    'LoopDetection',

    'NestSDFG',
]

GRAPH_IGNORE_LIST = [
    'adi.sdfg',
    'arc_distance.sdfg',
    'atax.sdfg',
    'azimint_hist.sdfg',
    'azimint_naive.sdfg',
    'bicg.sdfg',
    'cavity_flow.sdfg',
    'channel_flow.sdfg',
    'cholesky.sdfg',
    'cholesky2.sdfg',
    'compute.sdfg',
    'contour_integral.sdfg',
    'conv2d_bias.sdfg',
    'correlation.sdfg', # Done
    'covariance.sdfg',
    'crc16.sdfg',
    'deriche.sdfg',
    'doitgen.sdfg',
    'durbin.sdfg',
    'fdtd_2d.sdfg',
    'floyd_warshall.sdfg', # Done
    'gemm.sdfg',
    'gemver.sdfg',
    'gesummv.sdfg',
    'go_fast.sdfg',
    'gramschmidt.sdfg',
    'hdiff.sdfg',
    'heat_3d.sdfg', # Done
    'jacobi_1d.sdfg',
    'jacobi_2d.sdfg',
    'k2mm.sdfg',
    'k3mm.sdfg',
    'lenet.sdfg',
    'lu.sdfg',
    'ludcmp.sdfg',
    'mlp.sdfg',
    'mvt.sdfg',
    'nbody.sdfg',
    'nussinov.sdfg',
    'resnet.sdfg',
    'scattering_self_energies.sdfg',
    'seidel_2d.sdfg',
    'softmax.sdfg',
    'spmv.sdfg',
    'stockham_fft.sdfg',
    'symm.sdfg',
    #'syr2k.sdfg',
    'syrk.sdfg',
    'trisolv.sdfg',
    'trmm.sdfg',
    'vadv.sdfg',
]

prefix = 'run2'


def main():
    ignore_regex = re.compile(
        '^' + '$|^'.join(IGNORE_LIST).replace('*', '\w*') + '$'
    )

    for graph_name in os.listdir('tests/npbench_graphs'):
        if graph_name in GRAPH_IGNORE_LIST:
            continue
        print('Moving on to', graph_name)
        try:
            sdfg = SDFG.from_file(f'tests/npbench_graphs/{graph_name}')
            optimizer = SDFGOptimizer(sdfg)
            pattern_matches = optimizer.get_pattern_matches()
            i = 0
            for pattern in pattern_matches:
                if ignore_regex.search(pattern.__class__.__name__):
                    continue

                print(f'Verifying {pattern.__class__.__name__} on {graph_name}')
                out_dir = os.path.join(
                    '.testdata', prefix, 'npbench',
                    graph_name.split('.')[0], 'fails',
                    pattern.__class__.__name__ + '_' + str(i)
                )
                success_dir = os.path.join(
                    '.testdata', prefix, 'npbench',
                    graph_name.split('.')[0], 'successes',
                    pattern.__class__.__name__ + '_' + str(i)
                )
                verifier = ff.TransformationVerifier(
                    pattern, sdfg, output_dir=out_dir, success_dir=success_dir
                )
                valid = verifier.verify(
                    n_samples=100, status=ff.StatusLevel.BAR_ONLY
                )
                if not valid:
                    print('Instance invalid!')

                i += 1
        except Exception as e:
            print('Exception occurred!')


if __name__ == '__main__':
    main()
