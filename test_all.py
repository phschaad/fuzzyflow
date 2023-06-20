import os
import re
from dace.sdfg import SDFG
from dace.transformation.optimizer import SDFGOptimizer
import traceback

import fuzzyflow as ff


IGNORE_LIST = [
    'ElementWise*',
    'FPGA*',
    'MPI*',
    'GPU*',
    'ReductionNOperation',
    'OuterProductOperation',
    'CopyToDevice',
    'LoopDetection',
    'StreamingComposition',
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
    'conv2d_bias.sdfg', # Causes exceptions in pattern matching
    'correlation.sdfg',
    'covariance.sdfg',
    'crc16.sdfg',
    'deriche.sdfg',
    'doitgen.sdfg',
    'durbin.sdfg',
    'fdtd_2d.sdfg',
    'floyd_warshall.sdfg',
    'gemm.sdfg',
    'gemver.sdfg',
    'gesummv.sdfg',
    'go_fast.sdfg',
    'gramschmidt.sdfg',
    'hdiff.sdfg',
    'heat_3d.sdfg',
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
    'scattering_self_energies.sdfg', # Causes out of memory.
    'seidel_2d.sdfg',
    'softmax.sdfg',
    'spmv.sdfg',
    'stockham_fft.sdfg',
    'symm.sdfg',
    'syr2k.sdfg',
    'syrk.sdfg',
    'trisolv.sdfg',
    'trmm.sdfg',
    'vadv.sdfg',
]

prefix = 'run_3'


last_pattern = None
last_graph = None
exception_nr = 0

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

                last_pattern = pattern
                last_graph = graph_name

                print(f'Verifying {pattern.__class__.__name__} on {graph_name}')
                out_dir = os.path.join(
                    '.testdata', 'npbench', prefix,
                    graph_name.split('.')[0], 'fails',
                    pattern.__class__.__name__ + '_' + str(i)
                )
                success_dir = os.path.join(
                    '.testdata', 'npbench', prefix,
                    graph_name.split('.')[0], 'successes',
                    pattern.__class__.__name__ + '_' + str(i)
                )
                verifier = ff.TransformationVerifier(
                    pattern, sdfg, output_dir=out_dir, success_dir=success_dir,
                    build_dir_base_path='/dev/shm/buildcache'
                )
                valid, dt = verifier.verify(
                    n_samples=100, status=ff.StatusLevel.BAR_ONLY,
                    minimize_input=False, use_alibi_nodes=False,
                    maximum_data_dim=64
                )
                print('Valid:', valid, 'Time:', dt)
                if not valid:
                    print('Instance invalid!')

                i += 1
        except Exception as e:
            print('-' * 80)
            print('Exception occurred!')
            print(last_pattern)
            print(last_graph)
            print('-' * 80)
            with open(
                'exception_' + str(graph_name) + '_' + '.txt', 'w'
            ) as f:
                f.writelines([
                    str(e),
                    str(last_pattern),
                    str(last_graph),
                    '',
                ])
                traceback.print_tb(e.__traceback__, file=f)


if __name__ == '__main__':
    main()
