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
    'floyd_warshall.sdfg',
    'heat_3d.sdfg',
    'correlation.sdfg',

    'scattering_self_energies.sdfg',
    'stockham_fft.sdfg',
    'syrk.sdfg',
    'syr2k.sdfg',
]


def main():
    ignore_regex = re.compile(
        '^' + '$|^'.join(IGNORE_LIST).replace('*', '\w*') + '$'
    )

    for graph_name in os.listdir('tests/npbench_graphs'):
        if graph_name in GRAPH_IGNORE_LIST:
            continue
        sdfg = SDFG.from_file(f'tests/npbench_graphs/{graph_name}')
        optimizer = SDFGOptimizer(sdfg)
        pattern_matches = optimizer.get_pattern_matches()
        i = 0
        for pattern in pattern_matches:
            if ignore_regex.search(pattern.__class__.__name__):
                continue

            print(f'Verifying {pattern.__class__.__name__} on {graph_name}')
            out_dir = os.path.join(
                '.testdata', 'npbench', graph_name.split('.')[0], 'fails',
                'pattern_' + str(i), pattern.__class__.__name__
            )
            success_dir = os.path.join(
                '.testdata', 'npbench', graph_name.split('.')[0], 'successes',
                'pattern_' + str(i), pattern.__class__.__name__
            )
            verifier = ff.TransformationVerifier(
                pattern, sdfg, output_dir=out_dir, success_dir=success_dir
            )
            valid = verifier.verify(n_samples=100, status=ff.StatusLevel.DEBUG)
            if not valid:
                print('Instance invalid!')

            i += 1


if __name__ == '__main__':
    main()
