import os
import re
from dace.sdfg import SDFG
from dace.transformation.optimizer import SDFGOptimizer

import fuzzyflow as ff


IGNORE_LIST = [
    'ElementWise*',
    'FPGA*',
    #'GPU*',
    'MPI*',
    'ReductionNOperation',
    'CopyToDevice',
    'LoopDetection',

    #'NestSDFG',
]


def main():
    ignore_regex = re.compile(
        '^' + '$|^'.join(IGNORE_LIST).replace('*', '\w*') + '$'
    )

    for graph_name in os.listdir('tests/npbench_graphs'):
        sdfg = SDFG.from_file(f'tests/npbench_graphs/{graph_name}')
        optimizer = SDFGOptimizer(sdfg)
        pattern_matches = optimizer.get_pattern_matches()
        for pattern in pattern_matches:
            if ignore_regex.search(pattern.__class__.__name__):
                continue

            print(f'Verifying {pattern.__class__.__name__} on {graph_name}')
            out_dir = os.path.join(
                '.testdata', 'npbench', graph_name, pattern.__class__.__name__,
                'failed'
            )
            success_dir = os.path.join(
                '.testdata', 'npbench', graph_name, pattern.__class__.__name__,
                'success'
            )
            verifier = ff.TransformationVerifier(
                pattern, sdfg, output_dir=out_dir, success_dir=success_dir
            )
            valid = verifier.verify(n_samples=100, status=ff.StatusLevel.DEBUG)
            if not valid:
                print('Instance invalid!')


if __name__ == '__main__':
    main()
