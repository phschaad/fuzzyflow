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
    'ReductionNOperation',
    'OuterProductOperation',
    'CopyToDevice',
    'LoopDetection',
    'StreamingComposition',
    'GPUTransformSDFG',
    'NestSDFG'
]

prefix = 'run_1'


last_pattern = None
last_graph = None
exception_nr = 0

def main():
    ignore_regex = re.compile(
        '^' + '$|^'.join(IGNORE_LIST).replace('*', '\w*') + '$'
    )

    sdfg = SDFG.from_file(f'tests/case_studies/cloudsc/gpu_transform_sdfg_failure/cloudscexp2_parallel.sdfg')
    t_sdfg = sdfg.sdfg_list[1]
    optimizer = SDFGOptimizer(t_sdfg)
    pattern_matches = list(optimizer.get_pattern_matches())
    i = 0
    print(len(pattern_matches), 'pattern matches found.')
    for pattern in pattern_matches:
        if ignore_regex.search(pattern.__class__.__name__):
            continue

        print('Verifying', pattern.__class__.__name__)

        out_dir = os.path.join(
            '.testdata_case_studies', 'cloudsc_blanket_testing', prefix,
            'fails',
            pattern.__class__.__name__ + '_' + str(i)
        )
        success_dir = os.path.join(
            '.testdata_case_studies', 'cloudsc_blanket_testing', prefix,
            'successes',
            pattern.__class__.__name__ + '_' + str(i)
        )
        verifier = ff.TransformationVerifier(
            pattern, t_sdfg, output_dir=out_dir, success_dir=success_dir,
            build_dir_base_path='/mnt/ramdisk'
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

if __name__ == '__main__':
    main()
