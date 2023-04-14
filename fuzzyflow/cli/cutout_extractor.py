# Copyright 2022 ETH Zurich and the FuzzyFlow authors. All rights reserved.
# This file is part of FuzzyFlow, which is released under the BSD 3-Clause
# License. For details, see the LICENSE file.

import argparse
import copy
import os

from dace.sdfg import SDFG
from dace.sdfg.analysis.cutout import SDFGCutout
from fuzzyflow.util import (StatusLevel, apply_transformation,
                            load_transformation_from_file)
from fuzzyflow.verification import constraints
from fuzzyflow.verification.sampling import DataSampler
from fuzzyflow.verification.testcase_generator import TestCaseGenerator


def main():
    parser = argparse.ArgumentParser(
        description='Extract Program Cutouts for Transformations'
    )

    parser.add_argument(
        '-p',
        '--programpath',
        type=str,
        help='<PATH TO SDFG FILE>',
        required=True
    )
    parser.add_argument(
        '-x',
        '--transformpath',
        type=str,
        help='<PATH TO SDFG TRANSFORMATION FILE>',
        required=True
    )
    parser.add_argument(
        '-o',
        '--output',
        type=str,
        help='<PATH TO OUTPUT FOLDER>',
    )
    parser.add_argument(
        '-r',
        '--reduce',
        action=argparse.BooleanOptionalAction,
        help='Reduce the input configuration',
    )
    parser.add_argument(
        '--maxd',
        type=int,
        help='<Maximum data dimension size>',
        default=128
    )

    args = parser.parse_args()

    # Check if both the SDFG file and transformation file exist.
    sdfg_path = args.programpath
    if not os.path.isfile(sdfg_path):
        print('SDFG file', sdfg_path, 'not found')
        exit(1)

    xform_path = args.transformpath
    if not os.path.isfile(xform_path):
        print('Transformation file', xform_path, 'not found')
        exit(1)

    if args.output is not None:
        if not os.path.exists(args.output):
            os.makedirs(args.output, exist_ok=True)
    output_dir = args.output if os.path.exists(args.output) else None

    # Load and validate SDFG. Invalid SDFGs should fail this process.
    sdfg = SDFG.from_file(sdfg_path)
    sdfg.validate()

    xform, target_sdfg = load_transformation_from_file(xform_path, sdfg)
    if xform is None or target_sdfg is None:
        print('Failed to load transformation')
        exit(1)

    reduce = True if args.reduce else False

    cutout = SDFGCutout.from_transformation(
        sdfg, xform, use_alibi_nodes=False, reduce_input_config=reduce
    )
    print('Cutout obtained')

    orig_cutout = copy.deepcopy(cutout)
    apply_transformation(cutout, xform)

    cutout._name = cutout.name + '_transformed'

    sym_constraints = constraints.constrain_symbols(
        orig_cutout, sdfg, max_dim=args.maxd
    )

    sampler = DataSampler(seed=12121)
    tc_generator = TestCaseGenerator(
        output_dir, None, StatusLevel.DEBUG, sampler
    )

    tc_generator.save_success_case(
        orig_cutout, cutout, xform, cutout_symbol_constraints=sym_constraints,
        maximum_data_dim=args.maxd
    )


if __name__ == '__main__':
    main()
