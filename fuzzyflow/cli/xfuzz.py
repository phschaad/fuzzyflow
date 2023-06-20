# Copyright 2022 ETH Zurich and the FuzzyFlow authors. All rights reserved.
# This file is part of FuzzyFlow, which is released under the BSD 3-Clause
# License. For details, see the LICENSE file.

import argparse
import json
import os
import warnings

from dace.sdfg import SDFG

from fuzzyflow.util import StatusLevel, load_transformation_from_file
from fuzzyflow.verification.sampling import SamplingStrategy
from fuzzyflow.verification.verifier import TransformationVerifier


def main():
    parser = argparse.ArgumentParser(
        description='Fuzzing-Based Program Transformation Verifier'
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
        '-r',
        '--runs',
        type=int,
        help='<number of validation runs to perform>',
        default=200
    )

    parser.add_argument(
        '--maxd',
        type=int,
        help='<Maximum data dimension size>',
        default=128
    )

    parser.add_argument(
        '-o',
        '--output',
        type=str,
        help='<PATH TO OUTPUT FOLDER>',
    )

    parser.add_argument(
        '--success-dir',
        type=str,
        help='<PATH TO SUCCESS CASE FOLDER>',
    )

    parser.add_argument(
        '--reduce',
        action=argparse.BooleanOptionalAction,
        help='Reduce the input configuration',
    )

    parser.add_argument(
        '-s',
        '--sampling-strategy',
        type=SamplingStrategy,
        choices=list(SamplingStrategy),
        help='Strategy to use for sampling testing data',
        default=SamplingStrategy.SIMPLE_UNIFORM
    )

    parser.add_argument(
        '--data-constraints-file',
        type=str,
        help='<Path to constraints file for data containers>'
    )
    parser.add_argument(
        '--symbol-constraints-file',
        type=str,
        help='<Path to constraints file for symbols>'
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

    if args.success_dir is not None:
        if not os.path.exists(args.success_dir):
            os.makedirs(args.success_dir, exist_ok=True)
    success_dir = args.success_dir if os.path.exists(args.success_dir) else None

    # Load and validate SDFG. Invalid SDFGs should fail this process.
    sdfg = SDFG.from_file(sdfg_path)
    sdfg.validate()

    symbol_constraints = None
    data_constraints = None
    sc_file_path = args.symbol_constraints_file
    if sc_file_path is not None and os.path.exists(sc_file_path):
        with open(sc_file_path, 'r') as sc_file:
            symbol_constraints = json.load(sc_file)
    dc_file_path = args.data_constraints_file
    if dc_file_path is not None and os.path.exists(dc_file_path):
        with open(dc_file_path, 'r') as dc_file:
            data_constraints = json.load(dc_file)

    xform, target_sdfg = load_transformation_from_file(xform_path, sdfg)
    if xform is None or target_sdfg is None:
        print('Failed to load transformation')
        exit(1)

    warnings.filterwarnings(
        'ignore', message='.*already loaded, renaming file.*'
    )

    reduce = True if args.reduce else False
    verifier = TransformationVerifier(
        xform, sdfg, args.sampling_strategy, output_dir=output_dir,
        success_dir=success_dir, status=StatusLevel.BAR_ONLY,
    )

    valid, dt = verifier.verify(
        args.runs, enforce_finiteness=False,
        symbol_constraints=symbol_constraints,
        data_constraints=data_constraints, minimize_input=reduce,
        maximum_data_dim=args.maxd
    )

    print('Transformation is valid' if valid else 'INVALID Transformation!')
    print('Time taken (s):', dt / 1e9)


if __name__ == '__main__':
    main()
