# Copyright 2022 ETH Zurich and the FuzzyFlow authors. All rights reserved.
# This file is part of FuzzyFlow, which is released under the BSD 3-Clause
# License. For details, see the LICENSE file.

import argparse
import os

from dace.sdfg import SDFG

from fuzzyflow import cutout
from fuzzyflow.util import load_transformation_from_file
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
        '-c',
        '--cutout-strategy',
        type=cutout.CutoutStrategy,
        choices=list(cutout.CutoutStrategy),
        help='Strategy to use for selecting subgraph cutouts',
        default=cutout.CutoutStrategy.SIMPLE
    )
    parser.add_argument(
        '-s',
        '--sampling-strategy',
        type=SamplingStrategy,
        choices=list(SamplingStrategy),
        help='Strategy to use for sampling testing data',
        default=SamplingStrategy.SIMPLE_UNIFORM
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

    # Load and validate SDFG. Invalid SDFGs should fail this process.
    sdfg = SDFG.from_file(sdfg_path)
    sdfg.validate()

    xform, target_sdfg = load_transformation_from_file(xform_path, sdfg)
    if xform is None or target_sdfg is None:
        print('Failed to load transformation')
        exit(1)

    verifier = TransformationVerifier(
        xform, sdfg, args.cutout_strategy, args.sampling_strategy
    )

    valid = verifier.verify(args.runs, status=True, enforce_finiteness=True)

    print('Transformation is valid' if valid else 'INVALID Transformation!')


if __name__ == '__main__':
    main()
