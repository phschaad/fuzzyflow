# Copyright 2022 ETH Zurich and the FuzzyFlow authors. All rights reserved.
# This file is part of FuzzyFlow, which is released under the BSD 3-Clause
# License. For details, see the LICENSE file.

import argparse
from dace.sdfg import SDFG
import os

from fuzzyflow import cutout
from fuzzyflow.util import apply_transformation, load_transformation_from_file
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
    if xform.state_id < 0:
        raise NotImplementedError(
            'Processing multi-state transformations is currently not possible'
        )

    verifier = TransformationVerifier(xform, sdfg, cutout.CutoutStrategy.SIMPLE)

    n_verification_samples = 10
    valid = verifier.verify(n_verification_samples)

    print('Transformation is valid' if valid else 'INVALID Transformation!')


if __name__ == '__main__':
    main()
