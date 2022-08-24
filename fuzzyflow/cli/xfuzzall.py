# Copyright 2022 ETH Zurich and the FuzzyFlow authors. All rights reserved.
# This file is part of FuzzyFlow, which is released under the BSD 3-Clause
# License. For details, see the LICENSE file.

import argparse
import os
from typing import List
import warnings

from dace.sdfg import SDFG
from dace.transformation.passes.pattern_matching import match_patterns
import dace.transformation.dataflow as dxf
import dace.transformation.interstate as ixf
import dace.transformation.subgraph as sxf
import dace.transformation.passes as passes

from fuzzyflow import cutout
from fuzzyflow.verification.sampling import SamplingStrategy
from fuzzyflow.verification.verifier import StatusLevel, TransformationVerifier

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

    sdfg_path = args.programpath
    if not os.path.isfile(sdfg_path):
        print('SDFG file', sdfg_path, 'not found')
        exit(1)

    sdfg = SDFG.from_file(sdfg_path)
    sdfg.validate()

    matches: List[dxf.AugAssignToWCR] = list(
        match_patterns(sdfg, dxf.AugAssignToWCR)
    )
    n_matches = len(matches)
    print('Found', str(n_matches), 'matches')

    warnings.filterwarnings(
        'ignore', message='.*already loaded, renaming file.*'
    )

    i = 1
    invalid = set()
    for match in matches:
        print('Testing match', i, 'of', str(n_matches))
        verifier = TransformationVerifier(
            match, sdfg, args.cutout_strategy, args.sampling_strategy
        )
        valid = verifier.verify(
            args.runs, status=StatusLevel.DEBUG, enforce_finiteness=True
        )
        print('Transformation is valid' if valid else 'INVALID Transformation!')
        if not valid:
            invalid.add(i)
        i += 1

    if len(invalid) > 0:
        print('Invalid were the following', len(invalid), 'instances:')
        std_invalid = list(invalid).sort()
        for i in std_invalid:
            print(i)


if __name__ == '__main__':
    main()
