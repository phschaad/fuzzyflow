# Copyright 2022 ETH Zurich and the FuzzyFlow authors. All rights reserved.
# This file is part of FuzzyFlow, which is released under the BSD 3-Clause
# License. For details, see the LICENSE file.

import argparse
import json
import os
import warnings
import shutil
from typing import List

from dace import serialize
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
        '-t',
        '--transformation',
        type=str,
        help='<TRANSFORMATION TYPE>.<TRANSFORMATION>',
    )

    parser.add_argument(
        '--restore',
        action=argparse.BooleanOptionalAction,
        help='Restore from progress save file'
    )

    parser.add_argument(
        '--savepath',
        type=str,
        help='<Path to progress save file>',
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

    parser.add_argument(
        '--skip-n',
        type=int,
        help='Skip the first N instances of the transformation'
    )

    args = parser.parse_args()

    sdfg_path = args.programpath
    if not os.path.isfile(sdfg_path):
        print('SDFG file', sdfg_path, 'not found')
        exit(1)

    xfparam: str = args.transformation
    if ((xfparam is None or xfparam == '' or not '.' in xfparam) and
         not args.restore):
        print('No valid transformation parameter provided')
        parser.print_help()
        exit(1)

    progress_save_path = '.progressfile'
    if args.savepath is not None and args.savepath != '':
        progress_save_path = args.savepath

    warnings.filterwarnings(
        'ignore', message='.*already loaded, renaming file.*'
    )

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

    if args.output is not None:
        if not os.path.exists(args.output):
            os.makedirs(args.output, exist_ok=True)
    output_dir = args.output if (
        args.output is not None and os.path.exists(args.output)
    ) else None

    if args.success_dir is not None:
        if not os.path.exists(args.success_dir):
            os.makedirs(args.success_dir, exist_ok=True)
    success_dir = args.success_dir if (
        args.success_dir is not None and os.path.exists(args.success_dir)
    ) else None

    if args.restore and os.path.exists(progress_save_path):
        savefile = open(progress_save_path, 'r')
        if savefile is None:
            print(progress_save_path, 'is not a valid file')
            exit(1)
        file_contents = json.load(savefile)
        if savefile is None:
            print('Could not load progress')
            exit(1)
        savefile.close()

        i = file_contents['index']
        matches = [serialize.from_json(t) for t in file_contents['matches']]
        n_matches = len(matches)
        while (i - 1) < len(matches):
            print('Testing match', i, 'of', str(n_matches))
            match = matches[i - 1]
            match._sdfg = sdfg.sdfg_list[match.sdfg_id]
            instance_out_path = None
            if output_dir:
                instance_out_path = os.path.join(
                    output_dir, xf_name + '_' + str(i)
                )
            instance_success_path = None
            if success_dir:
                instance_success_path = os.path.join(
                    success_dir, xf_name + '_' + str(i)
                )
            verifier = TransformationVerifier(
                match, sdfg, args.sampling_strategy,
                instance_out_path, instance_success_path
            )
            valid = verifier.verify(
                args.runs, status=StatusLevel.DEBUG, enforce_finiteness=True,
                symbol_constraints=symbol_constraints,
                data_constraints=data_constraints
            )
            if not valid:
                print('INVALID Transformation!')
                file_contents['invalid_indices'].append(i)
            else:
                print('Transformation is valid')
            i += 1

            file_contents['index'] = i
            with open(progress_save_path, 'w') as savefile:
                json.dump(file_contents, savefile)

        if len(file_contents['invalid_indices']) > 0:
            print(
                'Invalid were the following',
                len(file_contents['invalid_indices']), 'instances:'
            )
            for i in file_contents['invalid_indices']:
                print(str(i))
    else:
        xf_split = xfparam.split('.')
        xf_type = xf_split[0]
        xf_name = xf_split[1]

        base_cls = None
        if xf_type == 'dataflow':
            base_cls = dxf
        elif xf_type == 'interstate':
            base_cls = ixf
        elif xf_type == 'subgraph':
            base_cls = sxf
        elif xf_type == 'passes':
            base_cls = passes
        else:
            print('Unknown transformation type', xf_type)
            print('Supported: dataflow | interstate | subgraph | passes')
            exit(1)

        if not hasattr(base_cls, xf_name):
            print('Transformation type', xf_type, 'has no member', xf_name)
            exit(1)

        matches: List = list(
            match_patterns(sdfg, getattr(base_cls, xf_name))
        )
        n_matches = len(matches)
        print('Found', str(n_matches), 'matches')

        file_contents = {
            'matches': [t.to_json() for t in matches],
            'index': 0,
            'invalid_indices': []
        }

        i = 1
        invalid = set()
        failed = set()
        for match in matches:
            print('Testing match', i, 'of', str(n_matches))
            if args.skip_n:
                if i <= args.skip_n:
                    print('Skipping')
                    i += 1
                    continue
            instance_out_path = None
            if output_dir:
                instance_out_path = os.path.join(
                    output_dir, xf_name + '_' + str(i)
                )
            instance_success_path = None
            if success_dir:
                instance_success_path = os.path.join(
                    success_dir, xf_name + '_' + str(i)
                )
            verifier = TransformationVerifier(
                match, sdfg, args.sampling_strategy, instance_out_path,
                instance_success_path
            )
            #try:
            valid = verifier.verify(
                args.runs, status=StatusLevel.DEBUG,
                enforce_finiteness=True,
                symbol_constraints=symbol_constraints,
                data_constraints=data_constraints
            )
            if not valid:
                print('INVALID Transformation!')
                invalid.add(i)
                file_contents['invalid_indices'].append(i)
            else:
                print('Transformation is valid')
            #except Exception as e:
            #    failed.add(i)
            #    print('Failed to validate with exception')
            #    print(e)
            for folder in os.listdir('.dacecache'):
                shutil.rmtree(os.path.join('.dacecache', folder))
            i += 1

            file_contents['index'] = i
            with open(progress_save_path, 'w') as savefile:
                json.dump(file_contents, savefile)

        if len(invalid) > 0:
            print('Invalid were the following', len(invalid), 'instances:')
            std_invalid = list(invalid)
            std_invalid.sort()
            for i in std_invalid:
                print(i)
        if len(failed) > 0:
            print('Failed in the following', len(failed), 'instances:')
            std_failed = list(failed)
            std_failed.sort()
            for i in std_failed:
                print(i)


if __name__ == '__main__':
    main()
