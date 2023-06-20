# Copyright 2022 ETH Zurich and the FuzzyFlow authors. All rights reserved.
# This file is part of FuzzyFlow, which is released under the BSD 3-Clause
# License. For details, see the LICENSE file.

import argparse
import copy
import os
import sympy as sp
from typing import Tuple

from dace.sdfg import SDFG
from dace.sdfg.analysis.cutout import SDFGCutout
from fuzzyflow.util import (StatusLevel, apply_transformation,
                            load_transformation_from_file)
from fuzzyflow.verification import constraints
from fuzzyflow.verification.sampling import DataSampler
from fuzzyflow.verification.testcase_generator import TestCaseGenerator


class CutoutExtractor:


    sdfg_path: str = None
    xform_path: str = None
    output_dir: str = None
    reduce: bool = False
    reduce_diff: bool = False
    maxd: int = 128


    def __init__(
        self, sdfg_path, xform_path, output_dir, reduce=False,
        reduce_diff=False, maxd=128
    ):
        self.sdfg_path = sdfg_path
        self.xform_path = xform_path
        self.output_dir = output_dir
        self.reduce = reduce
        self.reduce_diff = reduce_diff
        self.maxd = maxd


    def extract(self) -> Tuple[sp.Expr, sp.Expr, sp.Expr]:
        if self.reduce_diff:
            # Load and validate SDFG. Invalid SDFGs should fail this process.
            sdfg_noreduce = SDFG.from_file(self.sdfg_path)
            sdfg_noreduce.validate()
            sdfg_noreduce.name = 'noreduce_' + sdfg_noreduce.name
            sdfg_reduce = SDFG.from_file(self.sdfg_path)
            sdfg_reduce.validate()
            sdfg_reduce.name = 'reduce_' + sdfg_reduce.name

            xform_noreduce, _ = load_transformation_from_file(
                self.xform_path, sdfg_noreduce
            )
            if xform_noreduce is None:
                print('Failed to load transformation')
                exit(1)
            xform_reduce, _ = load_transformation_from_file(
                self.xform_path, sdfg_reduce
            )
            if xform_reduce is None:
                print('Failed to load transformation')
                exit(1)

            cutout_noreduce = SDFGCutout.from_transformation(
                sdfg_noreduce, xform_noreduce, use_alibi_nodes=False,
                reduce_input_config=False
            )

            orig_cutout_noreduce = copy.deepcopy(cutout_noreduce)
            apply_transformation(cutout_noreduce, xform_noreduce)

            cutout_noreduce._name = cutout_noreduce.name + '_transformed'

            # Calculate the size of the input space.
            input_size_noreduce: sp.Expr = sp.sympify(0)
            for icont in orig_cutout_noreduce.input_config:
                arr = orig_cutout_noreduce.arrays[icont]
                input_size_noreduce += arr.total_size
            simplified_noreduce = input_size_noreduce.simplify().subs(
                orig_cutout_noreduce.constants
            )

            sym_constraints = constraints.constrain_symbols(
                orig_cutout_noreduce, sdfg_noreduce, max_dim=self.maxd
            )

            sampler = DataSampler(seed=12121)
            tc_generator_noreduce = TestCaseGenerator(
                os.path.join(self.output_dir, 'noreduce'), None,
                StatusLevel.DEBUG, sampler
            )

            tc_generator_noreduce.save_success_case(
                orig_cutout_noreduce, cutout_noreduce, xform_noreduce,
                cutout_symbol_constraints=sym_constraints,
                maximum_data_dim=self.maxd
            )

            cutout_reduce = SDFGCutout.from_transformation(
                sdfg_reduce, xform_reduce, use_alibi_nodes=False,
                reduce_input_config=True
            )

            orig_cutout_reduce = copy.deepcopy(cutout_reduce)
            apply_transformation(cutout_reduce, xform_reduce)

            cutout_reduce._name = cutout_reduce.name + '_transformed'

            # Calculate the size of the input space.
            input_size_reduce: sp.Expr = sp.sympify(0)
            for icont in orig_cutout_reduce.input_config:
                arr = orig_cutout_reduce.arrays[icont]
                input_size_reduce += arr.total_size
            simplified_reduce = input_size_reduce.simplify().subs(
                orig_cutout_reduce.constants
            )

            sym_constraints = constraints.constrain_symbols(
                orig_cutout_reduce, sdfg_reduce, max_dim=self.maxd
            )

            sampler = DataSampler(seed=12121)
            tc_generator_reduce = TestCaseGenerator(
                os.path.join(self.output_dir, 'reduce'), None,
                StatusLevel.DEBUG, sampler
            )

            tc_generator_reduce.save_success_case(
                orig_cutout_reduce, cutout_reduce, xform_reduce,
                cutout_symbol_constraints=sym_constraints,
                maximum_data_dim=self.maxd
            )

            reduction_factor = (
                simplified_noreduce - simplified_reduce
            ) / simplified_noreduce
            reduction_factor = sp.simplify(reduction_factor * 100).evalf(n=10)
            print('Input space reduced by:', reduction_factor, '%')
            print('Original, non-reduced size:', simplified_noreduce)
            print('Reduced size:', simplified_reduce)

            return simplified_noreduce, simplified_reduce, reduction_factor
        else:
            # Load and validate SDFG. Invalid SDFGs should fail this process.
            sdfg = SDFG.from_file(self.sdfg_path)
            sdfg.validate()

            xform, target_sdfg = load_transformation_from_file(
                self.xform_path, sdfg
            )
            if xform is None or target_sdfg is None:
                print('Failed to load transformation')
                exit(1)

            cutout = SDFGCutout.from_transformation(
                sdfg, xform, use_alibi_nodes=False,
                reduce_input_config=self.reduce
            )
            print('Cutout obtained')

            orig_cutout = copy.deepcopy(cutout)
            apply_transformation(cutout, xform)

            cutout._name = cutout.name + '_transformed'

            # Calculate the size of the input space.
            input_size: sp.Expr = sp.sympify(0)
            for icont in orig_cutout.input_config:
                arr = orig_cutout.arrays[icont]
                input_size += arr.total_size
            simplified = input_size.simplify().subs(orig_cutout.constants)
            print('Input space size:', input_size.simplify())
            print('Input space size numeric:', simplified)

            sym_constraints = constraints.constrain_symbols(
                orig_cutout, sdfg, max_dim=self.maxd
            )

            sampler = DataSampler(seed=12121)
            tc_generator = TestCaseGenerator(
                self.output_dir, None, StatusLevel.DEBUG, sampler
            )

            tc_generator.save_success_case(
                orig_cutout, cutout, xform,
                cutout_symbol_constraints=sym_constraints,
                maximum_data_dim=self.maxd
            )

            return simplified, 0, 0


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
        '--noreduce-reduce-diff',
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

    reduce = True if args.reduce else False
    reduce_diff = True if args.noreduce_reduce_diff else False

    extractor = CutoutExtractor(
        sdfg_path, xform_path, output_dir, reduce, reduce_diff, args.maxd
    )
    extractor.extract()


if __name__ == '__main__':
    main()
