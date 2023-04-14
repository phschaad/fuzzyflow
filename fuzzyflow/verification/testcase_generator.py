# Copyright 2022 ETH Zurich and the FuzzyFlow authors. All rights reserved.
# This file is part of FuzzyFlow, which is released under the BSD 3-Clause
# License. For details, see the LICENSE file.

import json
import os
import pickle
import traceback
from typing import Optional, Set, Tuple, Union

import dace
from dace import symbolic
from dace.sdfg import SDFG
from dace.sdfg.analysis.cutout import SDFGCutout
from dace.transformation.transformation import (PatternTransformation,
                                                SubgraphTransformation)
from fuzzyflow.harness_generator import sdfg2cpp
from fuzzyflow.util import FailureReason, StatusLevel
from fuzzyflow.verification.sampling import DataSampler


class TestCaseGenerator:


    _status: StatusLevel
    _sampler: DataSampler
    _success_dir: str
    _failure_dir: str


    def __init__(
        self, success_dir: str, failure_dir: str, status: StatusLevel,
        sampler: Optional[DataSampler] = None
    ) -> None:
        self._success_dir = success_dir
        self._failure_dir = failure_dir
        self._status = status
        if sampler:
            self._sampler = sampler
        else:
            self._sampler = DataSampler(seed = 12121)


    def _save_harness(
        self, dir: str, sym_constraints: dict, pre: SDFG, post: SDFG,
        inputs: dict, free_symbols_map: dict
    ) -> None:
        if sym_constraints is None:
            return
        if inputs is None:
            return
        if free_symbols_map is None:
            return

        if self._status >= StatusLevel.VERBOSE:
            print('Generateing harness')

        init_args = {}
        for name in inputs.keys():
            init_args[name] = 'rand'

        try:
            sdfg2cpp.dump_args(
                'c++', os.path.join(dir, 'harness'), init_args,
                sym_constraints, pre, post, **inputs, **free_symbols_map
            )
        except Exception:
            with open(os.path.join(dir, 'EXCEPT'), 'w') as f:
                f.write('Failed to generate harness')
                traceback.print_exc(file=f)


    def _save_transformation(
        self, dir: str,
        xform: Union[SubgraphTransformation, PatternTransformation]
    ) -> None:
        if self._status >= StatusLevel.VERBOSE:
            print('Saving transformation')
        with open(os.path.join(dir, 'xform.json'), 'w') as f:
            json.dump(xform.to_json(), f, indent=4)


    def _save_inputs_dbg(
        self, dir: str, inputs: Optional[dict] = None,
        cutout_symbol_constraints: Optional[dict] = None,
        free_symbols_map: Optional[dict] = None
    ) -> None:
        if self._status >= StatusLevel.VERBOSE:
            print('Saving inputs for debugging purposes')
        if inputs is not None:
            with open(os.path.join(dir, 'inputs'), 'wb') as f:
                pickle.dump(inputs, f, protocol=pickle.HIGHEST_PROTOCOL)
        if cutout_symbol_constraints is not None:
            with open(os.path.join(dir, 'constraints'), 'wb') as f:
                pickle.dump(
                    cutout_symbol_constraints, f,
                    protocol=pickle.HIGHEST_PROTOCOL
                )
        if free_symbols_map is not None:
            with open(os.path.join(dir, 'symbols'), 'wb') as f:
                pickle.dump(
                    free_symbols_map, f, protocol=pickle.HIGHEST_PROTOCOL
                )


    def _sample_valid_inputs(
        self, pre: SDFG, post: SDFG, cutout_symbol_constraints: dict,
        maximum_data_dim: int, data_constraints: Optional[dict] = None,
        strict_config: bool = False
    ) -> Tuple[dict, dict]:
        if self._status >= StatusLevel.VERBOSE:
            print(
                'Sampling a new input set to save for test cases.'
            )
        symbols_map, free_symbols_map = self._sampler.sample_symbols_map_for(
            pre, post, constraints_map=cutout_symbol_constraints,
            maxval=maximum_data_dim
        )

        constraints_map = None
        if data_constraints is not None:
            constraints_map = {
                k: (symbolic.pystr_to_symbolic(lval),
                    symbolic.pystr_to_symbolic(hval)) for k, (lval, hval) in
                    data_constraints.items()
            }
        orig_in_config: Set[str] = set()
        new_in_config: Set[str] = set()
        output_config: Set[str] = set()
        if strict_config and isinstance(pre, SDFGCutout):
            orig_in_config = pre.input_config
            output_config = pre.output_config
        else:
            for name, desc in pre.arrays.items():
                if not desc.transient:
                    orig_in_config.add(name)
                    output_config.add(name)
        if isinstance(post, SDFGCutout):
            new_in_config = post.input_config
        else:
            for name, desc in post.arrays.items():
                if not desc.transient:
                    new_in_config.add(name)

        inputs = self._sampler.sample_inputs(
            pre, orig_in_config, symbols_map, constraints_map=constraints_map
        )

        return inputs, free_symbols_map


    def _save_cutouts(self, dir: str, pre: SDFG, post: SDFG) -> None:
        if self._status >= StatusLevel.VERBOSE:
            print('Saving cutouts')

        for sd in pre.all_sdfgs_recursive():
            for state in sd.states():
                for dn in state.data_nodes():
                    dn.instrument = \
                        dace.DataInstrumentationType.No_Instrumentation
        for sd in post.all_sdfgs_recursive():
            for state in sd.states():
                for dn in state.data_nodes():
                    dn.instrument = \
                        dace.DataInstrumentationType.No_Instrumentation

        post.save(os.path.join(dir, 'post.sdfg'))
        pre.save(os.path.join(dir, 'pre.sdfg'))


    def _save_original_sdfg(self, dir: str, sdfg: SDFG) -> None:
        if self._status >= StatusLevel.VERBOSE:
            print('Saving original SDFG')

        for sd in sdfg.all_sdfgs_recursive():
            for state in sd.states():
                for dn in state.data_nodes():
                    dn.instrument = \
                        dace.DataInstrumentationType.No_Instrumentation

        sdfg.save(os.path.join(dir, 'program.sdfg'))


    def save_failure_case(
        self, reason: FailureReason, details: str, pre: SDFG, post: SDFG,
        original_sdfg: SDFG,
        xform: Union[SubgraphTransformation, PatternTransformation],
        iteration: Optional[int] = None,
        exception: Optional[Exception] = None,
        inputs: Optional[dict] = None,
        symbol_constraints: Optional[dict] = None,
        symbols: Optional[dict] = None
    ) -> None:
        os.makedirs(self._failure_dir, exist_ok=True)

        # Save additional information about the failure.
        with open(os.path.join(self._failure_dir, reason.value), 'w') as f:
            if details:
                f.writelines([
                    'Reason: ' + reason.value + '\n', 'Details: \n',
                    details, '\n'
                ])
            else:
                f.writelines([
                    'Reason:' + reason.value + '\n', 'Details: \n-\n'
                ])

            if iteration is not None:
                f.writelines(['Iteration: ' + str(iteration) + '\n'])

            if exception is not None:
                traceback.print_tb(exception.__traceback__, file=f)

        if reason == FailureReason.EXCEPTION:
            self._save_original_sdfg(self._failure_dir, original_sdfg)
            self._save_transformation(self._failure_dir, xform)
        else:
            self._save_cutouts(self._failure_dir, pre, post)
            self._save_inputs_dbg(
                self._failure_dir, inputs, symbol_constraints,
                free_symbols_map=symbols
            )
            self._save_transformation(self._failure_dir, xform)
            self._save_harness(
                self._failure_dir, symbol_constraints, pre, post, inputs,
                symbols
            )


    def save_success_case(
        self, pre: SDFG, post: SDFG,
        xform: Union[PatternTransformation, SubgraphTransformation],
        strict_config: bool = False,
        cutout_symbol_constraints: Optional[dict] = None,
        data_constraints: Optional[dict] = None, maximum_data_dim: int = 128
    ) -> None:
        os.makedirs(self._success_dir, exist_ok=True)
        inputs, free_symbols_map = self._sample_valid_inputs(
            pre, post, cutout_symbol_constraints,
            maximum_data_dim=maximum_data_dim,
            data_constraints=data_constraints, strict_config=strict_config
        )
        self._save_cutouts(self._success_dir, pre, post)
        self._save_inputs_dbg(
            self._success_dir, inputs, cutout_symbol_constraints,
            free_symbols_map
        )
        self._save_transformation(self._success_dir, xform)
        self._save_harness(
            self._success_dir, cutout_symbol_constraints, pre, post, inputs,
            free_symbols_map
        )
