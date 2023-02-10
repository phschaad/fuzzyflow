# Copyright 2022 ETH Zurich and the FuzzyFlow authors. All rights reserved.
# This file is part of FuzzyFlow, which is released under the BSD 3-Clause
# License. For details, see the LICENSE file.

from copy import deepcopy
from collections import deque
from typing import Dict, List, Union, Optional, Set
import os
import numpy as np
from tqdm import tqdm
from enum import Enum
import json

import dace
from dace import config, dtypes
from dace.codegen.compiled_sdfg import CompiledSDFG
from dace.data import Scalar
from dace.sdfg import SDFG, SDFGState
from dace.sdfg import nodes as nd
from dace.transformation.transformation import (PatternTransformation,
                                                SubgraphTransformation)
from dace.symbolic import pystr_to_symbolic
from dace.sdfg.validation import InvalidSDFGError
from dace.transformation.passes.analysis import StateReachability
from dace.codegen.instrumentation.data.data_report import InstrumentedDataReport

from fuzzyflow.cutout import (CutoutStrategy,
                              TranslationDict,
                              cutout_determine_input_config,
                              cutout_determine_system_state,
                              find_cutout_for_transformation,
                              determine_cutout_reachability)
from fuzzyflow.runner import run_subprocess_precompiled
from fuzzyflow.util import StatusLevel, apply_transformation, cutout_determine_symbol_constraints
from fuzzyflow.verification.sampling import DataSampler, SamplingStrategy


class FailureReason(Enum):
    FAILED_VALIDATE = 'FAILED_VALIDATE'
    COMPILATION_FAILURE = 'COMPILATION_FAILURE'
    EXIT_CODE_MISMATCH = 'EXIT_CODE_MISMATCH'
    SYSTEM_STATE_MISMATCH = 'SYSTEM_STATE_MISMATCH'


class TransformationVerifier:

    xform: Union[SubgraphTransformation, PatternTransformation] = None
    sdfg: SDFG = None
    cutout_strategy: CutoutStrategy = CutoutStrategy.SIMPLE
    sampling_strategy: SamplingStrategy = SamplingStrategy.SIMPLE_UNIFORM
    output_dir: Optional[str] = None

    _cutout: SDFG = None
    _original_cutout: SDFG = None
    _translation_dict: TranslationDict = None
    _inverse_translation_dict: TranslationDict = None
    _states_reached_by_cutout: Set[SDFGState] = set()
    _states_reaching_cutout: Set[SDFGState] = set()
    _state_reachability_dict: Dict[int, Dict[SDFGState, Set[SDFGState]]] = None

    def __init__(
        self,
        xform: Union[SubgraphTransformation, PatternTransformation],
        sdfg: SDFG,
        cutout_strategy: CutoutStrategy = CutoutStrategy.SIMPLE,
        sampling_strategy: SamplingStrategy = SamplingStrategy.SIMPLE_UNIFORM,
        output_dir: Optional[str] = None,
    ):
        self.xform = xform
        self.sdfg = sdfg
        self.cutout_strategy = cutout_strategy
        self.sampling_strategy = sampling_strategy
        self.output_dir = output_dir


    def cutout(
        self, strategy: CutoutStrategy = None,
        status: StatusLevel = StatusLevel.OFF
    ) -> SDFG:
        recut = False
        if strategy is not None and strategy != self.cutout_strategy:
            self.cutout_strategy = strategy
            recut = True

        if self._cutout is None or recut:
            if status >= StatusLevel.DEBUG:
                print('Finding ideal cutout')
            (
                self._cutout, self._translation_dict
            ) = find_cutout_for_transformation(
                self.sdfg, self.xform, self.cutout_strategy
            )

            self._inverse_translation_dict = dict()
            for k, v in self._translation_dict.items():
                self._inverse_translation_dict[v] = k

            original_sdfg_id = self._inverse_translation_dict[self.sdfg.sdfg_id]
            self._state_reachability_dict = StateReachability().apply_pass(
                self.sdfg.sdfg_list[original_sdfg_id], None
            )
            state_reach = self._state_reachability_dict[original_sdfg_id]
            (
                self._states_reaching_cutout, self._states_reached_by_cutout
            ) = determine_cutout_reachability(
                self._cutout, self.sdfg, self._translation_dict,
                self._inverse_translation_dict, state_reach
            )

            in_nodes: List[nd.AccessNode] = self._cutout.input_arrays()
            for in_node in in_nodes:
                self._cutout.arrays[in_node.data].transient = False
            out_nodes: List[nd.AccessNode] = self._cutout.output_arrays()
            for out_node in out_nodes:
                self._cutout.arrays[out_node.data].transient = False
            if status >= StatusLevel.DEBUG:
                print('Cutout obtained')

        return self._cutout


    def _data_report_get_latest_version(
        self, report: InstrumentedDataReport, item: str
    ) -> dace.data.ArrayLike:
        filenames = report.files[item]
        desc = report.sdfg.arrays[item]
        dtype: dtypes.typeclass = desc.dtype
        npdtype = dtype.as_numpy_dtype()

        file = deque(iter(filenames), maxlen=1).pop()
        nparr, view = report._read_file(file, npdtype)
        report.loaded_arrays[item, -1] = nparr
        return view


    def _catch_failure(
        self, reason: FailureReason, details: Optional[str]
    ) -> None:
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)

            # Save SDFGs for cutout both before and after transforming.
            self._original_cutout.save(
                os.path.join(self.output_dir, 'pre.sdfg')
            )
            self._cutout.save(
                os.path.join(self.output_dir, 'post.sdfg')
            )
            # Save the transformation.
            with open(os.path.join(self.output_dir, 'xform.json'), 'w') as f:
                json.dump(self.xform.to_json(), f, indent=4)

            # Save additional information about the failure.
            with open(os.path.join(self.output_dir, reason.value), 'w') as f:
                if details:
                    f.writelines([
                        'Reason: ' + reason.value + '\n', 'Details: ', details
                    ])
                else:
                    f.writelines([
                        'Reason:' + reason.value + '\n', 'Details: -'
                    ])


    def _do_verify(
        self, n_samples: int = 1, status: StatusLevel = StatusLevel.OFF,
        debug_save_path: str = None, enforce_finiteness: bool = False,
        symbol_constraints: Dict = None, data_constraints: Dict = None
    ) -> bool:
        cutout = self.cutout(status=status)

        system_state = cutout_determine_system_state(
            cutout, self._states_reached_by_cutout, self._translation_dict
        )
        original_input_configuration = cutout_determine_input_config(
            cutout, self._states_reaching_cutout, self._translation_dict
        )

        self._original_cutout = deepcopy(cutout)
        if status >= StatusLevel.DEBUG:
            print('Applying transformation')
        try:
            apply_transformation(cutout, self.xform)
        except InvalidSDFGError as e:
            self._catch_failure(FailureReason.FAILED_VALIDATE, str(e))
            return False

        for dat in system_state:
            if dat not in cutout.arrays.keys():
                print(
                    'Warning: Transformation removed something from system ' +
                    'state!'
                )
                orig_array = self._original_cutout.arrays[dat]
                cutout.add_datadesc(dat, orig_array)

        xformed_input_configuration = cutout_determine_input_config(
            cutout, self._states_reaching_cutout, self._translation_dict,
            system_state
        )

        # Instrumentation
        for s in cutout.states():
            for dn in s.data_nodes():
                if dn.data in system_state:
                    dn.instrument = dtypes.DataInstrumentationType.Save
        for s in self._original_cutout.states():
            for dn in s.data_nodes():
                if dn.data in system_state:
                    dn.instrument = dtypes.DataInstrumentationType.Save

        if debug_save_path is not None:
            self._original_cutout.save(debug_save_path + '_orig.sdfg')
            cutout.save(debug_save_path + '_xformed.sdfg')

        ##if not xformed_input_configuration.issubset(
        ##    original_input_configuration
        ##):
        ##    print(
        ##        'Failed due to invalid input configuration after transforming!'
        ##    )
        ##    return False

        if status >= StatusLevel.DEBUG:
            print('Compiling pre-transformation cutout')
        prog_orig: CompiledSDFG = self._original_cutout.compile()
        if status >= StatusLevel.DEBUG:
            print('Compiling post-transformation cutout')
        prog_xformed: CompiledSDFG = None
        try:
            prog_xformed = cutout.compile(validate=True)
        except InvalidSDFGError as e:
            self._catch_failure(FailureReason.FAILED_VALIDATE, str(e))
            return False
        except Exception as e:
            self._catch_failure(FailureReason.COMPILATION_FAILURE, str(e))
            return False
        if status >= StatusLevel.DEBUG:
            print(
                'Verifying transformation over', n_samples,
                'sampling run' + ('s' if n_samples > 1 else '')
            )

        seed = 12121
        sampler = DataSampler(self.sampling_strategy, seed)

        general_constraints = None
        if symbol_constraints is not None:
            general_constraints = {
                k: (
                    pystr_to_symbolic(lval),
                    pystr_to_symbolic(hval),
                    pystr_to_symbolic(sval)
                ) for k, (lval, hval, sval) in symbol_constraints.items()
            }

        cutout_symbol_constraints = cutout_determine_symbol_constraints(
            cutout, self.sdfg, pre_constraints=general_constraints
        )

        with tqdm(total=n_samples, disable=(status == StatusLevel.OFF)) as bar:
            i = 0
            resample_attempt = 0
            decay_by = 0
            decays = []
            n_crashes = 0
            full_resampling_failures = 0
            while i < n_samples:
                if status >= StatusLevel.VERBOSE:
                    bar.write('Sampling symbols')
                symbols_map, free_symbols_map = sampler.sample_symbols_map_for(
                    self._original_cutout,
                    constraints_map=cutout_symbol_constraints, maxval=256
                )

                constraints_map = None
                if data_constraints is not None:
                    constraints_map = {
                        k: (pystr_to_symbolic(lval), pystr_to_symbolic(hval))
                        for k, (lval, hval) in data_constraints.items()
                    }

                if status >= StatusLevel.VERBOSE:
                    bar.write('Sampling inputs')
                inputs = sampler.sample_inputs(
                    self._original_cutout, original_input_configuration,
                    symbols_map, decay_by, constraints_map=constraints_map
                )

                if status >= StatusLevel.VERBOSE:
                    bar.write(
                        'Duplicating inputs for post-transformation cutout'
                    )
                inputs_xformed = dict()
                for k, v in inputs.items():
                    if k in xformed_input_configuration:
                        inputs_xformed[k] = deepcopy(v)

                if status >= StatusLevel.VERBOSE:
                    bar.write(
                        'Generating outputs for pre-transformation cutout'
                    )
                out_orig = sampler.generate_output_containers(
                    self._original_cutout, system_state,
                    original_input_configuration, symbols_map
                )
                if status >= StatusLevel.VERBOSE:
                    bar.write(
                        'Generating outputs for post-transformation cutout'
                    )
                out_xformed = sampler.generate_output_containers(
                    cutout, system_state, xformed_input_configuration,
                    symbols_map
                )

                orig_containers = dict()
                for k in inputs.keys():
                    if not k in orig_containers:
                        orig_containers[k] = inputs[k]
                for k in out_orig.keys():
                    if not k in orig_containers:
                        orig_containers[k] = out_orig[k]
                if status >= StatusLevel.VERBOSE:
                    bar.write(
                        'Running pre-transformation cutout in a subprocess'
                    )
                ret_orig = run_subprocess_precompiled(
                    prog_orig, orig_containers, free_symbols_map
                )
                if status >= StatusLevel.VERBOSE:
                    bar.write(
                        'Collecting pre-transformation cutout data reports'
                    )
                orig_drep: InstrumentedDataReport = None
                orig_drep = self._original_cutout.get_instrumented_data()

                xformed_containers = dict()
                for k in inputs.keys():
                    if k in cutout.arrays:
                        if not k in xformed_containers:
                            xformed_containers[k] = inputs_xformed[k]
                for k in out_xformed.keys():
                    if k in cutout.arrays:
                        if not k in xformed_containers:
                            xformed_containers[k] = out_xformed[k]
                if status >= StatusLevel.VERBOSE:
                    bar.write(
                        'Running post-transformation cutout in a subprocess'
                    )
                ret_xformed = run_subprocess_precompiled(
                    prog_xformed, xformed_containers, free_symbols_map
                )
                if status >= StatusLevel.VERBOSE:
                    bar.write(
                        'Collecting post-transformation cutout data reports'
                    )
                xformed_drep: InstrumentedDataReport = None
                xformed_drep = cutout.get_instrumented_data()

                if status >= StatusLevel.VERBOSE:
                    bar.write('Comparing results')
                if ((ret_orig != 0 and ret_xformed == 0) or
                    (ret_orig == 0 and ret_xformed != 0)):
                    if status >= StatusLevel.VERBOSE:
                        bar.write('One cutout failed to run, the other ran!')
                    self._catch_failure(
                        FailureReason.EXIT_CODE_MISMATCH,
                        f'Exit code (${ret_xformed}) does not match oringinal' +
                        f' exit code (${ret_orig})'
                    )
                    return False
                elif ret_orig == 0 and ret_xformed == 0:
                    resample = False
                    for dat in system_state:

                        try:
                            oval = self._data_report_get_latest_version(
                                orig_drep, dat
                            )

                            if dat in xformed_drep.files:
                                nval = self._data_report_get_latest_version(
                                    xformed_drep, dat
                                )
                            else:
                                if dat not in inputs:
                                    if status >= StatusLevel.VERBOSE:
                                        bar.write('System state mismatch!')
                                    self._catch_failure(
                                        FailureReason.SYSTEM_STATE_MISMATCH,
                                        f'Missing input ${dat}'
                                    )
                                    return False
                                if isinstance(cutout.arrays[dat], Scalar):
                                    nval = [inputs[dat]]
                                else:
                                    nval = inputs[dat]

                            if (enforce_finiteness and
                                not np.isfinite(oval).all()):
                                if status >= StatusLevel.VERBOSE:
                                    bar.write('Non-finite results, resampling')
                                resample = True

                            if isinstance(oval, np.ndarray):
                                if not np.allclose(oval, nval, equal_nan=True):
                                    if status >= StatusLevel.VERBOSE:
                                        bar.write('Result mismatch!')
                                    self._catch_failure(
                                        FailureReason.SYSTEM_STATE_MISMATCH,
                                        f'Mismatching results for ${dat}'
                                    )
                                    return False
                            else:
                                if not np.allclose(
                                    [oval], [nval], equal_nan=True
                                ):
                                    if status >= StatusLevel.VERBOSE:
                                        bar.write('Result mismatch!')
                                    self._catch_failure(
                                        FailureReason.SYSTEM_STATE_MISMATCH,
                                        f'Mismatching results for ${dat}'
                                    )
                                    return False
                        except:
                            print(
                                'WARNING: Missing instrumentation on system ' +
                                'state for container',
                                dat
                            )

                    if not resample or resample_attempt > 11:
                        if resample_attempt > 11:
                            full_resampling_failures += 1
                        resample_attempt = 0
                        if enforce_finiteness and decay_by > 0:
                            decays.append(decay_by)
                        decay_by = 0
                        i += 1
                        bar.update(1)
                    else:
                        if resample_attempt >= 2:
                            if decay_by == 0:
                                decay_by = 1
                            else:
                                decay_by *= 2
                        resample_attempt += 1
                else:
                    if status >= StatusLevel.VERBOSE:
                        bar.write('Both cutouts crashed')
                    n_crashes += 1
                    if resample_attempt > 11:
                        full_resampling_failures += 1
                    resample_attempt = 0
                    decay_by = 0
                    i += 1
                    bar.update(1)
            n_decayed = len(decays)
            if enforce_finiteness and n_decayed > 0:
                print(
                    'Decayed on', str(n_decayed), 'out of', str(n_samples),
                    'samples with a median decay factor of',
                    str((2 ** -np.median(decays)))
                )
                if full_resampling_failures:
                    print(
                        'Failed to decay even with a factor of',
                        str((2 ** -512)), str(full_resampling_failures),
                        'times'
                    )
            if n_crashes > 0:
                print(
                    str(n_crashes), 'trials out of', str(n_samples),
                    'caused crashes'
                )

        return True


    def verify(
        self, n_samples: int = 1, status: StatusLevel = StatusLevel.OFF,
        debug_save_path: str = None, enforce_finiteness: bool = False,
        symbol_constraints: Dict = None, data_constraints: Dict = None
    ) -> bool:
        with config.temporary_config():
            config.Config.set(
                'compiler',
                'cpu',
                'args',
                value='-std=c++14 -fPIC -Wall -Wextra -O2 ' +
                    '-Wno-unused-parameter -Wno-unused-label'
            )
            config.Config.set('compiler', 'allow_view_arguments', value=True)
            config.Config.set('profiling', value=False)
            config.Config.set('debugprint', value=False)
            config.Config.set('cache', value='hash')
            return self._do_verify(
                n_samples, status, debug_save_path, enforce_finiteness,
                symbol_constraints, data_constraints
            )
