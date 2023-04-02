# Copyright 2022 ETH Zurich and the FuzzyFlow authors. All rights reserved.
# This file is part of FuzzyFlow, which is released under the BSD 3-Clause
# License. For details, see the LICENSE file.

from copy import deepcopy
from typing import Dict, Union, Optional, Set, Any, Tuple, List
import os
import numpy as np
from tqdm import tqdm
from enum import Enum
import json
from struct import error as StructError
import pickle
import traceback
import tempfile
import warnings
import sympy as sp
import time

import dace
from dace import config
from dace.codegen.compiled_sdfg import CompiledSDFG
from dace.sdfg import SDFG
from dace.sdfg.analysis.cutout import SDFGCutout
from dace.transformation.transformation import (PatternTransformation,
                                                SubgraphTransformation)
from dace.symbolic import pystr_to_symbolic
from dace.sdfg.validation import InvalidSDFGError

from fuzzyflow.runner import run_subprocess_precompiled
from fuzzyflow.util import (StatusLevel,
                            data_report_get_latest_version,
                            apply_transformation,
                            cutout_determine_symbol_constraints)
from fuzzyflow.verification.sampling import DataSampler, SamplingStrategy
from fuzzyflow.harness_generator import sdfg2cpp


class FailureReason(Enum):
    EXCEPTION = 'EXCEPTION'
    FAILED_VALIDATE = 'FAILED_VALIDATE'
    COMPILATION_FAILURE = 'COMPILATION_FAILURE'
    EXIT_CODE_MISMATCH = 'EXIT_CODE_MISMATCH'
    SYSTEM_STATE_MISMATCH = 'SYSTEM_STATE_MISMATCH'
    FAILED_TO_APPLY = 'FAILED_TO_APPLY'


class TransformationVerifier:

    _orig_xform: Union[SubgraphTransformation, PatternTransformation] = None
    xform: Union[SubgraphTransformation, PatternTransformation] = None
    sdfg: SDFG = None
    sampling_strategy: SamplingStrategy = SamplingStrategy.SIMPLE_UNIFORM
    output_dir: Optional[str] = None
    success_dir: Optional[str] = None

    _time_measurements: Dict[str, List[int]] = {
        'cutout': [],
        'transformation_apply': [],
        'compiling': [],
        'constraints': [],
        'sampling': [],
        'running_pre': [],
        'running_post': [],
        'comparing': [],
    }

    _build_dir_base_path: str = '/mnt/ramdisk'

    _cutout: Union[SDFG, SDFGCutout] = None
    _original_cutout: Union[SDFG, SDFGCutout] = None

    def __init__(
        self,
        xform: Union[SubgraphTransformation, PatternTransformation],
        sdfg: SDFG,
        sampling_strategy: SamplingStrategy = SamplingStrategy.SIMPLE_UNIFORM,
        output_dir: Optional[str] = None, success_dir: Optional[str] = None,
        build_dir_base_path: str = '/mnt/ramdisk'
    ):
        self.xform = xform
        self._orig_xform = deepcopy(xform)
        self.sdfg = sdfg
        self.sampling_strategy = sampling_strategy
        self.output_dir = output_dir
        self.success_dir = success_dir
        self._build_dir_base_path = build_dir_base_path

    def cutout(
        self, status: StatusLevel = StatusLevel.OFF,
        use_alibi_nodes: bool = False, reduce_input_config: bool = False
    ) -> Union[SDFG, SDFGCutout]:
        if self._cutout is None:
            if status >= StatusLevel.DEBUG:
                print('Finding ideal cutout')
            self._cutout = SDFGCutout.from_transformation(
                self.sdfg, self.xform, use_alibi_nodes=use_alibi_nodes,
                reduce_input_config=reduce_input_config
            )
            if status >= StatusLevel.DEBUG:
                print('Cutout obtained')
        return self._cutout


    def _catch_failure(
        self, reason: FailureReason, details: Optional[str],
        status: StatusLevel, inputs: Optional[Dict[str, Any]] = None,
        symbol_constraints: Optional[Dict[str, Any]] = None,
        symbols: Optional[Dict[str, Any]] = None,
        exception: Optional[Exception] = None,
        iteration: Optional[int] = None
    ) -> None:
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)

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

                if exception is not None:
                    traceback.print_tb(exception.__traceback__, file=f)

            if status >= StatusLevel.VERBOSE:
                print('Saving cutouts')

            for sd in self._original_cutout.all_sdfgs_recursive():
                for state in sd.states():
                    for dn in state.data_nodes():
                        dn.instrument = \
                            dace.DataInstrumentationType.No_Instrumentation
            for sd in self._cutout.all_sdfgs_recursive():
                for state in sd.states():
                    for dn in state.data_nodes():
                        dn.instrument = \
                            dace.DataInstrumentationType.No_Instrumentation

            if reason == FailureReason.EXCEPTION:
                self.sdfg.save(os.path.join(self.output_dir, 'orig.sdfg'))

                if status >= StatusLevel.VERBOSE:
                    print('Saving transformation')
                with open(os.path.join(
                    self.output_dir, 'xform.json'
                ), 'w') as f:
                    json.dump(self.xform.to_json(), f, indent=4)
                return

            self._cutout.save(os.path.join(self.output_dir, 'post.sdfg'))
            self._original_cutout.save(
                os.path.join(self.output_dir, 'pre.sdfg')
            )

            if status >= StatusLevel.VERBOSE:
                print('Saving inputs for debugging purposes')
            if inputs is not None:
                with open(os.path.join(self.output_dir, 'inputs'), 'wb') as f:
                    pickle.dump(inputs, f, protocol=pickle.HIGHEST_PROTOCOL)
            if symbol_constraints is not None:
                with open(os.path.join(
                    self.output_dir, 'constraints'
                ), 'wb') as f:
                    pickle.dump(
                        symbol_constraints, f,
                        protocol=pickle.HIGHEST_PROTOCOL
                    )
            if symbols is not None:
                with open(os.path.join(self.output_dir, 'symbols'), 'wb') as f:
                    pickle.dump(
                        symbols, f, protocol=pickle.HIGHEST_PROTOCOL
                    )

            if status >= StatusLevel.VERBOSE:
                print('Saving transformation')
            with open(os.path.join(self.output_dir, 'xform.json'), 'w') as f:
                json.dump(self.xform.to_json(), f, indent=4)

            if (inputs is not None and symbol_constraints is not None and
                symbols is not None):
                if status >= StatusLevel.VERBOSE:
                    print('Generateing harness')

                init_args = {}
                for name in inputs.keys():
                    init_args[name] = 'rand'

                try:
                    sdfg2cpp.dump_args('c++',
                                       os.path.join(self.output_dir, 'harness'),
                                       init_args, symbol_constraints,
                                       self._cutout, self._original_cutout,
                                       **inputs, **symbols)
                except Exception:
                    pass


    def _do_verify(
        self, n_samples: int = 1, status: StatusLevel = StatusLevel.OFF,
        debug_save_path: str = None, enforce_finiteness: bool = False,
        symbol_constraints: Dict = None, data_constraints: Dict = None,
        strict_config: bool = False, use_alibi_nodes: bool = False,
        reduce_input_config: bool = False, maximum_data_dim: int = 128
    ) -> bool:
        t0 = time.perf_counter_ns()
        cutout = self.cutout(
            status=status, use_alibi_nodes=use_alibi_nodes,
            reduce_input_config=reduce_input_config
        )
        self._time_measurements['cutout'].append(time.perf_counter_ns() - t0)
        orig_cutout = deepcopy(cutout)
        self._original_cutout = orig_cutout
        if status >= StatusLevel.DEBUG:
            print('Applying transformation')
        try:
            t0 = time.perf_counter_ns()
            apply_transformation(cutout, self.xform)
            self._time_measurements['transformation_apply'].append(
                time.perf_counter_ns() - t0
            )
        except InvalidSDFGError as e:
            self._catch_failure(
                FailureReason.FAILED_VALIDATE, str(e), status, exception=e
            )
            return False
        except Exception as e:
            self._catch_failure(
                FailureReason.FAILED_TO_APPLY, str(e), status, exception=e
            )
            return False

        cutout._name = cutout.name + '_transformed'

        if isinstance(orig_cutout, SDFGCutout):
            for name in orig_cutout.output_config:
                if name not in cutout.arrays:
                    print(
                        'Warning: Transformation removed something from the ' +
                        'output configuration!'
                    )
                    cutout.add_datadesc(name, orig_cutout.arrays[name])
        else:
            # Cutout is equivalent to the entire SDFG.
            for name, desc in orig_cutout.arrays.items():
                if not desc.transient:
                    if name not in cutout.arrays:
                        print(
                            'Warning: Transformation removed something from ' +
                            'the output configuration!'
                        )
                        cutout.add_datadesc(name, orig_cutout.arrays[name])
                    if cutout.arrays[name].transient:
                        cutout.arrays[name].transient = False

        if debug_save_path is not None:
            self._original_cutout.save(debug_save_path + '_orig.sdfg')
            cutout.save(debug_save_path + '_xformed.sdfg')

        orig_in_config: Set[str] = set()
        new_in_config: Set[str] = set()
        output_config: Set[str] = set()
        if strict_config and isinstance(orig_cutout, SDFGCutout):
            orig_in_config = orig_cutout.input_config
            output_config = orig_cutout.output_config
        else:
            for name, desc in orig_cutout.arrays.items():
                if not desc.transient:
                    orig_in_config.add(name)
                    output_config.add(name)
        if isinstance(cutout, SDFGCutout):
            new_in_config = cutout.input_config
        else:
            for name, desc in cutout.arrays.items():
                if not desc.transient:
                    new_in_config.add(name)

        for sd in orig_cutout.all_sdfgs_recursive():
            for state in sd.states():
                for dn in state.data_nodes():
                    if dn.data in output_config:
                        dn.instrument = dace.DataInstrumentationType.Save
        for sd in cutout.all_sdfgs_recursive():
            for state in sd.states():
                for dn in state.data_nodes():
                    if dn.data in output_config:
                        dn.instrument = dace.DataInstrumentationType.Save

        t0 = time.perf_counter_ns()
        if status >= StatusLevel.DEBUG:
            print('Compiling pre-transformation cutout')
        prog_orig: CompiledSDFG = None
        try:
            prog_orig = orig_cutout.compile()
        except InvalidSDFGError as e:
            print('Failure during validation of original cutout')
            raise e
        except Exception as e:
            print('Failure during compilation')
            raise e
        if status >= StatusLevel.DEBUG:
            print('Compiling post-transformation cutout')
        prog_xformed: CompiledSDFG = None
        try:
            prog_xformed = cutout.compile(validate=True)
        except InvalidSDFGError as e:
            self._catch_failure(
                FailureReason.FAILED_VALIDATE, str(e), status, exception=e
            )
            return False
        except Exception as e:
            self._catch_failure(
                FailureReason.COMPILATION_FAILURE, str(e), status, exception=e
            )
            return False
        if status >= StatusLevel.DEBUG:
            print(
                'Verifying transformation over', n_samples,
                'sampling run' + ('s' if n_samples > 1 else '')
            )
        self._time_measurements['compiling'].append(time.perf_counter_ns() - t0)

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

        t0 = time.perf_counter_ns()
        cutout_symbol_constraints = cutout_determine_symbol_constraints(
            cutout, self.sdfg, pre_constraints=general_constraints,
            max_dim=maximum_data_dim
        )
        self._time_measurements['constraints'].append(
            time.perf_counter_ns() - t0
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
                t0 = time.perf_counter_ns()
                symbols_map, free_symbols_map = sampler.sample_symbols_map_for(
                    orig_cutout, constraints_map=cutout_symbol_constraints,
                    maxval=maximum_data_dim
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
                    orig_cutout, orig_in_config, symbols_map, decay_by,
                    constraints_map=constraints_map
                )

                if status >= StatusLevel.VERBOSE:
                    bar.write(
                        'Duplicating inputs for post-transformation cutout'
                    )
                inputs_xformed = dict()
                for k, v in inputs.items():
                    if not strict_config or k in new_in_config:
                        inputs_xformed[k] = deepcopy(v)
                inputs_save = deepcopy(inputs)

                if status >= StatusLevel.VERBOSE:
                    bar.write(
                        'Generating outputs for pre-transformation cutout'
                    )
                out_orig = sampler.generate_output_containers(
                    orig_cutout, output_config, orig_in_config, symbols_map
                )
                if status >= StatusLevel.VERBOSE:
                    bar.write(
                        'Generating outputs for post-transformation cutout'
                    )
                out_xformed = sampler.generate_output_containers(
                    cutout, output_config, new_in_config, symbols_map
                )
                self._time_measurements['sampling'].append(
                    time.perf_counter_ns() - t0
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
                t0 = time.perf_counter_ns()
                ret_orig = run_subprocess_precompiled(
                    prog_orig, orig_containers, free_symbols_map, status=status
                )
                self._time_measurements['running_pre'].append(
                    time.perf_counter_ns() - t0
                )
                if status >= StatusLevel.VERBOSE:
                    bar.write(
                        'Collecting pre-transformation cutout data reports'
                    )

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
                t0 = time.perf_counter_ns()
                ret_xformed = run_subprocess_precompiled(
                    prog_xformed, xformed_containers, free_symbols_map,
                    status=status
                )
                self._time_measurements['running_post'].append(
                    time.perf_counter_ns() - t0
                )
                if status >= StatusLevel.VERBOSE:
                    bar.write(
                        'Collecting post-transformation cutout data reports'
                    )

                t0 = time.perf_counter_ns()
                if status >= StatusLevel.VERBOSE:
                    bar.write('Comparing results')
                if ((ret_orig != 0 and ret_xformed == 0) or
                    (ret_orig == 0 and ret_xformed != 0)):
                    if status >= StatusLevel.VERBOSE:
                        bar.write('One cutout failed to run, the other ran!')
                    self._catch_failure(
                        FailureReason.EXIT_CODE_MISMATCH,
                        f'Exit code (${ret_xformed}) does not match oringinal' +
                        f' exit code (${ret_orig})',
                        status, inputs_save, cutout_symbol_constraints,
                        free_symbols_map
                    )
                    self._time_measurements['comparing'].append(
                        time.perf_counter_ns() - t0
                    )
                    return False
                elif ret_orig == 0 and ret_xformed == 0:
                    resample = False
                    orig_drep: dace.InstrumentedDataReport = \
                        orig_cutout.get_instrumented_data()
                    xf_drep: dace.InstrumentedDataReport = \
                        cutout.get_instrumented_data()
                    if orig_drep is None or xf_drep is None:
                        print('No data reports!')
                        self._catch_failure(
                            FailureReason.EXCEPTION, 'No data reports', status,
                            inputs_save, cutout_symbol_constraints,
                            free_symbols_map
                        )
                        return False
                    for dat in output_config:
                        try:
                            oval = data_report_get_latest_version(
                                orig_drep, dat
                            )
                            nval = data_report_get_latest_version(
                                xf_drep, dat
                            )

                            if isinstance(oval, sp.Basic):
                                resample = True
                                break
                            if isinstance(nval, sp.Basic):
                                resample = True
                                break

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
                                        f'Mismatching results for ${dat}',
                                        status, inputs_save,
                                        cutout_symbol_constraints,
                                        free_symbols_map
                                    )
                                    self._time_measurements['comparing'].append(
                                        time.perf_counter_ns() - t0
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
                                        f'Mismatching results for ${dat}',
                                        status, inputs_save,
                                        cutout_symbol_constraints,
                                        free_symbols_map
                                    )
                                    self._time_measurements['comparing'].append(
                                        time.perf_counter_ns() - t0
                                    )
                                    return False
                        except KeyError:
                            if strict_config:
                                print(
                                    'WARNING: Missing instrumentation on ' +
                                    'system state for container',
                                    dat
                                )
                            elif status >= StatusLevel.VERBOSE:
                                bar.write(
                                    'No instrumentation on system state ' +
                                    'for container',
                                    dat
                                )
                        except StructError as e:
                            print(e)
                            if status >= StatusLevel.VERBOSE:
                                bar.write(
                                    'Exception when reading data back',
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
                self._time_measurements['comparing'].append(
                    time.perf_counter_ns() - t0
                )
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

        if self.success_dir is not None:
            if status >= StatusLevel.VERBOSE:
                print(
                    'No problem found, sampling another input set to save for ',
                    'potentially fuzzing externally.'
                )
            symbols_map, free_symbols_map = sampler.sample_symbols_map_for(
                orig_cutout, constraints_map=cutout_symbol_constraints,
                maxval=maximum_data_dim
            )

            constraints_map = None
            if data_constraints is not None:
                constraints_map = {
                    k: (pystr_to_symbolic(lval), pystr_to_symbolic(hval))
                    for k, (lval, hval) in data_constraints.items()
                }
            if status >= StatusLevel.VERBOSE:
                print('Sampling inputs')
            orig_in_config: Set[str] = set()
            new_in_config: Set[str] = set()
            output_config: Set[str] = set()
            if strict_config and isinstance(orig_cutout, SDFGCutout):
                orig_in_config = orig_cutout.input_config
                output_config = orig_cutout.output_config
            else:
                for name, desc in orig_cutout.arrays.items():
                    if not desc.transient:
                        orig_in_config.add(name)
                        output_config.add(name)
            if isinstance(cutout, SDFGCutout):
                new_in_config = cutout.input_config
            else:
                for name, desc in cutout.arrays.items():
                    if not desc.transient:
                        new_in_config.add(name)

            inputs = sampler.sample_inputs(
                orig_cutout, orig_in_config, symbols_map, decay_by,
                constraints_map=constraints_map
            )
            init_args = {}
            for name in inputs.keys():
                init_args[name] = 'rand'

            if status >= StatusLevel.VERBOSE:
                print('Saving cutouts')

            for sd in orig_cutout.all_sdfgs_recursive():
                for state in sd.states():
                    for dn in state.data_nodes():
                        dn.instrument = \
                            dace.DataInstrumentationType.No_Instrumentation
            for sd in cutout.all_sdfgs_recursive():
                for state in sd.states():
                    for dn in state.data_nodes():
                        dn.instrument = \
                            dace.DataInstrumentationType.No_Instrumentation

            cutout.save(os.path.join(self.success_dir, 'post.sdfg'))
            orig_cutout.save(os.path.join(self.success_dir, 'pre.sdfg'))

            if status >= StatusLevel.VERBOSE:
                print('Saving inputs for debugging purposes')
            with open(os.path.join(self.success_dir, 'inputs'), 'wb') as f:
                pickle.dump(inputs, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open(os.path.join(self.success_dir, 'constraints'), 'wb') as f:
                pickle.dump(
                    cutout_symbol_constraints, f,
                    protocol=pickle.HIGHEST_PROTOCOL
                )
            with open(os.path.join(self.success_dir, 'symbols'), 'wb') as f:
                pickle.dump(
                    free_symbols_map, f, protocol=pickle.HIGHEST_PROTOCOL
                )

            if status >= StatusLevel.VERBOSE:
                print('Saving transformation')
            with open(os.path.join(self.success_dir, 'xform.json'), 'w') as f:
                json.dump(self.xform.to_json(), f, indent=4)

            if status >= StatusLevel.VERBOSE:
                print('Generateing harness')

            try:
                sdfg2cpp.dump_args('c++',
                                   os.path.join(self.success_dir, 'harness'),
                                   init_args, cutout_symbol_constraints, cutout,
                                   orig_cutout, **inputs, **free_symbols_map)
            except Exception:
                with open(os.path.join(self.success_dir, 'EXCEPT'), 'w') as f:
                    f.write('Failed to generate harness')
                    traceback.print_exc(file=f)

        return True


    def verify(
        self, n_samples: int = 1, status: StatusLevel = StatusLevel.OFF,
        debug_save_path: str = None, enforce_finiteness: bool = False,
        symbol_constraints: Dict = None, data_constraints: Dict = None,
        strict_config: bool = False, minimize_input: bool = False,
        use_alibi_nodes: bool = False, maximum_data_dim: int = 128
    ) -> Tuple[bool, int]:
        with tempfile.TemporaryDirectory(
            prefix='fuzzyflow_dacecache_',
            suffix='_' + self.xform.__class__.__name__,
            dir=self._build_dir_base_path,
        ) as build_dir:
            with config.temporary_config():
                config.Config.set(
                    'compiler',
                    'cpu',
                    'args',
                    value='-std=c++14 -fPIC -Wall -Wextra -O2 ' +
                        '-Wno-unused-parameter -Wno-unused-label'
                )
                config.Config.set(
                    'compiler', 'allow_view_arguments', value=True
                )
                config.Config.set('profiling', value=False)
                config.Config.set('debugprint', value=False)
                config.Config.set('cache', value='name')
                config.Config.set('default_build_folder', value=build_dir)
                with warnings.catch_warnings():
                    # Ignore library already loaded warnings.
                    warnings.simplefilter(
                        'ignore', lineno=104, append=True
                    )
                    # Ignore typecasting warnings.
                    warnings.simplefilter(
                        'ignore', lineno=393, append=True
                    )
                    try:
                        start_validate = time.perf_counter_ns()
                        retval = self._do_verify(
                            n_samples, status, debug_save_path,
                            enforce_finiteness,
                            symbol_constraints, data_constraints, strict_config,
                            use_alibi_nodes=use_alibi_nodes,
                            reduce_input_config=minimize_input,
                            maximum_data_dim=maximum_data_dim
                        )
                        end_validate = time.perf_counter_ns()
                        dt = end_validate - start_validate
                        for k, v in self._time_measurements.items():
                            print(
                                k + ': median ', str(np.median(v) / 1e9),
                                's, mean ', str(np.mean(v) / 1e9),
                                's, std ', str(np.std(v) / 1e9), 's'
                            )
                        return retval, dt
                    except Exception as e:
                        self._catch_failure(
                            FailureReason.EXCEPTION, str(e), status, exception=e
                        )
