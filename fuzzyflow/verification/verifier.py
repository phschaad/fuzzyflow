# Copyright 2022 ETH Zurich and the FuzzyFlow authors. All rights reserved.
# This file is part of FuzzyFlow, which is released under the BSD 3-Clause
# License. For details, see the LICENSE file.

from copy import deepcopy
from collections import deque
from typing import Dict, List, Union, Optional, Set, Any
import os
import numpy as np
from tqdm import tqdm
from enum import Enum
import json
from struct import error as StructError
import pickle

import dace
from dace import config, dtypes
from dace.codegen.compiled_sdfg import CompiledSDFG
from dace.data import Scalar
from dace.sdfg import SDFG, nodes as nd
from dace.sdfg.analysis.cutout import SDFGCutout
from dace.transformation.transformation import (PatternTransformation,
                                                SubgraphTransformation)
from dace.symbolic import pystr_to_symbolic
from dace.sdfg.validation import InvalidSDFGError
from dace.codegen.instrumentation.data.data_report import InstrumentedDataReport

from fuzzyflow.runner import run_subprocess_precompiled
from fuzzyflow.util import (StatusLevel,
                            apply_transformation,
                            cutout_determine_symbol_constraints)
from fuzzyflow.verification.sampling import DataSampler, SamplingStrategy
from fuzzyflow.harness_generator import sdfg2cpp


class FailureReason(Enum):
    FAILED_VALIDATE = 'FAILED_VALIDATE'
    COMPILATION_FAILURE = 'COMPILATION_FAILURE'
    EXIT_CODE_MISMATCH = 'EXIT_CODE_MISMATCH'
    SYSTEM_STATE_MISMATCH = 'SYSTEM_STATE_MISMATCH'


class TransformationVerifier:

    xform: Union[SubgraphTransformation, PatternTransformation] = None
    sdfg: SDFG = None
    sampling_strategy: SamplingStrategy = SamplingStrategy.SIMPLE_UNIFORM
    output_dir: Optional[str] = None
    success_dir: Optional[str] = None

    _cutout: Union[SDFG, SDFGCutout] = None
    _original_cutout: Union[SDFG, SDFGCutout] = None

    def __init__(
        self,
        xform: Union[SubgraphTransformation, PatternTransformation],
        sdfg: SDFG,
        sampling_strategy: SamplingStrategy = SamplingStrategy.SIMPLE_UNIFORM,
        output_dir: Optional[str] = None, success_dir: Optional[str] = None
    ):
        self.xform = xform
        self.sdfg = sdfg
        self.sampling_strategy = sampling_strategy
        self.output_dir = output_dir
        self.success_dir = success_dir

    def cutout(
        self, status: StatusLevel = StatusLevel.OFF
    ) -> Union[SDFG, SDFGCutout]:
        if self._cutout is None:
            if status >= StatusLevel.DEBUG:
                print('Finding ideal cutout')
            self._cutout = SDFGCutout.from_transformation(
                self.sdfg, self.xform, use_alibi_nodes=False
            )
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
        nparr, view = report._read_array_file(file, npdtype)
        report.loaded_values[item, -1] = nparr
        return view


    def _catch_failure(
        self, reason: FailureReason, details: Optional[str],
        status: StatusLevel, inputs: Dict[str, Any],
        symbol_constraints: Dict[str, Any], symbols: Dict[str, Any]
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

            if status >= StatusLevel.VERBOSE:
                print('Saving cutouts')

            noinstr = dace.DataInstrumentationType.No_Instrumentation
            for sd in self._cutout.all_sdfgs_recursive():
                for s in sd.states():
                    s.symbol_instrument = noinstr
                    for dn in s.data_nodes():
                        dn.instrument = noinstr
            for sd in self._original_cutout.all_sdfgs_recursive():
                for s in sd.states():
                    s.symbol_instrument = noinstr
                    for dn in s.data_nodes():
                        dn.instrument = noinstr
            self._cutout.save(os.path.join(self.output_dir, 'post.sdfg'))
            self._original_cutout.save(
                os.path.join(self.output_dir, 'pre.sdfg')
            )

            if status >= StatusLevel.VERBOSE:
                print('Saving inputs for debugging purposes')
            with open(os.path.join(self.output_dir, 'inputs'), 'wb') as f:
                pickle.dump(inputs, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open(os.path.join(self.output_dir, 'constraints'), 'wb') as f:
                pickle.dump(
                    symbol_constraints, f,
                    protocol=pickle.HIGHEST_PROTOCOL
                )
            with open(os.path.join(self.output_dir, 'symbols'), 'wb') as f:
                pickle.dump(
                    symbols, f, protocol=pickle.HIGHEST_PROTOCOL
                )

            if status >= StatusLevel.VERBOSE:
                print('Saving transformation')
            with open(os.path.join(self.output_dir, 'xform.json'), 'w') as f:
                json.dump(self.xform.to_json(), f, indent=4)

            if status >= StatusLevel.VERBOSE:
                print('Generateing harness')

            init_args = {}
            for name in inputs.keys():
                init_args[name] = 'rand'

            sdfg2cpp.dump_args('c++', os.path.join(self.output_dir, 'harness'),
                               init_args, symbol_constraints, self._cutout,
                               self._original_cutout, **inputs, **symbols)


    def _do_verify(
        self, n_samples: int = 1, status: StatusLevel = StatusLevel.OFF,
        debug_save_path: str = None, enforce_finiteness: bool = False,
        symbol_constraints: Dict = None, data_constraints: Dict = None,
        strict_config: bool = False
    ) -> bool:
        cutout = self.cutout(status=status)
        orig_cutout = deepcopy(cutout)
        self._original_cutout = orig_cutout
        if status >= StatusLevel.DEBUG:
            print('Applying transformation')
        try:
            apply_transformation(cutout, self.xform)
        except InvalidSDFGError as e:
            self._catch_failure(FailureReason.FAILED_VALIDATE, str(e))
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
                    if (name not in cutout.arrays or
                        not cutout.arrays[name].transient):
                        print(
                            'Warning: Transformation removed something from ' +
                            'the output configuration!'
                        )
                        cutout.add_datadesc(name, orig_cutout.arrays[name])

        # Instrumentation.
        for s in cutout.states():
            for dn in s.data_nodes():
                if dn.data in cutout.output_config:
                    dn.instrument = dtypes.DataInstrumentationType.Save
        for s in orig_cutout.states():
            for dn in s.data_nodes():
                if dn.data in orig_cutout.output_config:
                    dn.instrument = dtypes.DataInstrumentationType.Save

        if debug_save_path is not None:
            self._original_cutout.save(debug_save_path + '_orig.sdfg')
            cutout.save(debug_save_path + '_xformed.sdfg')

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
                    orig_cutout, constraints_map=cutout_symbol_constraints,
                    maxval=128
                )

                constraints_map = None
                if data_constraints is not None:
                    constraints_map = {
                        k: (pystr_to_symbolic(lval), pystr_to_symbolic(hval))
                        for k, (lval, hval) in data_constraints.items()
                    }

                if status >= StatusLevel.VERBOSE:
                    bar.write('Sampling inputs')
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

                if status >= StatusLevel.VERBOSE:
                    bar.write(
                        'Duplicating inputs for post-transformation cutout'
                    )
                inputs_xformed = dict()
                for k, v in inputs.items():
                    if not strict_config or k in new_in_config:
                        inputs_xformed[k] = deepcopy(v)

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
                orig_drep = orig_cutout.get_instrumented_data()

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
                    for dat in output_config:
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
                maxval=128
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

            noinstr = dace.DataInstrumentationType.No_Instrumentation
            for sd in cutout.all_sdfgs_recursive():
                for s in sd.states():
                    s.symbol_instrument = noinstr
                    for dn in s.data_nodes():
                        dn.instrument = noinstr
            for sd in orig_cutout.all_sdfgs_recursive():
                for s in sd.states():
                    s.symbol_instrument = noinstr
                    for dn in s.data_nodes():
                        dn.instrument = noinstr
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

            sdfg2cpp.dump_args('c++', os.path.join(self.success_dir, 'harness'),
                               init_args, cutout_symbol_constraints, cutout,
                               orig_cutout, **inputs, **free_symbols_map)

        return True


    def verify(
        self, n_samples: int = 1, status: StatusLevel = StatusLevel.OFF,
        debug_save_path: str = None, enforce_finiteness: bool = False,
        symbol_constraints: Dict = None, data_constraints: Dict = None,
        strict_config: bool = False
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
            config.Config.set('cache', value='name')
            return self._do_verify(
                n_samples, status, debug_save_path, enforce_finiteness,
                symbol_constraints, data_constraints, strict_config
            )
