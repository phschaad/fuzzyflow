# Copyright 2022 ETH Zurich and the FuzzyFlow authors. All rights reserved.
# This file is part of FuzzyFlow, which is released under the BSD 3-Clause
# License. For details, see the LICENSE file.

from alive_progress import alive_bar
from copy import deepcopy
from typing import List, Union
import numpy as np

from dace import config, dtypes
from dace.codegen.compiled_sdfg import CompiledSDFG
from dace.data import Scalar
from dace.sdfg import SDFG
from dace.sdfg import nodes as nd
from dace.transformation.transformation import (PatternTransformation,
                                                SubgraphTransformation)

from fuzzyflow.cutout import CutoutStrategy, TranslationDict, cutout_determine_input_config, cutout_determine_system_state, find_cutout_for_transformation
from fuzzyflow.runner import run_subprocess_precompiled
from fuzzyflow.util import apply_transformation
from fuzzyflow.verification.sampling import DataSampler, SamplingStrategy


class TransformationVerifier:

    xform: Union[SubgraphTransformation, PatternTransformation] = None
    sdfg: SDFG = None
    cutout_strategy: CutoutStrategy = CutoutStrategy.SIMPLE
    sampling_strategy: SamplingStrategy = SamplingStrategy.SIMPLE_UNIFORM

    _cutout: SDFG = None
    _translation_dict: TranslationDict = None

    def __init__(
        self,
        xform: Union[SubgraphTransformation, PatternTransformation],
        sdfg: SDFG,
        cutout_strategy: CutoutStrategy = CutoutStrategy.SIMPLE,
        sampling_strategy: SamplingStrategy = SamplingStrategy.SIMPLE_UNIFORM
    ):
        self.xform = xform
        self.sdfg = sdfg
        self.cutout_strategy = cutout_strategy
        self.sampling_strategy = sampling_strategy


    def cutout(
        self, strategy: CutoutStrategy = None, status: bool = False
    ) -> SDFG:
        recut = False
        if strategy is not None and strategy != self.cutout_strategy:
            self.cutout_strategy = strategy
            recut = True

        if self._cutout is None or recut:
            if status:
                print('Finding ideal cutout')
            (
                self._cutout, self._translation_dict
            ) = find_cutout_for_transformation(
                self.sdfg, self.xform, self.cutout_strategy
            )

            in_nodes: List[nd.AccessNode] = self._cutout.input_arrays()
            for in_node in in_nodes:
                self._cutout.arrays[in_node.data].transient = False
            out_nodes: List[nd.AccessNode] = self._cutout.output_arrays()
            for out_node in out_nodes:
                self._cutout.arrays[out_node.data].transient = False
            if status:
                print('Cutout obtained')

        return self._cutout


    def _do_verify(
        self, n_samples: int = 1, status: bool = False,
        debug_save_path: str = None, enforce_finiteness: bool = False
    ) -> bool:
        cutout = self.cutout(status=status)
        system_state = cutout_determine_system_state(
            cutout, self.sdfg, self._translation_dict
        )
        original_input_configuration = cutout_determine_input_config(
            cutout, self.sdfg, self._translation_dict
        )

        original_cutout = deepcopy(cutout)
        if status:
            print('Applying transformation')
        apply_transformation(cutout, self.xform)
        for dat in system_state:
            if dat not in cutout.arrays.keys():
                print('Warning: Transformation removed something from system state!')
                orig_array = original_cutout.arrays[dat]
                cutout.add_datadesc(dat, orig_array)

        xformed_input_configuration = cutout_determine_input_config(
            cutout, self.sdfg, self._translation_dict, system_state
        )

        # Instrumentation
        for s in cutout.states():
            for dn in s.data_nodes():
                if dn.data in system_state:
                    dn.instrument = dtypes.DataInstrumentationType.Save
        for s in original_cutout.states():
            for dn in s.data_nodes():
                if dn.data in system_state:
                    dn.instrument = dtypes.DataInstrumentationType.Save

        if debug_save_path is not None:
            original_cutout.save(debug_save_path + '_orig.sdfg')
            cutout.save(debug_save_path + '_xformed.sdfg')

        if not xformed_input_configuration.issubset(
            original_input_configuration
        ):
            print(
                'Failed due to invalid input configuration after transforming!'
            )
            return False

        if status:
            print('Compiling pre-transformation cutout')
        prog_orig: CompiledSDFG = original_cutout.compile()
        if status:
            print('Compiling post-transformation cutout')
        prog_xformed: CompiledSDFG = cutout.compile()
        if status:
            print(
                'Verifying transformation over', n_samples,
                'sampling run' + ('s' if n_samples > 1 else '')
            )

        seed = 12121
        sampler = DataSampler(self.sampling_strategy, seed)

        with alive_bar(n_samples, disable=(not status)) as bar:
            i = 0
            resample_attempt = 0
            decay_by = 0
            decays = []
            full_resampling_failures = 0
            while i < n_samples:
                symbols_map, free_symbols_map = sampler.sample_symbols_map_for(
                    original_cutout
                )

                inputs = sampler.sample_inputs(
                    original_cutout, original_input_configuration, symbols_map,
                    decay_by
                )

                inputs_xformed = dict()
                for k, v in inputs.items():
                    if k in xformed_input_configuration:
                        inputs_xformed[k] = deepcopy(v)

                out_orig = sampler.generate_output_containers(
                    original_cutout, system_state, original_input_configuration,
                    symbols_map
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
                '''
                prog_orig.__call__(
                    **orig_containers, **free_symbols_map
                )
                '''
                ret_orig = run_subprocess_precompiled(
                    prog_orig, orig_containers, free_symbols_map
                )
                orig_drep = original_cutout.get_instrumented_data()

                xformed_containers = dict()
                for k in inputs.keys():
                    if k in cutout.arrays:
                        if not k in xformed_containers:
                            xformed_containers[k] = inputs_xformed[k]
                for k in out_xformed.keys():
                    if k in cutout.arrays:
                        if not k in xformed_containers:
                            xformed_containers[k] = out_xformed[k]
                '''
                prog_xformed.__call__(
                    **xformed_containers, **free_symbols_map
                )
                '''
                ret_xformed = run_subprocess_precompiled(
                    prog_xformed, xformed_containers, free_symbols_map
                )
                xformed_drep = cutout.get_instrumented_data()

                if ret_orig != ret_xformed:
                    return False
                elif ret_orig == 0 and ret_xformed == 0:
                    resample = False
                    for dat in system_state:

                        oval = orig_drep.get_latest_version(dat)
                        if dat in xformed_drep.files:
                            nval = xformed_drep.get_latest_version(dat)
                        else:
                            if isinstance(cutout.arrays[dat], Scalar):
                                nval = [inputs[dat]]
                            else:
                                nval = inputs[dat]

                        if enforce_finiteness and not np.isfinite(oval).all():
                            resample = True

                        if isinstance(oval, np.ndarray):
                            if not np.allclose(oval, nval, equal_nan=True):
                                return False
                        else:
                            if not np.allclose([oval], [nval], equal_nan=True):
                                return False

                    if not resample or resample_attempt > 11:
                        if resample_attempt > 11:
                            full_resampling_failures += 1
                        resample_attempt = 0
                        if enforce_finiteness and decay_by > 0:
                            decays.append(decay_by)
                        decay_by = 0
                        bar()
                        i += 1
                    else:
                        if resample_attempt >= 2:
                            if decay_by == 0:
                                decay_by = 1
                            else:
                                decay_by *= 2
                        resample_attempt += 1
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

        return True


    def verify(
        self, n_samples: int = 1, status: bool = False,
        debug_save_path: str = None, enforce_finiteness: bool = False
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
                n_samples, status, debug_save_path, enforce_finiteness
            )
