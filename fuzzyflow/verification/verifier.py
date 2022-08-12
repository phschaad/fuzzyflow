# Copyright 2022 ETH Zurich and the FuzzyFlow authors. All rights reserved.
# This file is part of FuzzyFlow, which is released under the BSD 3-Clause
# License. For details, see the LICENSE file.

from alive_progress import alive_bar
from copy import deepcopy
from typing import List, Union
import numpy as np

from dace import config
from dace.codegen.compiled_sdfg import CompiledSDFG
from dace.sdfg import SDFG
from dace.sdfg import nodes as nd
from dace.transformation.transformation import (PatternTransformation,
                                                SubgraphTransformation)

from fuzzyflow.cutout import CutoutStrategy, find_cutout_for_transformation
from fuzzyflow.util import apply_transformation
from fuzzyflow.verification.sampling import DataSampler, SamplingStrategy


class TransformationVerifier:

    xform: Union[SubgraphTransformation, PatternTransformation] = None
    sdfg: SDFG = None
    cutout_strategy: CutoutStrategy = CutoutStrategy.SIMPLE
    sampling_strategy: SamplingStrategy = SamplingStrategy.SIMPLE_UNIFORM

    _cutout: SDFG = None

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
            self._cutout = find_cutout_for_transformation(
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
        original_cutout = deepcopy(cutout)
        if status:
            print('Applying transformation')
        apply_transformation(cutout, self.xform)

        if debug_save_path is not None:
            original_cutout.save(debug_save_path + '_orig.sdfg')
            cutout.save(debug_save_path + '_xformed.sdfg')

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
                inputs = sampler.sample_inputs_for(
                    original_cutout, symbols_map, decay_by=decay_by
                )
                inputs_xformed = deepcopy(inputs)
                out_orig = sampler.generate_output_containers_for(
                    original_cutout, symbols_map
                )
                out_xformed = sampler.generate_output_containers_for(
                    cutout, symbols_map
                )

                orig_containers = dict()
                for k in inputs.keys():
                    if not k in orig_containers:
                        orig_containers[k] = inputs[k]
                for k in out_orig.keys():
                    if not k in orig_containers:
                        orig_containers[k] = out_orig[k]
                pass
                prog_orig.__call__(
                    **orig_containers, **free_symbols_map
                )

                xformed_containers = dict()
                for k in inputs.keys():
                    if not k in xformed_containers:
                        xformed_containers[k] = inputs_xformed[k]
                for k in out_xformed.keys():
                    if not k in xformed_containers:
                        xformed_containers[k] = out_xformed[k]
                pass
                prog_xformed.__call__(
                    **xformed_containers, **free_symbols_map
                )

                resample = False
                for k in orig_containers.keys():
                    # Skip any containers that don't exist in the new cutout
                    # TODO: This is probably not how we should do it...
                    if k not in xformed_containers:
                        continue
                    oval = orig_containers[k]
                    nval = xformed_containers[k]

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
            return self._do_verify(
                n_samples, status, debug_save_path, enforce_finiteness
            )
