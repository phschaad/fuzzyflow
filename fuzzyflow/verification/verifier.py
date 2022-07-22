# Copyright 2022 ETH Zurich and the FuzzyFlow authors. All rights reserved.
# This file is part of FuzzyFlow, which is released under the BSD 3-Clause
# License. For details, see the LICENSE file.

from copy import deepcopy
from dace.sdfg import SDFG, nodes as nd
from dace.transformation.transformation import SubgraphTransformation, PatternTransformation
from typing import List, Union

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


    def cutout(self, strategy: CutoutStrategy = None) -> SDFG:
        recut = False
        if strategy is not None and strategy != self.cutout_strategy:
            self.cutout_strategy = strategy
            recut = True

        if self._cutout is None or recut:
            self._cutout = find_cutout_for_transformation(
                self.sdfg, self.xform, self.cutout_strategy
            )

            in_nodes: List[nd.AccessNode] = self._cutout.input_arrays()
            for in_node in in_nodes:
                self._cutout.arrays[in_node.data].transient = False
            out_nodes: List[nd.AccessNode] = self._cutout.output_arrays()
            for out_node in out_nodes:
                self._cutout.arrays[out_node.data].transient = False

        return self._cutout


    def verify(self, n_samples: int = 1) -> bool:
        cutout = self.cutout()
        original_cutout = deepcopy(cutout)
        apply_transformation(cutout, self.xform)

        seed = 12121
        sampler = DataSampler(self.sampling_strategy, seed)

        for i in range(n_samples):
            symbols_map = sampler.sample_symbols_map_for(original_cutout)
            inputs = sampler.sample_inputs_for(original_cutout, symbols_map)
            original_outputs = sampler.generate_output_containers_for(
                original_cutout, symbols_map
            )
            outputs = sampler.generate_output_containers_for(
                original_cutout, symbols_map
            )

            original_cutout(**inputs, **original_outputs, **symbols_map)
            cutout(**inputs, **outputs, **symbols_map)

            for k in original_outputs.keys():
                oval = original_outputs[k]
                nval = outputs[k]
                if (oval != nval).any():
                    return False

            print('Run', i + 1, 'of', n_samples, 'successful')

        return True
