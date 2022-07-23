# Copyright 2022 ETH Zurich and the FuzzyFlow authors. All rights reserved.
# This file is part of FuzzyFlow, which is released under the BSD 3-Clause
# License. For details, see the LICENSE file.

from enum import Flag
import random
from typing import Dict, List, Set, Tuple
from dace.sdfg import SDFG, nodes as nd
from dace.data import Data
from dace.symbolic import symbol
import numpy as np
from sympy.core import Expr
from sympy.core.numbers import Number


class SamplingStrategy(Flag):
    SIMPLE_UNIFORM = 0


class DataSampler:

    strategy: SamplingStrategy = None

    def __init__(
        self,
        strategy: SamplingStrategy = SamplingStrategy.SIMPLE_UNIFORM,
        seed: int = None
    ):
        self.strategy = strategy
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)


    def _sample_data_for_nodes(
        self, nodes: List[nd.AccessNode], sdfg: SDFG,
        symbols_map: Dict[str, int], sample: bool = True
    ) -> Dict[str, np.ndarray]:
        retdict = dict()
        data: Set[Tuple[str, Data]] = set()
        for node in nodes:
            array = sdfg.arrays[node.data]
            data.add((node.data, array))

        for dat in data:
            name, array = dat
            shape = []
            for x in array.shape:
                if isinstance(x, symbol):
                    if x.name in symbols_map:
                        shape.append(symbols_map[x.name])
                    else:
                        raise Exception(
                            'Can\'t find a definition for symbol', x.name
                        )
                elif isinstance(x, Expr):
                    res = x.subs(symbols_map)
                    if isinstance(res, Number) and res.is_Integer:
                        shape.append(int(res))
                    else:
                        raise Exception(
                            'Can\'t evaluate shape expression', x
                        )
                else:
                    shape.append(x)
            if sample:
                retdict[name] = np.random.rand(*shape)
            else:
                retdict[name] = np.zeros(shape)

        return retdict


    def sample_symbols_map_for(
        self, sdfg: SDFG, maxval: int = 256
    ) -> Dict[str, int]:
        symbol_map = dict()
        free_symbols_map = dict()
        for k, v in sdfg.constants.items():
            symbol_map[k] = int(v)
        for k in sdfg.free_symbols:
            symbol_map[k] = random.randint(0, maxval)
            free_symbols_map[k] = symbol_map[k]
        return symbol_map, free_symbols_map


    def sample_inputs_for(
        self, sdfg: SDFG, symbols_map: Dict[str, int] = None
    ) -> Dict[str, np.ndarray]:
        if symbols_map is None:
            symbols_map = self.sample_symbols_map_for(sdfg)
        return self._sample_data_for_nodes(
            sdfg.input_arrays(), sdfg, symbols_map
        )


    def generate_output_containers_for(
        self, sdfg: SDFG, symbols_map: Dict[str, int] = None
    ) -> Dict[str, np.ndarray]:
        if symbols_map is None:
            symbols_map = self.sample_symbols_map_for(sdfg)
        return self._sample_data_for_nodes(
            sdfg.output_arrays(), sdfg, symbols_map, False
        )
