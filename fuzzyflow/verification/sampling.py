# Copyright 2022 ETH Zurich and the FuzzyFlow authors. All rights reserved.
# This file is part of FuzzyFlow, which is released under the BSD 3-Clause
# License. For details, see the LICENSE file.

import random
from enum import Enum
from typing import Dict, List, Set, Tuple, Union

import numpy as np
from dace import dtypes as ddtypes
from dace.data import Data, Scalar, make_array_from_descriptor, Array
from dace.sdfg import SDFG
from dace.sdfg import nodes as nd
from dace.symbolic import symbol
from dace.libraries.standard.memory import aligned_ndarray
from sympy.core import Expr
from sympy.core.numbers import Number


class SamplingStrategy(Enum):
    SIMPLE_UNIFORM = 'SIMPLE_UNIFORM'

    def __str__(self):
        return self.value


class DataSampler:

    strategy: SamplingStrategy = None
    random_state: np.random.RandomState = None

    def __init__(
        self,
        strategy: SamplingStrategy = SamplingStrategy.SIMPLE_UNIFORM,
        seed: int = None
    ):
        self.strategy = strategy
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        self.random_state = np.random.RandomState(seed)


    def _uniform_samling(
        self, name: str, array: Data, shape: Union[List, Tuple],
        decay_by: int = 0
    ) -> Union[np.ndarray, np.number]:
        npdt = array.dtype.as_numpy_dtype()
        if npdt in [np.float16, np.float32, np.float64]:
            # TODO: We can't sample directly over [float64.min, float64.max],
            # because `np.uniform` would complain about an overflow. We need to
            # find an alternative way of ensuring the entire float64 spectrum
            # is covered and sampled.
            sample_dtype = npdt
            if npdt == np.float64 and decay_by == 0:
                sample_dtype = np.float32

            low = np.finfo(sample_dtype).min * (2 ** -decay_by)
            high = np.finfo(sample_dtype).max * (2 ** -decay_by)
            if isinstance(array, Scalar):
                return self.random_state.uniform(low=low, high=high)
            else:
                return self.random_state.uniform(
                    low=low, high=high, size=shape
                ).astype(npdt)
        elif npdt in [
            np.int8, np.int16, np.int32, np.int64,
            np.uint8, np.uint16, np.uint32, np.uint64
        ]:
            low = np.iinfo(npdt).min * (2 ** -decay_by)
            high = np.iinfo(npdt).max * (2 ** -decay_by)
            if isinstance(array, Scalar):
                return np.random.randint(low, high)
            else:
                return np.random.randint(low, high, size=shape).astype(npdt)
        elif array.dtype in [ddtypes.bool, ddtypes.bool_]:
            if isinstance(array, Scalar):
                return np.random.randint(low=0, high=2)
            else:
                return np.random.randint(low=0, high=2, size=shape).astype(npdt)
        if isinstance(array, Scalar):
            return np.random.rand()
        else:
            return np.random.rand(*shape)


    def _sample_data_for_nodes(
        self, nodes: List[nd.AccessNode], sdfg: SDFG,
        symbols_map: Dict[str, int], sample: bool = True,
        decay_by: int = 0
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
                if self.strategy == SamplingStrategy.SIMPLE_UNIFORM:
                    newdata = self._uniform_samling(
                        name, array, shape, decay_by
                    )
                else:
                    raise NotImplementedError()
            else:
                if isinstance(array, Scalar):
                    newdata = 0
                else:
                    newdata = np.zeros(shape).astype(
                        array.dtype.as_numpy_dtype()
                    )

            if isinstance(array, Scalar):
                retdict[name] = newdata
            else:
                view: np.ndarray = make_array_from_descriptor(
                    array, newdata, symbols_map
                )
                if isinstance(array, Array) and array.alignment:
                    aligned = aligned_ndarray(view, array.alignment)
                    retdict[name] = aligned
                else:
                    retdict[name] = view

        return retdict


    def sample_symbols_map_for(
        self, sdfg: SDFG, maxval: int = 128
    ) -> Dict[str, int]:
        symbol_map = dict()
        free_symbols_map = dict()
        for k, v in sdfg.constants.items():
            symbol_map[k] = int(v)
        for k in sdfg.free_symbols:
            symbol_map[k] = random.randint(0, maxval)
            free_symbols_map[k] = symbol_map[k]
        for k in sdfg.symbols:
            if k not in symbol_map:
                symbol_map[k] = 0
        return symbol_map, free_symbols_map


    def sample_inputs_for(
        self, sdfg: SDFG, symbols_map: Dict[str, int] = None,
        decay_by: int = 0
    ) -> Dict[str, np.ndarray]:
        if symbols_map is None:
            symbols_map = self.sample_symbols_map_for(sdfg)

        in_nodes = sdfg.input_arrays()
        for s in sdfg.states():
            for dn in s.data_nodes():
                if not sdfg.arrays[dn.data].transient and not dn in in_nodes:
                    in_nodes.append(dn)

        return self._sample_data_for_nodes(
            in_nodes, sdfg, symbols_map, sample=True, decay_by=decay_by
        )


    def generate_output_containers_for(
        self, sdfg: SDFG, symbols_map: Dict[str, int] = None
    ) -> Dict[str, np.ndarray]:
        if symbols_map is None:
            symbols_map = self.sample_symbols_map_for(sdfg)
        return self._sample_data_for_nodes(
            sdfg.output_arrays(), sdfg, symbols_map, sample=False
        )
