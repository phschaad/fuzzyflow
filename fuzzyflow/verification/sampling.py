# Copyright 2022 ETH Zurich and the FuzzyFlow authors. All rights reserved.
# This file is part of FuzzyFlow, which is released under the BSD 3-Clause
# License. For details, see the LICENSE file.

from collections import deque
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
        self, array: Data, shape: Union[List, Tuple], decay_by: int = 0,
        constraints: Tuple[Number, Number] = None
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
            if constraints is not None:
                low, high = constraints
            if isinstance(array, Scalar):
                if low == high:
                    return low
                return self.random_state.uniform(low=low, high=high)
            else:
                if low == high:
                    rval = self.random_state.uniform(size=shape).astype(npdt)
                    rval.fill(low)
                    return rval.astype(npdt)
                return self.random_state.uniform(
                    low=low, high=high, size=shape
                ).astype(npdt)
        elif npdt in [
            np.int8, np.int16, np.int32, np.int64,
            np.uint8, np.uint16, np.uint32, np.uint64
        ]:
            low = np.iinfo(npdt).min * (2 ** -decay_by)
            high = np.iinfo(npdt).max * (2 ** -decay_by)
            if constraints is not None:
                low, high = constraints
            if isinstance(array, Scalar):
                if low == high:
                    return low
                return np.random.randint(low, high)
            else:
                if low == high:
                    rval = np.random.randint(
                        low=0, high=1, size=shape
                    ).astype(npdt)
                    rval.fill(low)
                    return rval.astype(npdt)
                return np.random.randint(low, high, size=shape).astype(npdt)
        elif array.dtype in [ddtypes.bool, ddtypes.bool_]:
            if isinstance(array, Scalar):
                return np.random.randint(low=0, high=2)
            else:
                return np.random.randint(low=0, high=2, size=shape).astype(npdt)

        if constraints is not None:
            low, high = constraints
            if isinstance(array, Scalar):
                if low == high:
                    return low
                return self.random_state.uniform(low=low, high=high)
            else:
                if low == high:
                    rval = self.random_state.uniform(size=shape)
                    rval.fill(low)
                    return rval
                return self.random_state.uniform(low=low, high=high, size=shape)
        else:
            if isinstance(array, Scalar):
                return np.random.rand()
            else:
                return np.random.rand(*shape)


    def _get_container_shape(
        self, array: Data, symbols_map: Dict[str, int]
    ) -> List:
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
        return shape


    def _sample_container(
        self, array: Data, symbols_map: Dict[str, int], decay_by: int = 0,
        constraints: Tuple[int, int] = None
    ) -> np.ndarray:
        shape = self._get_container_shape(array, symbols_map)
        newdata = None
        if self.strategy == SamplingStrategy.SIMPLE_UNIFORM:
            newdata = self._uniform_samling(array, shape, decay_by, constraints)
        else:
            raise NotImplementedError()

        if isinstance(array, Scalar):
            return newdata
        else:
            return self._align_container(array, symbols_map, newdata)


    def _empty_container(
        self, array: Data, symbols_map: Dict[str, int]
    ) -> Union[int, float, np.ndarray]:
        shape = self._get_container_shape(array, symbols_map)
        if isinstance(array, Scalar):
            npdt = array.dtype.as_numpy_dtype()
            if npdt in [np.float16, np.float32, np.float64]:
                return 0.0
            else:
                return 0
        else:
            empty_container = np.zeros(shape).astype(
                array.dtype.as_numpy_dtype()
            )
            return self._align_container(array, symbols_map, empty_container)


    def _align_container(
        self,  array: Data, symbols_map: Dict[str, int], container: np.ndarray
    ) -> np.ndarray:
        view: np.ndarray = make_array_from_descriptor(
            array, container, symbols_map
        )
        if isinstance(array, Array) and array.alignment:
            return aligned_ndarray(view, array.alignment)
        else:
            return view


    def sample_symbols_map_for(
        self, sdfg: SDFG, o_sdfg: SDFG, maxval: int = 1024,
        constraints_map: Dict = None
    ) -> Tuple[Dict[str, int], Dict[str, int]]:
        cutoff = 10

        symbol_map = dict()
        free_symbols_map = dict()
        for k, v in sdfg.constants.items():
            symbol_map[k] = int(v)

        deferred = deque()
        for k in sdfg.free_symbols.union(o_sdfg.free_symbols):
            if k in constraints_map:
                deferred.append((k, constraints_map[k], 0))
            else:
                symbol_map[k] = random.randint(-maxval, maxval)
                free_symbols_map[k] = symbol_map[k]

        while len(deferred) > 0:
            k, (low, high, step), count = deferred.popleft()
            retlow = None
            if isinstance(low, symbol):
                if str(low) in symbol_map:
                    retlow = symbol_map[str(low)]
            elif isinstance(low, Expr):
                res = low.subs(symbol_map)
                if isinstance(res, Number) and res.is_Integer:
                    retlow = int(res)
            elif isinstance(low, int):
                retlow = low
            rethigh = None
            if isinstance(high, symbol):
                if str(high) in symbol_map:
                    rethigh = symbol_map[str(high)]
            elif isinstance(high, Expr):
                res = high.subs(symbol_map)
                if isinstance(res, Number) and res.is_Integer:
                    rethigh = int(res)
            elif isinstance(high, int):
                rethigh = high
            if retlow is None or rethigh is None or retlow > rethigh:
                if count < cutoff:
                    deferred.append((k, (low, high, step), count + 1))
                else:
                    symbol_map[k] = random.randint(-maxval, maxval)
                    free_symbols_map[k] = symbol_map[k]
                    print(
                        'Warning: sampling dependent symbol between',
                        str(-maxval), 'and', str(maxval),
                        'because dependent value cannot be found'
                    )
            else:
                symbol_map[k] = random.randint(retlow, rethigh)
                free_symbols_map[k] = symbol_map[k]

        for k in sdfg.symbols:
            if k not in symbol_map:
                symbol_map[k] = 0
        return symbol_map, free_symbols_map


    def sample_inputs(
        self, sdfg: SDFG, input_configuration: Set[str],
        symbols_map: Dict[str, int], decay_by: int = 0,
        constraints_map: Dict[str, Tuple[int, int]] = None
    ) -> Dict[str, Union[float, int, np.ndarray]]:
        inputs = dict()
        for name in input_configuration:
            array = sdfg.arrays[name]
            constraints = None
            if constraints_map is not None and name in constraints_map:
                lconstr, hconstr = constraints_map[name]
                constraints = (
                    lconstr.subs(symbols_map), hconstr.subs(symbols_map)
                )
            container = self._sample_container(
                array, symbols_map, decay_by, constraints=constraints
            )
            inputs[name] = container
        return inputs


    def generate_output_containers(
        self, sdfg: SDFG, system_state: Set[str], input_configuration: Set[str],
        symbols_map: Dict[str, int]
    ) -> Dict[str, Union[float, int, np.ndarray]]:
        outputs = dict()
        for name in system_state:
            if not name in input_configuration:
                array = sdfg.arrays[name]
                container = self._empty_container(array, symbols_map)
                outputs[name] = container
        return outputs
