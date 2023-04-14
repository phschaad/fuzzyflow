# Copyright 2022 ETH Zurich and the FuzzyFlow authors. All rights reserved.
# This file is part of FuzzyFlow, which is released under the BSD 3-Clause
# License. For details, see the LICENSE file.

from collections import deque
import json
from enum import Enum
from functools import total_ordering
from typing import Tuple, Union

from dace import serialize
from dace.data import ArrayLike, dtypes
from dace.sdfg import SDFG
from dace.transformation.transformation import (PatternTransformation,
                                                SubgraphTransformation)


class FailureReason(Enum):
    EXCEPTION = 'EXCEPTION'
    FAILED_VALIDATE = 'FAILED_VALIDATE'
    COMPILATION_FAILURE = 'COMPILATION_FAILURE'
    EXIT_CODE_MISMATCH = 'EXIT_CODE_MISMATCH'
    SYSTEM_STATE_MISMATCH = 'SYSTEM_STATE_MISMATCH'
    FAILED_TO_APPLY = 'FAILED_TO_APPLY'


@total_ordering
class StatusLevel(Enum):
    OFF = 0
    BAR_ONLY = 1
    DEBUG = 2
    VERBOSE = 3

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        raise ValueError('Cannot compare StatusLevel to', str(other.__class__))


def data_report_get_latest_version(report, item) -> ArrayLike:
        if report is None:
            return None
        filenames = report.files[item]
        desc = report.sdfg.arrays[item]
        dtype: dtypes.typeclass = desc.dtype
        npdtype = dtype.as_numpy_dtype()

        file = deque(iter(filenames), maxlen=1).pop()
        nparr, view = report._read_array_file(file, npdtype)
        report.loaded_values[item, -1] = nparr
        return view


def apply_transformation(
    sdfg: SDFG,
    xform: Union[SubgraphTransformation, PatternTransformation]
) -> None:
    if isinstance(xform, SubgraphTransformation):
        sdfg.append_transformation(xform)
        xform.apply(sdfg)
    else:
        xform.apply_pattern(append=False)


def load_transformation_from_file(
    xform_path: str, sdfg: SDFG
) -> Tuple[Union[SubgraphTransformation, PatternTransformation], SDFG]:
    xform = None
    target_sdfg = None
    with open(xform_path, 'r') as xform_file:
        xform_json = json.load(xform_file)
        xform = serialize.from_json(xform_json)
        if isinstance(xform, (SubgraphTransformation, PatternTransformation)):
            target_sdfg = sdfg.sdfg_list[xform.sdfg_id]
            xform._sdfg = target_sdfg
        else:
            raise Exception(
                'Transformations of type', type(xform), 'cannot be handled'
            )
    if hasattr(xform, 'simplify'):
        xform.simplify = False
    return xform, target_sdfg


'''
def make_shared_desc_array(
        name: str, descriptor: Array,
        original_array: Optional[ArrayLike] = None,
        symbols: Optional[Dict[str, Any]] = None
) -> ArrayLike:
    symbols = symbols or {}

    free_syms = set(map(str, descriptor.free_symbols)) - symbols.keys()
    if free_syms:
        raise NotImplementedError(
            'Cannot make Python references to arrays ' +
            'with undefined symbolic sizes:',
            free_syms
        )

    def create_array(shape: Tuple[int], dtype: np.dtype, total_size: int, strides: Tuple[int]) -> ArrayLike:
        buffer = np.ndarray([total_size], dtype=dtype)
        view = np.ndarray(shape, dtype, buffer=buffer, strides=[s * dtype.itemsize for s in strides])
        return view

    shared_arr = shared_memory.SharedMemory(name, )

    def copy_array(dst, src):
        dst[:] = src

    # Make numpy array from data descriptor
    npdtype = descriptor.dtype.as_numpy_dtype()
    evaluated_shape = tuple(symbolic.evaluate(s, symbols) for s in descriptor.shape)
    evaluated_size = symbolic.evaluate(descriptor.total_size, symbols)
    evaluated_strides = tuple(symbolic.evaluate(s, symbols) for s in descriptor.strides)
    view = create_array(evaluated_shape, npdtype, evaluated_size, evaluated_strides)
    if original_array is not None:
        copy_array(view, original_array)

    return view
    '''
