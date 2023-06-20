# Copyright 2022 ETH Zurich and the FuzzyFlow authors. All rights reserved.
# This file is part of FuzzyFlow, which is released under the BSD 3-Clause
# License. For details, see the LICENSE file.

from queue import Empty
import traceback
import multiprocessing as mp
import numpy as np
from typing import Any, Dict, List, Union

from dace.codegen.compiled_sdfg import CompiledSDFG, ReloadableDLL
from dace.sdfg import SDFG
from dace.data import make_array_from_descriptor, Array, Scalar
from dace.libraries.standard.memory import aligned_ndarray

from fuzzyflow.util import StatusLevel


def run_subprocess_precompiled(
    program: CompiledSDFG, containers: Dict[str, Union[np.ndarray, np.number]],
    symbols: Dict[str, Any], status: StatusLevel = StatusLevel.OFF
) -> int:
    if status >= StatusLevel.VERBOSE:
        print('Serializing containers')
    containers_serialized = dict()
    for k, v in containers.items():
        if isinstance(v, np.ndarray):
            containers_serialized[k] = v.tolist()
        else:
            containers_serialized[k] = v

    sdfg: SDFG = program.sdfg
    process = RunnerProcess(
        target=_subprocess_runner_precompiled,
        args=(
            sdfg.to_json(),
            program.argnames,
            program._lib._library_filename,
            sdfg.name,
            containers_serialized,
            symbols,
            status
        )
    )

    if status >= StatusLevel.VERBOSE:
        print('Launching runner')
    process.start()
    if status >= StatusLevel.VERBOSE:
        print('Waiting for runner')
    process.join()
    if status >= StatusLevel.VERBOSE:
        print('Runner joined')

    if process.exitcode != 0:
        #print('Error occured in execution')
        return process.exitcode
    if process.exception:
        error, trace = process.exception
        print(trace)
        print('Error occured in execution:', error)
        return -1
    return 0


def _subprocess_runner_precompiled(
    sdfg_json: object, argnames: List[str], filename: str, name: str,
    containers_serialized: Dict[str, list],
    symbols: Dict[str, int], status: StatusLevel = StatusLevel.OFF
) -> None:
    if status >= StatusLevel.VERBOSE:
        print('Runner started, deserializing SDFG', flush=True)
    sdfg = SDFG.from_json(sdfg_json)
    if status >= StatusLevel.VERBOSE:
        print('Loading library', flush=True)
    lib = ReloadableDLL(filename, name)
    if status >= StatusLevel.VERBOSE:
        print('Loading compiled SDFG', flush=True)
    program = CompiledSDFG(sdfg, lib, argnames)

    containers: Dict[str, Union[np.number, np.ndarray]] = dict()

    reconstructed_symbols = {}
    for k, v in symbols.items():
        reconstructed_symbols[k] = v
    for k, v in sdfg.constants.items():
        reconstructed_symbols[k] = v

    if status >= StatusLevel.VERBOSE:
        print('Deserializing continers', flush=True)
    for k, v in containers_serialized.items():
        array = sdfg.arrays[k]
        if isinstance(array, Scalar):
            containers[k] = v
        else:
            view = make_array_from_descriptor(array, v, reconstructed_symbols)
            if isinstance(array, Array) and array.alignment:
                containers[k] = aligned_ndarray(view, array.alignment)
            else:
                containers[k] = view

    if status >= StatusLevel.VERBOSE:
        print('Executing compiled SDFG', flush=True)
    program.__call__(**containers, **symbols)
    if status >= StatusLevel.VERBOSE:
        print('Execution completed, runner exiting', flush=True)
    exit(0)


def run_subprocess(
    program: SDFG, containers: Dict[str, Union[np.ndarray, np.number]],
    symbols: Dict[str, Any]
) -> int:
    queue = mp.Queue()

    containers_serialized = dict()
    for k, v in containers.items():
        if isinstance(v, np.ndarray):
            containers_serialized[k] = v.tolist()
        else:
            containers_serialized[k] = v
    process = RunnerProcess(
        target=_subprocess_runner,
        args=(program.to_json(), containers_serialized, symbols, queue)
    )

    process.start()
    ret = None
    while process.is_alive():
        try:
            ret = queue.get(block=True, timeout=1)
        except Empty:
            pass

    if process.exitcode != 0:
        #print('Error occured in execution')
        return process.exitcode
    if process.exception:
        error, trace = process.exception
        print(trace)
        print('Error occured in execution:', error)
        return -1

    if ret is not None:
        for k, v in ret.items():
            array = program.arrays[k]
            containers[k] = make_array_from_descriptor(array, v, symbols)
    else:
        return -2
    return 0


def _subprocess_runner(
    program_json: object, containers_serialized: Dict[str, list],
    symbols: Dict[str, int], queue: mp.Queue
) -> None:
    program = SDFG.from_json(program_json)

    containers: Dict[str, Union[np.number, np.ndarray]] = dict()

    for k, v in containers_serialized.items():
        array = program.arrays[k]
        containers[k] = make_array_from_descriptor(array, v, symbols)

    compiled_program = program.compile()
    compiled_program.__call__(**containers, **symbols)

    ret_containers_serialized = dict()
    for k, v in containers.items():
        if isinstance(v, np.ndarray):
            ret_containers_serialized[k] = v.tolist()
        else:
            ret_containers_serialized[k] = v
    queue.put(ret_containers_serialized, block=False)


class RunnerProcess(mp.Process):

    def __init__(self, *args, **kwargs):
        mp.Process.__init__(self, *args, **kwargs)
        self._pconn, self._cconn = mp.Pipe()
        self._exception = None


    def run(self):
        try:
            mp.Process.run(self)
            self._cconn.send(None)
        except Exception as e:
            tb = traceback.format_exc()
            self._cconn.send((e, tb))
            raise e


    @property
    def exception(self):
        if self._pconn.poll():
            self._exception = self._pconn.recv()
        return self._exception
