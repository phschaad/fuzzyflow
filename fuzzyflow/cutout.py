# Copyright 2022 ETH Zurich and the FuzzyFlow authors. All rights reserved.
# This file is part of FuzzyFlow, which is released under the BSD 3-Clause
# License. For details, see the LICENSE file.

from collections import deque
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Union
import sympy as sp
import networkx as nx

from dace.data import Data
from dace.data import dtypes
from dace.sdfg import SDFG, SDFGState, InterstateEdge
from dace.sdfg import nodes as nd
from dace.sdfg.analysis import cutout as dcut
from dace.transformation.passes.analysis import StateReachability, AccessSets
from dace.transformation.transformation import (PatternTransformation,
                                                SingleStateTransformation,
                                                MultiStateTransformation,
                                                SubgraphTransformation)

from fuzzyflow import util


TranslationDict = Dict[Union[nd.Node, SDFGState], Union[nd.Node, SDFGState]]
AccessSet = Dict[int, Dict[SDFGState, Tuple[Set[str], Set[str]]]]


class CutoutStrategy(Enum):
    SIMPLE = 'SIMPLE'
    MINIMUM_DOMINATOR_FLOW = 'MINIMUM_DOMINATOR_FLOW'

    def __str__(self):
        return self.value


def _find_program_inputs(
    cutout: SDFG, inverted_closure: Dict[SDFGState, Set[SDFGState]],
    state_access_set: AccessSet, translation_dict: TranslationDict,
    inverse_translation_dict: TranslationDict
) -> Set[str]:
    result: Set[str] = set()
    access_set = state_access_set[cutout.sdfg_id]
    for state in cutout.states():
        for dname in access_set[state][0]:
            if dname in result:
                continue

            data = cutout.arrays[dname]
            if data.transient:
                # Transients belong to the input if they have writes to them in
                # at least one of the states preceeding the cutout. We detect
                # states preceeding the cutout by checking the inverse state
                # reachability from the cutout start state.
                start_state = inverse_translation_dict[cutout.start_state]
                predecessor_states = inverted_closure[start_state]
                for pred_state in predecessor_states:
                    if (pred_state not in translation_dict or
                        translation_dict[pred_state] not in cutout.states()):
                        for n in pred_state.nodes():
                            if (isinstance(n, nd.AccessNode) and
                                n.data == dname and
                                pred_state.in_degree(n) > 0):
                                result.add(dname)
            else:
                # Anything non-transient belongs to the input in all cases.
                result.add(dname)

    # Add any source access nodes in singlestate cutouts.
    if len(cutout.states()) == 1:
        inodes: List[nd.AccessNode] = cutout.input_arrays()
        for inode in inodes:
            if inode.data not in result:
                result.add(inode.data)

    return result

def _find_program_outputs(
    cutout: SDFG, access_sets: AccessSet, graph_access_set: AccessSet,
    state_reachability_sets: Dict[int, Dict[SDFGState, Set[SDFGState]]],
    translation_dict: TranslationDict, inverse_translation_dict: TranslationDict
) -> Set[str]:
    result: Set[str] = set()
    access_set = access_sets[cutout.sdfg_id]
    state_reachability_set = state_reachability_sets[cutout.sdfg_id]

    reachable_states = set()
    for state in cutout.states():
        if state not in state_reachability_set:
            continue
        reach = state_reachability_set[inverse_translation_dict[state]]
        for s in reach:
            if (s not in translation_dict or
                translation_dict[s] not in cutout.states()):
                reachable_states.add(s)

    for state in cutout.states():
        for dname in access_set[state][1]:
            if dname in result:
                continue

            data = cutout.arrays[dname]
            if data.transient:
                # Transients belong to the output if they have reads _after_
                # the cutout. States after the cutout are taken from the
                # state reachability (i.e., the transitive closure) of all
                # cutout states.
                for s in reachable_states:
                    if dname in graph_access_set[s][1]:
                        result.add(dname)
                        break
            else:
                # Anything non-transient belongs to the output in all cases.
                result.add(dname)

    # Add any sink access nodes in singlestate cutouts.
    if len(cutout.states()) == 1:
        onodes: List[nd.AccessNode] = cutout.output_arrays()
        for onode in onodes:
            if onode.data not in result:
                result.add(onode.data)

    return result


def find_program_parameters(
    cutout: SDFG, original_sdfg: SDFG, translation_dict: TranslationDict
):
    state_reachability_sdfg = StateReachability().apply_pass(original_sdfg, {})

    inverse_translation_dict: TranslationDict = dict()
    for k, v in translation_dict.items():
        inverse_translation_dict[v] = k

    inverted_cutout_reachability = {}
    inverted_sdfg = original_sdfg.nx.reverse()
    tc: nx.DiGraph = nx.transitive_closure(inverted_sdfg)
    for state in inverted_sdfg.nodes():
        inverted_cutout_reachability[state] = set(tc.successors(state))
    access_sets_cutout = AccessSets().apply_pass(cutout, {})
    access_sets_sdfg = AccessSets().apply_pass(original_sdfg, {})

    inputs = _find_program_inputs(
        cutout, inverted_cutout_reachability, access_sets_cutout,
        translation_dict, inverse_translation_dict
    )

    outputs = _find_program_outputs(
        cutout, access_sets_cutout, access_sets_sdfg[original_sdfg.sdfg_id],
        state_reachability_sdfg, translation_dict,
        inverse_translation_dict
    )

    return inputs, outputs


def _input_volume_for_nodes(
    state: SDFGState, nodes: Set[nd.Node]
) -> Tuple[sp.Expr, List[nd.AccessNode]]:
    cutout = dcut.cutout_state(state, *nodes)
    inputs: List[nd.AccessNode] = cutout.input_arrays()
    indata = [cutout.arrays[n.data] for n in inputs]
    total_volume = 0
    for dat in indata:
        total_volume += dat.total_size
    return total_volume, inputs


def _minimum_dominator_flow_cutout(
    sdfg: SDFG, xform: Union[SubgraphTransformation, PatternTransformation]
) -> Tuple[SDFG, TranslationDict]:
    affected_nodes = util.transformation_get_affected_nodes(sdfg, xform)
    if (isinstance(xform, SubgraphTransformation) or
        isinstance(xform, SingleStateTransformation)):
        state = sdfg.node(xform.state_id)
        base_volume, inputs = _input_volume_for_nodes(state, affected_nodes)
    elif isinstance(xform, MultiStateTransformation):
        raise NotImplementedError('Multistate cutouts not yet supported')
    raise Exception('This type of transformation cannot be supported')


def _minimal_transformation_cutout(
    sdfg: SDFG, xform: Union[SubgraphTransformation, PatternTransformation]
) -> Tuple[SDFG, TranslationDict]:
    affected_nodes = util.transformation_get_affected_nodes(sdfg, xform)
    if (isinstance(xform, SubgraphTransformation) or
        isinstance(xform, SingleStateTransformation)):
        state = sdfg.node(xform.state_id)
        translation_dict: Dict[nd.Node, nd.Node] = dict()
        ct: SDFG = dcut.cutout_state(
            state, *affected_nodes, make_copy=True,
            inserted_nodes=translation_dict
        )
        util.translate_transformation(xform, sdfg, ct, translation_dict)

        return ct, translation_dict
    elif isinstance(xform, MultiStateTransformation):
        translation_dict: Dict[SDFGState, SDFGState] = dict()
        ct: SDFG = dcut.multistate_cutout(
            sdfg, *affected_nodes, inserted_states=translation_dict
        )
        util.translate_transformation(xform, sdfg, ct, translation_dict)

        return ct, translation_dict
    raise Exception('This type of transformation cannot be supported')


def find_cutout_for_transformation(
    sdfg: SDFG, xform: Union[SubgraphTransformation, PatternTransformation],
    strategy: CutoutStrategy
) -> Tuple[SDFG, TranslationDict]:
    if strategy == CutoutStrategy.SIMPLE:
        return _minimal_transformation_cutout(sdfg, xform)
    elif strategy == CutoutStrategy.MINIMUM_DOMINATOR_FLOW:
        return _minimum_dominator_flow_cutout(sdfg, xform)
    raise Exception('Unknown cutout strategy')
