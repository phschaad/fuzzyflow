# Copyright 2022 ETH Zurich and the FuzzyFlow authors. All rights reserved.
# This file is part of FuzzyFlow, which is released under the BSD 3-Clause
# License. For details, see the LICENSE file.

import json
from enum import Enum
from functools import total_ordering
from typing import Dict, List, Set, Tuple, Union

from dace import serialize
from dace.sdfg import SDFG, ScopeSubgraphView, SDFGState
from dace.sdfg import nodes as nd
from dace.sdfg.state import StateSubgraphView
from dace.transformation.interstate.loop_detection import (DetectLoop,
                                                           find_for_loop)
from dace.transformation.passes.pattern_matching import match_patterns
from dace.transformation.transformation import (MultiStateTransformation,
                                                PatternTransformation,
                                                SingleStateTransformation,
                                                SubgraphTransformation)


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


class LoopDetection(DetectLoop, MultiStateTransformation):

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        return super().can_be_applied(graph, expr_index, sdfg, permissive)


    def apply(self, _, sdfg):
        return super().apply(_, sdfg)


def transformation_get_affected_nodes(
    p_sdfg: SDFG, xform: Union[PatternTransformation, SubgraphTransformation],
    strict: bool = False
) -> Set[Union[nd.Node, SDFGState]]:
    if xform.sdfg_id >= 0 and p_sdfg.sdfg_list:
        sdfg = p_sdfg.sdfg_list[xform.sdfg_id]
    else:
        sdfg = p_sdfg

    affected_nodes: Set[Union[nd.Node, SDFGState]] = set()
    if isinstance(xform, PatternTransformation):
        if isinstance(xform, DetectLoop):
            to_visit = [xform.loop_begin]
            while to_visit:
                state = to_visit.pop(0)
                for _, dst, _ in state.parent.out_edges(state):
                    if dst not in affected_nodes and dst is not xform.loop_guard:
                        to_visit.append(dst)
                affected_nodes.add(state)

            affected_nodes.add(xform.loop_guard)
            affected_nodes.add(xform.exit_state)
            for iedge in xform.loop_begin.parent.in_edges(xform.loop_guard):
                if iedge.src not in affected_nodes:
                    affected_nodes.add(iedge.src)
        else:
            for k, _ in xform._get_pattern_nodes().items():
                try:
                    affected_nodes.add(getattr(xform, k))
                except KeyError:
                    # Ignored.
                    pass
    elif isinstance(xform, SubgraphTransformation):
        sgv = xform.get_subgraph(sdfg)
        if isinstance(sgv, StateSubgraphView):
            for n in sgv.nodes():
                affected_nodes.add(n)
    else:
        raise Exception('Unknown transformation type')

    if strict:
        return affected_nodes

    expanded: Set[Union[nd.Node, SDFGState]] = set()
    if xform.state_id >= 0:
        state = sdfg.node(xform.state_id)
        for node in affected_nodes:
            expanded.add(node)
            if isinstance(node, nd.EntryNode):
                scope: ScopeSubgraphView = state.scope_subgraph(
                    node, include_entry=True, include_exit=True
                )
                for n in scope.nodes():
                    expanded.add(n)
            elif isinstance(node, nd.ExitNode):
                entry = state.entry_node(node)
                scope: ScopeSubgraphView = state.scope_subgraph(
                    entry, include_entry=True, include_exit=True
                )
                for n in scope.nodes():
                    expanded.add(n)
    else:
        # Multistate, all affected nodes are states, return as is.
        # TODO: Do we want to cut out entire loops and/or branch constructsl
        # here?
        return affected_nodes

    return expanded


def apply_transformation(
    sdfg: SDFG,
    xform: Union[SubgraphTransformation, PatternTransformation]
) -> None:
    if isinstance(xform, SubgraphTransformation):
        sdfg.append_transformation(xform)
        xform.apply(sdfg)
    else:
        xform.apply_pattern(sdfg)


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
    return xform, target_sdfg


def translate_transformation(
    xform: Union[PatternTransformation, SubgraphTransformation],
    old_sdfg: SDFG, target_sdfg: SDFG,
    translation_dict: Dict[nd.Node, nd.Node]
) -> None:
    if isinstance(xform, SingleStateTransformation):
        old_state = old_sdfg.node(xform.state_id)
        xform.state_id = target_sdfg.node_id(target_sdfg.start_state)
        xform._sdfg = target_sdfg
        xform.sdfg_id = 0
        for k in xform.subgraph.keys():
            old_node = old_state.node(xform.subgraph[k])
            if old_node in translation_dict:
                new_node = translation_dict[old_node]
                xform.subgraph[k] = target_sdfg.start_state.node_id(new_node)
    elif isinstance(xform, MultiStateTransformation):
        xform._sdfg = target_sdfg
        xform.sdfg_id = 0
        for k in xform.subgraph.keys():
            old_node = old_sdfg.node(xform.subgraph[k])
            if old_node in translation_dict:
                new_node = translation_dict[old_node]
                xform.subgraph[k] = target_sdfg.node_id(new_node)
    else:
        old_state = old_sdfg.node(xform.state_id)
        xform.state_id = target_sdfg.node_id(target_sdfg.start_state)
        new_subgraph: Set[int] = set()
        for k in xform.subgraph:
            old_node = old_state.node(k)
            if old_node in translation_dict:
                new_node = translation_dict[old_node]
                new_subgraph.add(target_sdfg.start_state.node_id(new_node))
        xform.subgraph = new_subgraph


def cutout_determine_symbol_constraints(
    ct: SDFG, sdfg: SDFG, pre_constraints: Dict = None
) -> Dict:
    general_constraints = dict()
    if pre_constraints is not None:
        general_constraints = pre_constraints

    loop_matches: List[LoopDetection] = list(
        match_patterns(sdfg, LoopDetection)
    )
    for ld in loop_matches:
        if (ld.loop_guard is not None and ld.loop_begin is not None and
            ld.exit_state is not None):
            res = find_for_loop(ld._sdfg, ld.loop_guard, ld.loop_begin)
            if res is not None and res[0] not in general_constraints:
                itvar, rng, _ = res
                general_constraints[itvar] = rng

    cutout_constraints = dict()
    for s in ct.free_symbols:
        if s in general_constraints:
            cutout_constraints[s] = general_constraints[s]

    return cutout_constraints
