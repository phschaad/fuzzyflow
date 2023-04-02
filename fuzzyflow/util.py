# Copyright 2022 ETH Zurich and the FuzzyFlow authors. All rights reserved.
# This file is part of FuzzyFlow, which is released under the BSD 3-Clause
# License. For details, see the LICENSE file.

from collections import deque
import json
from enum import Enum
from functools import total_ordering
from typing import Dict, List, Set, Tuple, Union
import sympy as sp
from math import floor

from dace import serialize
from dace.data import ArrayLike, dtypes
from dace.sdfg import SDFG, ScopeSubgraphView, SDFGState
from dace.sdfg import nodes as nd
from dace.sdfg.state import StateSubgraphView
from dace.memlet import Memlet
from dace.subsets import Range
from dace.symbolic import SymExpr
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
    ct: SDFG, sdfg: SDFG, pre_constraints: Dict = None, max_dim: int = 1024
) -> Dict:
    general_constraints = dict()
    if pre_constraints is not None:
        general_constraints = pre_constraints

    # Construct symbol constraints from loops the cutout may be a part of.
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

    # Construct symbol constraints from all data accesses in the cutout.
    cutout_memlets: Set[Memlet] = set()
    for state in ct.states():
        for iedge in ct.in_edges(state):
            ise_memlets = iedge.data.get_read_memlets(ct.arrays)
            for memlet in ise_memlets:
                cutout_memlets.add(memlet)
        for oedge in ct.out_edges(state):
            ise_memlets = oedge.data.get_read_memlets(ct.arrays)
            for memlet in ise_memlets:
                cutout_memlets.add(memlet)
        for edge in state.edges():
            cutout_memlets.add(edge.data)

    memlet_constraints = dict()
    for memlet in cutout_memlets:
        if isinstance(memlet.subset, Range):
            desc = ct.arrays[memlet.data]
            for i, r in enumerate(memlet.subset.ranges):
                lower_free_symbols = set()
                upper_free_symbols = set()
                step_free_symbols = set()
                shape_free_symbols = set()
                if isinstance(r[0], sp.Basic):
                    lower_free_symbols = r[0].free_symbols
                elif isinstance(r[0], SymExpr):
                    lower_free_symbols = r[0].expr.free_symbols

                if isinstance(r[1], sp.Basic):
                    upper_free_symbols = r[1].free_symbols
                elif isinstance(r[1], SymExpr):
                    upper_free_symbols = r[1].expr.free_symbols

                if isinstance(r[2], sp.Basic):
                    step_free_symbols = r[2].free_symbols
                elif isinstance(r[2], SymExpr):
                    step_free_symbols = r[2].expr.free_symbols

                if isinstance(desc.shape[i], sp.Basic):
                    shape_free_symbols = desc.shape[i].free_symbols
                elif isinstance(desc.shape[i], SymExpr):
                    shape_free_symbols = desc.shape[i].expr.free_symbols

                if len(lower_free_symbols) > 0:
                    lhs_free = lower_free_symbols
                    rhs_free = upper_free_symbols
                    if not any([s in lhs_free for s in rhs_free]):
                        memlet_constraints[r[0]] = (0, r[1] - 1)

                if len(upper_free_symbols) > 0:
                    lhs_free = upper_free_symbols
                    rhs_free = shape_free_symbols
                    if not any([s in lhs_free for s in rhs_free]):
                        memlet_constraints[r[1]] = (0, desc.shape[i] - 1)

                if len(step_free_symbols) > 0:
                    lhs_free = step_free_symbols
                    rhs_free = upper_free_symbols
                    if not any([s in lhs_free for s in rhs_free]):
                        memlet_constraints[r[2]] = (1, r[1] - 1)

    for k, v in memlet_constraints.items():
        if isinstance(k, sp.Basic):
            for sym in k.free_symbols:
                s = str(sym)
                if s in ct.free_symbols and not s in cutout_constraints:
                    cutout_constraints[s] = (v[0] + 1, v[1], 1)
        else:
            if k in ct.free_symbols and not k in cutout_constraints:
                cutout_constraints[k] = (v[0] + 1, v[1], 1)

    # Constrain anything used as a data size to be at least 1. Additionally,
    # make sure that the size of no data container can be greater than 1GB -
    # assuming that each element is 8 bytes, conservatively.
    for k, v in ct.arrays.items():
        n_dims = len(v.shape)
        max_dim_size = min(floor((1e9 / 8) ** (1 / n_dims)), max_dim)
        for s in v.shape:
            if isinstance(s, sp.Basic):
                for sym in s.free_symbols:
                    s = str(sym)
                    if s in ct.free_symbols and not s in cutout_constraints:
                        cutout_constraints[s] = (1, max_dim_size, 1)
            else:
                if s in ct.free_symbols and not s in cutout_constraints:
                    cutout_constraints[s] = (1, max_dim_size, 1)

    return cutout_constraints
