# Copyright 2022 ETH Zurich and the FuzzyFlow authors. All rights reserved.
# This file is part of FuzzyFlow, which is released under the BSD 3-Clause
# License. For details, see the LICENSE file.

from enum import Enum
from typing import Optional, Set, Union
import sympy as sp
import networkx as nx

from dace.sdfg import SDFG, SDFGState
from dace.sdfg import nodes as nd
from dace.sdfg.analysis.cutout import (SDFGCutout,
                                       _transformation_determine_affected_nodes,
                                       _extend_subgraph_with_access_nodes)
from dace.sdfg.state import StateSubgraphView
from dace.transformation.transformation import (PatternTransformation,
                                                SingleStateTransformation,
                                                MultiStateTransformation,
                                                SubgraphTransformation)

from fuzzyflow import util


class CutoutStrategy(Enum):
    SIMPLE = 'SIMPLE'
    MINIMUM_DOMINATOR_FLOW = 'MINIMUM_DOMINATOR_FLOW'

    def __str__(self):
        return self.value


def _reduce_in_configuration(
    state: SDFGState, sdfg: SDFG,
    affected_nodes: Set[nd.Node]
) -> Set[Union[nd.Node, SDFGState]]:
    subgraph: StateSubgraphView = StateSubgraphView(state, affected_nodes)
    subgraph = _extend_subgraph_with_access_nodes(state, subgraph)
    subgraph_nodes = set(subgraph.nodes())

    symbols_map = dict()
    for s in sdfg.free_symbols:
        symbols_map[s] = 20 # TODO: we should probably fix this differently.

    source_candidates = set()
    for n in subgraph_nodes:
        source_candidates.add(state.entry_node(n))

    source = None
    if len(source_candidates) == 1:
        source = list(source_candidates)[0]

    proxy = nx.DiGraph()
    scope = set(state.nodes())
    if source == None:
        source = 'PROXY_SOURCE'
    else:
        child_dict = state.scope_children()
        scope = child_dict[source]
    proxy.add_node(source)
    sink = 'PROXY_SINK'
    proxy.add_node(sink)

    #state_input_config = _determine_state_input_config(sdfg, state)

    proxy_in_volume = 0
    for n in scope:
        if n in subgraph_nodes or n == source:
            for iedge in state.in_edges(n):
                if (iedge.src not in subgraph_nodes
                        and isinstance(n, nd.AccessNode)):
                    vol = sdfg.arrays[n.data].total_size
                    if isinstance(vol, sp.Expr):
                        vol = vol.subs(symbols_map)
                    proxy_in_volume += vol
                    if proxy.has_edge(iedge.src, sink):
                        proxy[iedge.src][sink]['capacity'] += vol
                        proxy[iedge.src][sink]['orig'].append(iedge)
                    else:
                        proxy.add_edge(iedge.src,
                                       sink,
                                       capacity=vol,
                                       orig=[iedge])
        else:
            proxy.add_node(n)

            for iedge in state.in_edges(n):
                if iedge.src not in subgraph_nodes:
                    vol = iedge.data.volume
                    if isinstance(vol, sp.Expr):
                        vol = vol.subs(symbols_map)
                    if proxy.has_edge(iedge.src, n):
                        proxy[iedge.src][n]['capacity'] += vol
                        proxy[iedge.src][n]['orig'].append(iedge)
                    else:
                        proxy.add_edge(iedge.src, n, capacity=vol, orig=[iedge])
                else:
                    vol = iedge.data.volume
                    if isinstance(vol, sp.Expr):
                        vol = vol.subs(symbols_map)
                    if proxy.has_edge(sink, n):
                        proxy[sink][n]['capacity'] += vol
                        proxy[sink][n]['orig'].append(iedge)
                    else:
                        proxy.add_edge(sink, n, capacity=vol, orig=[iedge])
    if proxy_in_volume > 0:
        proxy.add_edge(source, sink, capacity=proxy_in_volume)

    cut_val, (_, non_reachable) = nx.minimum_cut(proxy,
                                                 source,
                                                 sink,
                                                 flow_func=nx.flow.edmonds_karp)

    if cut_val < proxy_in_volume * 2:
        reachability_dict = dict(nx.all_pairs_bellman_ford_path_length(proxy))
        expanded_affected = set(affected_nodes)
        for n in non_reachable:
            if n not in affected_nodes and n != sink:
                post_sink_reachable = reachability_dict[sink]
                if n not in post_sink_reachable:
                    expanded_affected.add(n)
        return expanded_affected
    else:
        return affected_nodes


def _minimum_dominator_flow_cutout(
    p_sdfg: SDFG, xform: Union[SubgraphTransformation, PatternTransformation]
) -> SDFGCutout:
    affected_nodes = _transformation_determine_affected_nodes(p_sdfg, xform)

    if (isinstance(xform, SubgraphTransformation)
            or isinstance(xform, SingleStateTransformation)):
        if xform.sdfg_id >= 0 and p_sdfg.sdfg_list:
            sdfg = p_sdfg.sdfg_list[xform.sdfg_id]
        else:
            sdfg = p_sdfg

        state = sdfg.node(xform.state_id)

        reduced_affected_nodes = _reduce_in_configuration(
            state, sdfg, affected_nodes)

        translation_dict = dict()
        ct = cutout(*reduced_affected_nodes,
                    translation=translation_dict,
                    state=state)
        util.translate_transformation(xform, sdfg, ct, translation_dict)
        return ct, translation_dict
    else:
        # For multistate cutouts we do not attempt to reduce the input
        # configuration, since the configuration is dependent on having a single
        # input state to the cutout.
        return _minimal_transformation_cutout(p_sdfg, xform, affected_nodes)


def _minimal_transformation_cutout(
    p_sdfg: SDFG,
    xform: Union[SubgraphTransformation, PatternTransformation],
    affected_nodes: Optional[Set[Union[nd.Node, SDFGState]]] = None
) -> SDFGCutout:
    if affected_nodes is None:
        affected_nodes = _transformation_determine_affected_nodes(p_sdfg, xform)

    if xform.sdfg_id >= 0 and p_sdfg.sdfg_list:
        sdfg = p_sdfg.sdfg_list[xform.sdfg_id]
    else:
        sdfg = p_sdfg

    if (isinstance(xform, SubgraphTransformation)
            or isinstance(xform, SingleStateTransformation)):
        state = sdfg.node(xform.state_id)
        translation_dict = dict()
        ct = cutout(*affected_nodes, translation=translation_dict, state=state)
        util.translate_transformation(xform, sdfg, ct, translation_dict)
        return ct, translation_dict
    elif isinstance(xform, MultiStateTransformation):
        translation_dict = dict()
        ct: SDFG = cutout(*affected_nodes,
                          translation=translation_dict,
                          state=None)
        o_sdfg = list(affected_nodes)[0].parent
        util.translate_transformation(xform, o_sdfg, ct, translation_dict)
        return ct, translation_dict
    raise Exception('This type of transformation cannot be supported')


def find_cutout_for_transformation(
    sdfg: SDFG, xform: Union[SubgraphTransformation, PatternTransformation],
    strategy: CutoutStrategy
) -> SDFGCutout:
    if strategy == CutoutStrategy.SIMPLE:
        return _minimal_transformation_cutout(sdfg, xform)
    elif strategy == CutoutStrategy.MINIMUM_DOMINATOR_FLOW:
        return _minimum_dominator_flow_cutout(sdfg, xform)
    raise Exception('Unknown cutout strategy')


def cutout(*nodes: Union[nd.Node, SDFGState],
           state: Optional[SDFGState] = None) -> SDFG:
    if state is not None:
        if any([isinstance(n, SDFGState) for n in nodes]):
            raise Exception(
                'Mixing cutout nodes of type Node and SDFGState is not allowed')
        new_sdfg = SDFGCutout.singlestate_cutout(state, *nodes, make_copy=True)
    else:
        if any([isinstance(n, nd.Node) for n in nodes]):
            raise Exception(
                'Mixing cutout nodes of type Node and SDFGState is not allowed')
        new_sdfg = SDFGCutout.multistate_cutout(*nodes)

    # Ensure the parent relationships and SDFG list is correct.
    for s in new_sdfg.states():
        for node in s.nodes():
            if isinstance(node, nd.NestedSDFG):
                node.sdfg._parent_sdfg = new_sdfg
    new_sdfg.reset_sdfg_list()

    return new_sdfg
