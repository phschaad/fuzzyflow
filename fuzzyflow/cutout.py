# Copyright 2022 ETH Zurich and the FuzzyFlow authors. All rights reserved.
# This file is part of FuzzyFlow, which is released under the BSD 3-Clause
# License. For details, see the LICENSE file.

from collections import deque
import copy
from enum import Enum
from typing import Deque, Dict, List, Optional, Set, Tuple, Union
import sympy as sp
import networkx as nx

from dace import data, Memlet
from dace.sdfg import SDFG, SDFGState
from dace.sdfg import nodes as nd, utils as sdutil
from dace.sdfg.graph import MultiConnectorEdge
from dace.sdfg.state import StateSubgraphView, SubgraphView
from dace.transformation.transformation import (PatternTransformation,
                                                SingleStateTransformation,
                                                MultiStateTransformation,
                                                SubgraphTransformation)
from dace.transformation.passes.analysis import StateReachability

from fuzzyflow import util


TranslationDict = Dict[Union[nd.Node, SDFGState], Union[nd.Node, SDFGState]]


class CutoutStrategy(Enum):
    SIMPLE = 'SIMPLE'
    MINIMUM_DOMINATOR_FLOW = 'MINIMUM_DOMINATOR_FLOW'

    def __str__(self):
        return self.value


def _reduce_in_configuration(
    state: SDFGState, sdfg: SDFG, affected_nodes: Set[nd.Node]
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
                if (iedge.src not in subgraph_nodes and
                    isinstance(n, nd.AccessNode)):
                    vol = sdfg.arrays[n.data].total_size
                    if isinstance(vol, sp.Expr):
                        vol = vol.subs(symbols_map)
                    proxy_in_volume += vol
                    if proxy.has_edge(iedge.src, sink):
                        proxy[iedge.src][sink]['capacity'] += vol
                        proxy[iedge.src][sink]['orig'].append(iedge)
                    else:
                        proxy.add_edge(
                            iedge.src, sink, capacity=vol, orig=[iedge]
                        )
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

    cut_val, (_, non_reachable) = nx.minimum_cut(
        proxy, source, sink, flow_func=nx.flow.edmonds_karp
    )

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
) -> Tuple[SDFG, TranslationDict]:
    affected_nodes = util.transformation_get_affected_nodes(p_sdfg, xform)

    if (isinstance(xform, SubgraphTransformation) or
        isinstance(xform, SingleStateTransformation)):
        if xform.sdfg_id >= 0 and p_sdfg.sdfg_list:
            sdfg = p_sdfg.sdfg_list[xform.sdfg_id]
        else:
            sdfg = p_sdfg

        state = sdfg.node(xform.state_id)

        reduced_affected_nodes = _reduce_in_configuration(
            state, sdfg, affected_nodes
        )

        translation_dict: TranslationDict = dict()
        ct = cutout(
            *reduced_affected_nodes, translation=translation_dict, state=state
        )
        util.translate_transformation(xform, sdfg, ct, translation_dict)
        return ct, translation_dict
    else:
        # For multistate cutouts we do not attempt to reduce the input
        # configuration, since the configuration is dependent on having a single
        # input state to the cutout.
        return _minimal_transformation_cutout(p_sdfg, xform, affected_nodes)


def _minimal_transformation_cutout(
    p_sdfg: SDFG, xform: Union[SubgraphTransformation, PatternTransformation],
    affected_nodes: Optional[Set[Union[nd.Node, SDFGState]]] = None
) -> Tuple[SDFG, TranslationDict]:
    if affected_nodes is None:
        affected_nodes = util.transformation_get_affected_nodes(p_sdfg, xform)

    if xform.sdfg_id >= 0 and p_sdfg.sdfg_list:
        sdfg = p_sdfg.sdfg_list[xform.sdfg_id]
    else:
        sdfg = p_sdfg

    if (isinstance(xform, SubgraphTransformation) or
        isinstance(xform, SingleStateTransformation)):
        state = sdfg.node(xform.state_id)
        translation_dict: TranslationDict = dict()
        ct = cutout(
            *affected_nodes, translation=translation_dict, state=state
        )
        util.translate_transformation(xform, sdfg, ct, translation_dict)
        return ct, translation_dict
    elif isinstance(xform, MultiStateTransformation):
        translation_dict: TranslationDict = dict()
        ct: SDFG = cutout(
            *affected_nodes, translation=translation_dict, state=None
        )
        o_sdfg = list(affected_nodes)[0].parent
        util.translate_transformation(xform, o_sdfg, ct, translation_dict)
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


def _stateset_frontier(states: Set[SDFGState]) -> Tuple[Set[SDFGState], Set]:
    """
    For a set of states, return the frontier.
    The frontier in this case refers to the predecessor states leading into the
    given set of states.
    :param states: The set of states for which to gather the frontier.
    :return: A 2-tuple with the state frontier, and all corresponding frontier
             edges.
    """
    frontier = set()
    frontier_edges = set()
    for state in states:
        for iedge in state.parent.in_edges(state):
            if iedge.src not in states:
                if iedge.src not in frontier:
                    frontier.add(iedge.src)
                if iedge not in frontier_edges:
                    frontier_edges.add(iedge)
    return frontier, frontier_edges


def multistate_cutout(
    *states: SDFGState, inserted_states: TranslationDict = None
) -> SDFG:
    """
    Cut out a multi-state subgraph from an SDFG to run separately for localized
    testing or optimization.
    The subgraph defined by the list of states will be extended to include anyl
    further states necessary to make the resulting cutout executable, i.e, to
    ensure that there is a distinct start state. This is achieved by gradually
    adding more states from the cutout's state machine frontier until a
    distinct, single entry state is obtained.
    :param states: The states in the subgraph to cut out.
    :param inserted_states: A dictionary that provides a mapping from the
                            original states to their cutout counterparts.
    """
    create_element = copy.deepcopy

    # Check that all states are inside the same SDFG.
    sdfg = list(states)[0].parent
    if any(i.parent != sdfg for i in states):
        raise Exception('Not all cutout states reside in the same SDFG')

    cutout_states: Set[SDFGState] = set(states)

    # Determine the start state and ensure there IS a unique start state. If
    # there is no unique start state, keep adding states from the predecessor
    # frontier in the state machine until a unique start state can be
    # determined.
    start_state: SDFGState = None
    for state in cutout_states:
        if state == sdfg.start_state:
            start_state = state
            break

    if start_state is None:
        bfs_queue = deque()
        bfs_queue.append(_stateset_frontier(cutout_states))

        while len(bfs_queue) > 0:
            frontier, frontier_edges = bfs_queue.popleft()
            if len(frontier_edges) == 0:
                # No explicit start state, but also no frontier to select from.
                return sdfg
            elif len(frontier_edges) == 1:
                # The destination of the only frontier edge must be the
                # start state, since only one edge leads into the subgraph.
                start_state = list(frontier_edges)[0].dst
            else:
                if len(frontier) == 0:
                    # No explicit start state, but also no frontier to select
                    # from.
                    return sdfg
                if len(frontier) == 1:
                    # For many frontier edges but only one frontier state,
                    # the frontier state is the new start state and is
                    # included in the cutout.
                    start_state = list(frontier)[0]
                    cutout_states.add(start_state)
                else:
                    for s in frontier:
                        cutout_states.add(s)
                    bfs_queue.append(_stateset_frontier(cutout_states))

    subgraph: SubgraphView = SubgraphView(sdfg, cutout_states)

    # Make a new SDFG with the included constants, used symbols, and data
    # containers.
    new_sdfg = SDFG(f'{sdfg.name}_cutout', sdfg.constants_prop)
    defined_symbols: Dict[str, data.Data] = dict()
    free_symbols: Set[str] = set()
    for state in cutout_states:
        free_symbols |= state.free_symbols
        state_defined_symbols = state.defined_symbols()
        for sym in state_defined_symbols:
            defined_symbols[sym] = state_defined_symbols[sym]
    for sym in free_symbols:
        new_sdfg.add_symbol(sym, defined_symbols[sym])

    for state in cutout_states:
        for dnode in state.data_nodes():
            if dnode.data in new_sdfg.arrays:
                continue
            new_desc = sdfg.arrays[dnode.data].clone()
            new_sdfg.add_datadesc(dnode.data, new_desc)

    # Add all states and state transitions required to the new cutout SDFG by
    # traversing the state machine edges.
    if inserted_states is None:
        inserted_states: TranslationDict = {}
    for is_edge in subgraph.edges():
        if is_edge.src not in inserted_states:
            inserted_states[is_edge.src] = create_element(is_edge.src)
            new_sdfg.add_node(
                inserted_states[is_edge.src],
                is_start_state=(is_edge.src == start_state)
            )
            inserted_states[is_edge.src].parent = new_sdfg
        if is_edge.dst not in inserted_states:
            inserted_states[is_edge.dst] = create_element(is_edge.dst)
            new_sdfg.add_node(
                inserted_states[is_edge.dst],
                is_start_state=(is_edge.dst == start_state)
            )
            inserted_states[is_edge.dst].parent = new_sdfg
        new_sdfg.add_edge(
            inserted_states[is_edge.src],
            inserted_states[is_edge.dst],
            create_element(is_edge.data)
        )

    # Add remaining necessary states.
    for state in subgraph.nodes():
        if state not in inserted_states:
            inserted_states[state] = create_element(state)
            new_sdfg.add_node(
                inserted_states[state], is_start_state=(state == start_state)
            )
            inserted_states[state].parent = new_sdfg

    inserted_states[sdfg.sdfg_id] = new_sdfg.sdfg_id

    return new_sdfg


def cutout_state(
    state: SDFGState, *nodes: nd.Node, make_copy: bool = True,
    inserted_nodes: TranslationDict = None
) -> SDFG:
    """
    Cut out a subgraph of a state from an SDFG to run separately for localized
    testing or optimization.
    The subgraph defined by the list of nodes will be extended to include access
    nodes of data containers necessary to run the graph separately. In addition,
    all transient data containers created outside the cut out graph will
    become global.
    :param state: The SDFG state in which the subgraph resides.
    :param nodes: The nodes in the subgraph to cut out.
    :param make_copy: If True, deep-copies every SDFG element in the copy.
                      Otherwise, original references are kept.
    :param inserted_nodes: A dictionary that maps nodes from the original SDFG
                           to their cutout counterparts.
    """
    create_element = copy.deepcopy if make_copy else (lambda x: x)
    sdfg = state.parent
    subgraph: StateSubgraphView = StateSubgraphView(state, nodes)
    subgraph = _extend_subgraph_with_access_nodes(state, subgraph)
    #other_arrays = _containers_defined_outside(sdfg, state, subgraph)

    # Make a new SDFG with the included constants, used symbols, and data
    # containers
    new_sdfg = SDFG(f'{state.parent.name}_cutout', sdfg.constants_prop)
    defined_syms = subgraph.defined_symbols()
    freesyms = subgraph.free_symbols
    for sym in freesyms:
        new_sdfg.add_symbol(sym, defined_syms[sym])

    if sdfg.parent_nsdfg_node is not None:
        for s, _ in sdfg.parent_nsdfg_node.symbol_mapping.items():
            if s not in new_sdfg.symbols and s in defined_syms:
                new_sdfg.add_symbol(s, defined_syms[s])

    for dnode in subgraph.data_nodes():
        if dnode.data in new_sdfg.arrays:
            continue
        new_desc = sdfg.arrays[dnode.data].clone()
        # If transient is defined outside, it becomes a global
        #if dnode.data in other_arrays:
        #    new_desc.transient = False
        new_sdfg.add_datadesc(dnode.data, new_desc)

    # Add a single state with the extended subgraph
    new_state = new_sdfg.add_state(state.label, is_start_state=True)
    if inserted_nodes is None:
        inserted_nodes: TranslationDict = {}
    for e in subgraph.edges():
        if e.src not in inserted_nodes:
            inserted_nodes[e.src] = create_element(e.src)
        if e.dst not in inserted_nodes:
            inserted_nodes[e.dst] = create_element(e.dst)
        new_state.add_edge(
            inserted_nodes[e.src], e.src_conn, inserted_nodes[e.dst],
            e.dst_conn, create_element(e.data)
        )

    # Insert remaining isolated nodes
    for n in subgraph.nodes():
        if n not in inserted_nodes:
            inserted_nodes[n] = create_element(n)
            new_state.add_node(inserted_nodes[n])

    # Remove remaining dangling connectors from scope nodes and add new data
    # containers and corresponding accesses for dangling connectors on other
    # nodes
    for orig_node in inserted_nodes.keys():
        new_node = inserted_nodes[orig_node]
        if isinstance(orig_node, (nd.EntryNode, nd.ExitNode)):
            used_connectors = set(
                e.dst_conn for e in new_state.in_edges(new_node)
            )
            for conn in (new_node.in_connectors.keys() - used_connectors):
                new_node.remove_in_connector(conn)
            used_connectors = set(
                e.src_conn for e in new_state.out_edges(new_node)
            )
            for conn in (new_node.out_connectors.keys() - used_connectors):
                new_node.remove_out_connector(conn)
        else:
            used_connectors = set(
                e.dst_conn for e in new_state.in_edges(new_node)
            )
            for conn in (new_node.in_connectors.keys() - used_connectors):
                for e in state.in_edges(orig_node):
                    if e.dst_conn and e.dst_conn == conn:
                        _create_alibi_access_node_for_edge(
                            new_sdfg, new_state, sdfg, e, None, None, new_node,
                            conn
                        )
                        prune = False
                        break
                if prune:
                    new_node.remove_in_connector(conn)
            used_connectors = set(
                e.src_conn for e in new_state.out_edges(new_node)
            )
            for conn in (new_node.out_connectors.keys() - used_connectors):
                prune = True
                for e in state.out_edges(orig_node):
                    if e.src_conn and e.src_conn == conn:
                        _create_alibi_access_node_for_edge(
                            new_sdfg, new_state, sdfg, e, new_node, conn, None,
                            None
                        )
                        prune = False
                        break
                if prune:
                    new_node.remove_out_connector(conn)

    inserted_nodes[state] = new_state
    inserted_nodes[sdfg.sdfg_id] = new_sdfg.sdfg_id

    return new_sdfg


def _create_alibi_access_node_for_edge(
    target_sdfg: SDFG, target_state: SDFGState, original_sdfg: SDFG,
    original_edge: MultiConnectorEdge[Memlet], from_node: Union[nd.Node, None],
    from_connector: Union[str, None], to_node: Union[nd.Node, None],
    to_connector: Union[str, None]
) -> data.Data:
    print('making an alibi node')
    """
    Add an alibi data container and access node to a dangling connector inside
    of scopes.
    """
    original_edge.data
    access_size = original_edge.data.subset.size_exact()
    container_name = '__cutout_' + str(original_edge.data.data)
    container_name = data.find_new_name(
        container_name, target_sdfg._arrays.keys()
    )
    original_array = original_sdfg._arrays[original_edge.data.data]
    memlet_str = ''
    if original_edge.data.subset.num_elements_exact() > 1:
        access_size = original_edge.data.subset.size_exact()
        target_sdfg.add_array(
            container_name, access_size, original_array.dtype
        )
        memlet_str = container_name + '['
        sep = None
        for dim_len in original_edge.data.subset.bounding_box_size():
            if sep is not None:
                memlet_str += ','
            if dim_len > 1:
                memlet_str += '0:' + str(dim_len - 1)
            else:
                memlet_str += '0'
            sep = ','
        memlet_str += ']'
    else:
        target_sdfg.add_scalar(container_name, original_array.dtype)
        memlet_str = container_name + '[0]'
    alibi_access_node = target_state.add_access(container_name)
    if from_node is None:
        target_state.add_edge(
            alibi_access_node, None, to_node, to_connector, Memlet(
                memlet_str
            )
        )
    else:
        target_state.add_edge(
            from_node, from_connector, alibi_access_node, None, Memlet(
                memlet_str
            )
        )


def _extend_subgraph_with_access_nodes(
    state: SDFGState, subgraph: StateSubgraphView
) -> StateSubgraphView:
    """
    Expands a subgraph view to include necessary input/output access nodes,
    using memlet paths.
    """
    sdfg = state.parent
    result: List[nd.Node] = copy.copy(subgraph.nodes())
    queue: Deque[nd.Node] = deque(subgraph.nodes())

    # Add all nodes in memlet paths
    while len(queue) > 0:
        node = queue.pop()
        if isinstance(node, nd.AccessNode):
            if isinstance(node.desc(sdfg), data.View):
                vnode = sdutil.get_view_node(state, node)
                result.append(vnode)
                queue.append(vnode)
            continue
        for e in state.in_edges(node):
            # Special case: IN_* connectors are not traversed further
            if (isinstance(e.dst, (nd.EntryNode, nd.ExitNode)) and
                (e.dst_conn is None or e.dst_conn.startswith('IN_'))):
                continue

            # We don't want to extend access nodes over scope entry nodes, but
            # rather we want to introduce alibi data containers for the correct
            # subset instead. Handled separately.
            if (isinstance(e.src, nd.EntryNode) and e.src not in result and
                state.exit_node(e.src) not in result):
                continue
            else:
                mpath = state.memlet_path(e)
                new_nodes = [mpe.src for mpe in mpath if mpe.src not in result]
                result.extend(new_nodes)
                # Memlet path may end in a code node, continue traversing and
                # expanding graph
                queue.extend(new_nodes)

        for e in state.out_edges(node):
            # Special case: OUT_* connectors are not traversed further
            if (isinstance(e.src, (nd.EntryNode, nd.ExitNode)) and
                (e.src_conn is None or e.src_conn.startswith('OUT_'))):
                continue

            # We don't want to extend access nodes over scope entry nodes, but
            # rather we want to introduce alibi data containers for the correct
            # subset instead. Handled separately.
            if (isinstance(e.dst, nd.ExitNode) and e.dst not in result and
                state.entry_node(e.dst) not in result):
                continue
            else:
                mpath = state.memlet_path(e)
                new_nodes = [mpe.dst for mpe in mpath if mpe.dst not in result]
                result.extend(new_nodes)
                # Memlet path may end in a code node, continue traversing and
                # expanding graph
                queue.extend(new_nodes)

    # Check for mismatch in scopes
    for node in result:
        enode = None
        if isinstance(node, nd.EntryNode) and state.exit_node(node) not in result:
            enode = state.exit_node(node)
        if isinstance(node, nd.ExitNode) and state.entry_node(node) not in result:
            enode = state.entry_node(node)
        if enode is not None:
            raise ValueError(f'Cutout cannot expand graph implicitly since "{node}" is in the graph and "{enode}" is '
                             'not. Please provide more nodes in the subgraph as necessary.')

    return StateSubgraphView(state, result)


def _containers_defined_outside(sdfg: SDFG, state: SDFGState, subgraph: StateSubgraphView) -> Set[str]:
    """ Returns a list of containers set outside the given subgraph. """
    # Since we care about containers that are written to, we only need to look at access nodes rather than interstate
    # edges
    result: Set[str] = set()
    for ostate in sdfg.nodes():
        for node in ostate.data_nodes():
            if ostate is not state or node not in subgraph.nodes():
                if ostate.in_degree(node) > 0:
                    result.add(node.data)

    # Add all new sink nodes of new subgraph
    for dnode in subgraph.data_nodes():
        if subgraph.out_degree(dnode) == 0 and state.out_degree(dnode) > 0:
            result.add(dnode.data)

    return result


def cutout(
    *nodes: Union[nd.Node, SDFGState], translation: TranslationDict,
    state: Optional[SDFGState] = None
) -> SDFG:
    if state is not None:
        if any([isinstance(n, SDFGState) for n in nodes]):
            raise Exception(
                'Mixing cutout nodes of type Node and SDFGState is not allowed'
            )
        new_sdfg = cutout_state(state, *nodes, make_copy=True, inserted_nodes=translation)
    else:
        if any([isinstance(n, nd.Node) for n in nodes]):
            raise Exception(
                'Mixing cutout nodes of type Node and SDFGState is not allowed'
            )
        new_sdfg = multistate_cutout(*nodes, inserted_states=translation)

    # Ensure the parent relationships and SDFG list is correct.
    for s in new_sdfg.states():
        for node in s.nodes():
            if isinstance(node, nd.NestedSDFG):
                node.sdfg._parent_sdfg = new_sdfg
    new_sdfg.reset_sdfg_list()

    return new_sdfg


def _determine_state_input_config(sdfg: SDFG, state: SDFGState) -> Set[str]:
    input_configuration = set()

    check_for_write_before = set()

    state_reach_sdfgs = StateReachability().apply_pass(sdfg, None)
    state_reach = state_reach_sdfgs[sdfg.sdfg_id]
    inverse_reach: Set[SDFGState] = set()

    for dn in state.data_nodes():
        array = sdfg.arrays[dn.data]
        if not array.transient:
            # Non-transients are always part of the system state.
            input_configuration.add(dn.data)
        elif state.out_degree(dn) > 0:
            # This is read from, add to the system state if it is written
            # anywhere else in the graph.
            check_for_write_before.add(dn.data)

    for k, v in state_reach.items():
        if k != state and state in v:
            inverse_reach.add(k)

    for state in inverse_reach:
        for dn in state.data_nodes():
            if state.in_degree(dn) > 0:
                # For any writes, check if they are reads from the cutout that
                # need to be checked. If they are, they're part of the system
                # state.
                if dn.data in check_for_write_before:
                    input_configuration.add(dn.data)

    return input_configuration


def cutout_determine_input_config(
    ct: SDFG, sdfg: SDFG, translation_dict: TranslationDict,
    system_state: Set[str] = None
) -> Set[str]:
    input_configuration = set()

    check_for_write_before = set()

    original_sdfg_id = None
    for k, v in translation_dict.items():
        if v == sdfg.sdfg_id:
            original_sdfg_id = k
            break
    if original_sdfg_id is None:
        raise KeyError('Could not find SDFG ID in translation')

    state_reach_sdfgs = StateReachability().apply_pass(
        sdfg.sdfg_list[original_sdfg_id], None
    )
    state_reach = state_reach_sdfgs[original_sdfg_id]
    inverse_cutout_reach: Set[SDFGState] = set()
    cutout_states = set(ct.states())

    must_have_descriptors = set()
    if system_state is not None:
        must_have_descriptors.update(system_state)

    for state in cutout_states:
        for dn in state.data_nodes():
            if dn.data in must_have_descriptors:
                must_have_descriptors.remove(dn.data)

            array = ct.arrays[dn.data]
            if not array.transient:
                # Non-transients are always part of the system state.
                input_configuration.add(dn.data)
            elif state.out_degree(dn) > 0:
                # This is read from, add to the system state if it is written
                # anywhere else in the graph.
                check_for_write_before.add(dn.data)

        original_state = None
        for k, v in translation_dict.items():
            if v == state:
                original_state = k
                break
        for k, v in state_reach.items():
            if ((k not in translation_dict or
                 translation_dict[k] not in cutout_states)
                 and original_state is not None and original_state in v):
                inverse_cutout_reach.add(k)

        # If the cutout consists of only one state, we need to check inside the
        # same state of the original SDFG as well.
        if len(cutout_states) == 1:
            if original_state is None:
                raise KeyError('Could not find state in translation')
            for dn in original_state.data_nodes():
                if original_state.in_degree(dn) > 0:
                    iedges = original_state.in_edges(dn)
                    if any([i.src not in translation_dict for i in iedges]):
                        if dn.data in check_for_write_before:
                            input_configuration.add(dn.data)

    for state in inverse_cutout_reach:
        for dn in state.data_nodes():
            if state.in_degree(dn) > 0:
                # For any writes, check if they are reads from the cutout that
                # need to be checked. If they are, they're part of the system
                # state.
                if dn.data in check_for_write_before:
                    input_configuration.add(dn.data)

    # Anything that doesn't have a correpsonding access node must be used as
    # well.
    for desc in must_have_descriptors:
        if desc not in input_configuration:
            input_configuration.add(desc)

    return input_configuration


def cutout_determine_system_state(
    ct: SDFG, sdfg: SDFG, translation_dict: TranslationDict
) -> Set[str]:
    system_state = set()

    check_for_read_after = set()

    original_sdfg_id = None
    for k, v in translation_dict.items():
        if v == sdfg.sdfg_id:
            original_sdfg_id = k
            break
    if original_sdfg_id is None:
        raise KeyError('Could not find SDFG ID in translation')

    state_reach_sdfgs = StateReachability().apply_pass(
        sdfg.sdfg_list[original_sdfg_id], None
    )
    state_reach = state_reach_sdfgs[original_sdfg_id]
    cutout_reach: Set[SDFGState] = set()
    cutout_states = set(ct.states())

    for state in cutout_states:
        for dn in state.data_nodes():
            array = ct.arrays[dn.data]
            if not array.transient:
                # Non-transients are always part of the system state.
                system_state.add(dn.data)
            elif state.in_degree(dn) > 0:
                # This is written to, add to the system state if it is read
                # anywhere else in the graph.
                check_for_read_after.add(dn.data)

        original_state = None
        for k, v in translation_dict.items():
            if v == state:
                original_state = k
                break
        if original_state is None:
            raise KeyError('Could not find state in translation')
        for rstate in state_reach[original_state]:
            if (rstate not in translation_dict or
                translation_dict[rstate] not in cutout_states):
                cutout_reach.add(rstate)

        # If the cutout consists of only one state, we need to check inside the
        # same state of the original SDFG as well.
        if len(cutout_states) == 1:
            for dn in original_state.data_nodes():
                if original_state.out_degree(dn) > 0:
                    oedges = original_state.out_edges(dn)
                    if any([o.dst not in translation_dict for o in oedges]):
                        if dn.data in check_for_read_after:
                            system_state.add(dn.data)

    for state in cutout_reach:
        for dn in state.data_nodes():
            if state.out_degree(dn) > 0:
                # For any reads, check if they are writes from the cutout that
                # need to be checked. If they are, they're part of the system
                # state.
                if dn.data in check_for_read_after:
                    system_state.add(dn.data)

    return system_state
