# Copyright 2022 ETH Zurich and the FuzzyFlow authors. All rights reserved.
# This file is part of FuzzyFlow, which is released under the BSD 3-Clause
# License. For details, see the LICENSE file.

from enum import Flag
from typing import Set, Union
from dace.sdfg.analysis import cutout as dcut
from dace.sdfg import SDFG, nodes as nd, ScopeSubgraphView
from dace.transformation.transformation import SubgraphTransformation, PatternTransformation

from fuzzyflow import util


class CutoutStrategy(Flag):
    SIMPLE = 0


def _minimal_transformation_cutout(
    sdfg: SDFG, xform: Union[SubgraphTransformation, PatternTransformation]
) -> SDFG:
    state = sdfg.node(xform.state_id)

    affected_nodes = util.transformation_get_affected_nodes(sdfg, xform)
    cutout_nodes: Set[nd.Node] = set()
    for node in affected_nodes:
        cutout_nodes.add(node)
        if isinstance(node, nd.EntryNode):
            scope: ScopeSubgraphView = state.scope_subgraph(
                node, include_entry=True, include_exit=True
            )
            for n in scope.nodes():
                cutout_nodes.add(n)
        elif isinstance(node, nd.ExitNode):
            entry = state.entry_node(node)
            scope: ScopeSubgraphView = state.scope_subgraph(
                entry, include_entry=True, include_exit=True
            )
            for n in scope.nodes():
                cutout_nodes.add(n)

    ct: SDFG = dcut.cutout_state(state, *cutout_nodes, make_copy=False)

    util.translate_transformation(xform, state, sdfg, ct, affected_nodes)

    return ct


def find_cutout_for_transformation(
    sdfg: SDFG, xform: Union[SubgraphTransformation, PatternTransformation],
    strategy: CutoutStrategy
) -> SDFG:
    if strategy == CutoutStrategy.SIMPLE:
        return _minimal_transformation_cutout(sdfg, xform)
    raise Exception('Unknown cutout strategy')
