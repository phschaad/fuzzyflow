# Copyright 2022 ETH Zurich and the FuzzyFlow authors. All rights reserved.
# This file is part of FuzzyFlow, which is released under the BSD 3-Clause
# License. For details, see the LICENSE file.

import json
from typing import Dict, Set, Tuple, Union
from dace import serialize
from dace.sdfg import SDFG, nodes as nd, SDFGState
from dace.sdfg.state import StateSubgraphView
from dace.transformation.transformation import PatternTransformation, SubgraphTransformation


def transformation_get_affected_nodes(
    sdfg: SDFG, xform: Union[PatternTransformation, SubgraphTransformation]
) -> Set[nd.Node]:
    affected_nodes: Set[nd.Node] = set()
    if isinstance(xform, PatternTransformation):
        for k, _ in xform._get_pattern_nodes().items():
            affected_nodes.add(getattr(xform, k))
    else:
        sgv = xform.get_subgraph(sdfg)
        if isinstance(sgv, StateSubgraphView):
            for n in sgv.nodes():
                affected_nodes.add(n)
        else:
            raise NotImplementedError(
                'Multi-state subgraph transformations are not available yet'
            )
    return affected_nodes


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


# NOTE: This requires the target_sdfg be a REFERENCE cutout of the original_sdfg
# instead of a deep copy! It will not work otherwise.
def translate_transformation(
    xform: Union[PatternTransformation, SubgraphTransformation],
    state: SDFGState, original_sdfg: SDFG, target_sdfg: SDFG,
    affected_nodes: Set[nd.Node] = None
) -> None:
    if affected_nodes is None:
        affected_nodes = transformation_get_affected_nodes(original_sdfg, xform)

    target_state = target_sdfg.nodes()[0]

    xform.state_id = target_sdfg.node_id(target_state)
    xform._sdfg = target_sdfg

    translate_dict: Dict[int, int] = dict()
    for n in affected_nodes:
        n_id = state.node_id(n)
        tnode_id = target_state.node_id(n)
        translate_dict[n_id] = tnode_id
    for k in xform.subgraph.keys():
        prev = xform.subgraph[k]
        if prev in translate_dict:
            newval = translate_dict[prev]
            xform.subgraph[k] = newval
