# Copyright 2022 ETH Zurich and the FuzzyFlow authors. All rights reserved.
# This file is part of FuzzyFlow, which is released under the BSD 3-Clause
# License. For details, see the LICENSE file.

from enum import Enum
from typing import Dict, List, Set, Tuple, Union
import sympy as sp

from dace.sdfg import SDFG, SDFGState
from dace.sdfg import nodes as nd
from dace.sdfg.analysis import cutout as dcut
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
) -> SDFG:
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
) -> SDFG:
    affected_nodes = util.transformation_get_affected_nodes(sdfg, xform)
    if (isinstance(xform, SubgraphTransformation) or
        isinstance(xform, SingleStateTransformation)):
        state = sdfg.node(xform.state_id)
        translation_dict: Dict[nd.Node, nd.Node] = dict()
        ct: SDFG = dcut.cutout_state(
            state, *affected_nodes, make_copy=True,
            inserted_nodes=translation_dict
        )
        util.translate_transformation(xform, state, ct, translation_dict)

        return ct
    elif isinstance(xform, MultiStateTransformation):
        raise NotImplementedError('Multistate cutouts not yet supported')
    raise Exception('This type of transformation cannot be supported')


def find_cutout_for_transformation(
    sdfg: SDFG, xform: Union[SubgraphTransformation, PatternTransformation],
    strategy: CutoutStrategy
) -> SDFG:
    if strategy == CutoutStrategy.SIMPLE:
        return _minimal_transformation_cutout(sdfg, xform)
    elif strategy == CutoutStrategy.MINIMUM_DOMINATOR_FLOW:
        return _minimum_dominator_flow_cutout(sdfg, xform)
    raise Exception('Unknown cutout strategy')
