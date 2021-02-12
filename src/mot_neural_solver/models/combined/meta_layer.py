from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from mot_neural_solver.models.combined.node_type_model import NodeTypeModel
from mot_neural_solver.models.combined.types import EdgeType, NodeType
from mot_neural_solver.models.mpn import EdgeModel


class MetaLayer(torch.nn.Module):
    """
    Core Message Passing Network Class. Extracted from torch_geometric, with minor modifications.
    (https://rusty1s.github.io/pytorch_geometric/build/html/modules/nn.html)
    """

    def __init__(
        self,
        node_types: List[NodeType],
        edge_models: Optional[Dict[EdgeType, EdgeModel]] = None,
        node_type_models: Optional[Dict[NodeType, NodeTypeModel]] = None,
    ):
        super(MetaLayer, self).__init__()

        self.edge_models = nn.ModuleDict(edge_models)
        self.node_type_models = nn.ModuleDict(node_type_models)
        self.reset_parameters()

        self.node_types = node_types

    def reset_parameters(self):
        for item in self.node_type_models.values():
            if hasattr(item, "reset_parameters"):
                item.reset_parameters()
        for item in self.edge_models.values():
            if hasattr(item, "reset_parameters"):
                item.reset_parameters()

    def forward(
        self,
        xs: Dict[NodeType, torch.Tensor],
        edge_indexs: Dict[EdgeType, torch.Tensor],
        edge_attrs: Dict[EdgeType, torch.Tensor],
    ):
        new_edge_attrs: Dict[EdgeType, torch.Tensor] = {}

        for i, tpe1 in enumerate(self.node_types):
            for tpe2 in self.node_types[i:]:
                key = f"{tpe1}-{tpe2}"
                if key in self.edge_models:
                    row, col = edge_indexs[key]

                    x1 = xs[tpe1][row]
                    x2 = xs[tpe2][col]
                    edge_attr = edge_attrs[key]

                    new_edge_attrs[key] = self.edge_models[key](x1, x2, edge_attr)
                else:
                    new_edge_attrs[key] = edge_attrs[key]

        new_xs: Dict[NodeType, torch.Tensor] = {}
        for i, tpe in enumerate(self.node_types):
            x = xs[tpe]
            edge_index: Dict[NodeType, torch.Module] = {}
            edge_attr: Dict[NodeType, torch.Tensor] = {}
            for tpe2 in self.node_types[:i]:
                key = f"{tpe2}-{tpe}"
                edge_index[tpe2] = edge_indexs[key]
                edge_attr[tpe2] = new_edge_attrs[key]
            for tpe2 in self.node_types[i:]:
                key = f"{tpe}-{tpe2}"
                edge_index[tpe2] = edge_indexs[key]
                edge_attr[tpe2] = new_edge_attrs[key]

            new_xs[tpe] = self.node_type_models[tpe](x, xs, edge_index, edge_attr)

        return new_xs, new_edge_attrs

    def __repr__(self):
        # TODO: Fix
        return "{}(edge_model={}, node_model={})".format(
            self.__class__.__name__, self.edge_model, self.node_model
        )
