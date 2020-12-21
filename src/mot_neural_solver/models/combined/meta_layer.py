from typing import Dict, List, Optional, Tuple
import torch

from mot_neural_solver.models.combined.types import EdgeType, NodeType


class MetaLayer(torch.nn.Module):
    """
    Core Message Passing Network Class. Extracted from torch_geometric, with minor modifications.
    (https://rusty1s.github.io/pytorch_geometric/build/html/modules/nn.html)
    """
    def __init__(self,
                 node_types: List[NodeType],
                 edge_models: Optional[Dict[EdgeType, torch.nn.Module]]=None,
                 node_type_models: Optional[Dict[NodeType, torch.nn.Module]]=None):
        super(MetaLayer, self).__init__()

        self.edge_models = edge_models
        self.node_type_models = node_type_models
        self.reset_parameters()

        self.node_types = node_types

    def reset_parameters(self):
        for item in self.node_models.values():
            if hasattr(item, 'reset_parameters'):
                item.reset_parameters()
        for item in self.edge_models.values():
            if hasattr(item, 'reset_parameters'):
                item.reset_parameters()

    def forward(self,
                xs: Dict[NodeType, torch.Tensor],
                edge_indexs: Dict[EdgeType, torch.Tensor],
                edge_attrs: Dict[EdgeType, torch.Tensor]):
        new_edge_attrs: Dict[EdgeType, torch.Tensor] = {}

        for i, tpe1 in enumerate(self.node_types):
            for tpe2 in self.node_types[i:]:
                row, col = edge_indexs[[tpe1, tpe2]]

                x1 = xs[tpe1][row]
                x2 = xs[tpe2][col]
                edge_attr = edge_attrs[(tpe1, tpe2)]

                # TODO: Check if edge model is defined
                new_edge_attrs[(tpe1, tpe2)] = self.edge_models[(tpe1, tpe2)](x1, x2, edge_attr)

        new_xs: Dict[NodeType, torch.Tensor] = {}
        for i, tpe in enumerate(self.node_types):
            x = xs[tpe]
            edge_index: Dict[NodeType, torch.Module] = {}
            edge_attr: Dict[NodeType, torch.Tensor] = {}
            for tpe2 in self.node_types[:i]:
                edge_index[tpe2] = edge_indexs[(tpe2, tpe)]
                edge_attr[tpe2] = new_edge_attrs[(tpe2, tpe)]
            for tpe2 in self.node_types[i:]:
                edge_index[tpe2] = edge_indexs[(tpe, tpe2)]
                edge_attr[tpe2] = new_edge_attrs[(tpe, tpe2)]

            new_xs[tpe] = self.node_type_models[tpe](x, edge_index, edge_attr)

        return new_xs, new_edge_attrs

    def __repr__(self):
        # TODO: Fix
        return '{}(edge_model={}, node_model={})'.format(self.__class__.__name__, self.edge_model, self.node_model)