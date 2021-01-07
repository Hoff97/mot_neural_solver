from typing import Dict, List

import torch
import torch.nn as nn
from mot_neural_solver.models.combined.types import NodeType


class NodeTypeModel(nn.Module):
    """
    Class used to peform the node update during Neural mwssage passing
    """

    def __init__(
        self,
        message_models: Dict[NodeType, torch.nn.Module],
        update_model: torch.nn.Module,
        node_type: NodeType,
        node_types: List[NodeType],
    ):
        super(NodeTypeModel, self).__init__()

        self.message_models = torch.nn.ModuleDict(message_models)
        self.update_model = update_model
        self.node_type = node_type
        self.node_types = node_types
        self.node_type_ix = self.node_types.index(self.node_type)

    def forward(self, x, xs, edge_indexs, edge_attrs):
        messages = []
        for i, tpe in enumerate(self.node_types):
            edge_indices = edge_indexs[tpe]
            attrs = edge_attrs[tpe]

            x_in = xs[tpe]

            if i < self.node_type_ix:
                edge_indices = edge_indices[[1, 0]]

            message = self.message_models[tpe](x, x_in, edge_indices, attrs)
            messages.append(message)

        input = torch.cat(messages, dim=1)
        output = self.update_model(input)

        return output
