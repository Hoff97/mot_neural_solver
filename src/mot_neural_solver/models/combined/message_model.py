from typing import Dict, List
import torch
import torch.nn as nn

from mot_neural_solver.models.combined.types import NodeType

class MessageModel(nn.Module):
    """
    Generates and aggregates messages for a single node type
    """
    def __init__(self,
                 mlp: nn.Module,
                 node_agg_fn):
        super(MessageModel, self).__init__()

        self.mlp = mlp
        self.node_agg_fn = node_agg_fn

    def forward(self, x, edge_index, edge_attr):
        row, col = edge_index
        input = torch.cat([x[col], edge_attr], dim=1)
        messages = self.mlp(input)
        message = self.node_agg_fn(messages, row, x.size(0))

        return message