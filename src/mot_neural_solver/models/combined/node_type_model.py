from typing import Dict, List
import torch
import torch.nn as nn

from mot_neural_solver.models.combined.types import NodeType

class NodeTypeModel(nn.Module):
    """
    Class used to peform the node update during Neural mwssage passing
    """
    def __init__(self,
                 message_models: Dict[NodeType, torch.nn.Module],
                 update_model: torch.nn.Module,
                 node_type: NodeType,
                 node_types: List[NodeType]):
        super(NodeTypeModel, self).__init__()

        self.message_models = message_models
        self.update_model = update_model
        self.node_type = node_type
        self.node_types = node_types

    def forward(self, x, edge_indexs, edge_attrs):
        messages = []
        for tpe in self.node_types:
            message = self.message_models[tpe](x, edge_indexs[tpe], edge_attrs[tpe])
            messages.append(message)

        input = torch.cat(messages, dim=1)
        output = self.update_model(input)

        return output