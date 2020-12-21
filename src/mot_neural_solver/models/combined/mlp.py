from typing import Dict, List, Optional, Tuple
from mot_neural_solver.models.combined.types import EdgeType, NodeType
from mot_neural_solver.models.mlp import MLP
import torch
from torch import nn

class MLPGraphIndependent(nn.Module):
    """
    Class used to to encode (resp. classify) features before (resp. after) neural message passing
    for graphs with a number (N) of different node types. The node types have names.

    It consists of N node MLP's (which map the input features of each of the input node types)
    and N(N+1)/2 edge MLP's (one MLP for each combination of node types).

    The node mlps are indexed by their respective node type.

    The edge mlps are indexed by a tuple of their respective node type combination.
    """

    def __init__(self,
                 node_types: List[str],
                 edge_in_dims: Optional[Dict[EdgeType, int]] = None,
                 node_in_dims: Optional[Dict[NodeType, int]] = None,
                 edge_out_dims: Optional[Dict[EdgeType, int]] = None,
                 node_out_dims: Optional[Dict[NodeType, int]] = None,
                 node_fc_dims: Optional[Dict[NodeType, List[int]]] = None,
                 edge_fc_dims: Optional[Dict[EdgeType, List[int]]] = None,
                 dropout_p = None,
                 use_batchnorm = None):
        super(MLPGraphIndependent, self).__init__()
        self.node_types = node_types

        if node_in_dims is not None :
            self.node_mlps = {}
            for tpe in node_types:
                if tpe in node_in_dims:
                    mlp = MLP(input_dim=node_in_dims[tpe], fc_dims=list(node_fc_dims[tpe]) + [node_out_dims[tpe]],
                            dropout_p=dropout_p, use_batchnorm=use_batchnorm)
                    self.node_mlps[tpe] = mlp
        else:
            self.node_mlps = None

        if edge_in_dims is not None:
            self.edge_mlps = {}
            for i, tpe1 in enumerate(node_types):
                for tpe2 in node_types[i:]:
                    if (tpe1, tpe2) in edge_in_dims:
                        mlp = MLP(input_dim=edge_in_dims[(tpe1, tpe2)],
                                fc_dims=list(edge_fc_dims[(tpe1, tpe2)]) + [edge_out_dims[(tpe1, tpe2)]],
                                dropout_p=dropout_p, use_batchnorm=use_batchnorm)
                        self.edge_mlps[(tpe1, tpe2)] = mlp
        else:
            self.edge_mlps = None

    def forward(self,
                edge_feats: Optional[Dict[EdgeType, torch.Tensor]] = None,
                nodes_feats: Optional[Dict[NodeType, torch.Tensor]] = None):

        if self.node_mlps is not None:
            out_node_feats = {}
            for tpe in self.node_types:
                if tpe in self.node_mlps and tpe in nodes_feats:
                    out_node_feats[tpe] = self.node_mlps[tpe](nodes_feats[tpe])
        else:
            out_node_feats = nodes_feats

        if self.edge_mlps is not None:
            out_edge_feats = {}
            for i, tpe1 in enumerate(self.node_types):
                for tpe2 in self.node_types[i:]:
                    if (tpe1, tpe2) in self.edge_mlps and (tpe1, tpe2) in edge_feats:
                        out_edge_feats[(tpe1, tpe2)] = self.edge_mlps[(tpe1, tpe2)](edge_feats[(tpe1, tpe2)])
        else:
            out_edge_feats = edge_feats

        return out_edge_feats, out_node_feats