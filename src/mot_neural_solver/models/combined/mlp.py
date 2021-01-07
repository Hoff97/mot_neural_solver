from typing import Dict, List, Optional, Tuple

import torch
from mot_neural_solver.models.combined.types import EdgeType, NodeType
from mot_neural_solver.models.mlp import MLP
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

    def __init__(
        self,
        node_types: List[str],
        edge_in_dims: Optional[Dict[EdgeType, int]] = None,
        node_in_dims: Optional[Dict[NodeType, int]] = None,
        edge_out_dims: Optional[Dict[EdgeType, int]] = None,
        node_out_dims: Optional[Dict[NodeType, int]] = None,
        node_fc_dims: Optional[Dict[NodeType, List[int]]] = None,
        edge_fc_dims: Optional[Dict[EdgeType, List[int]]] = None,
        dropout_p=None,
        use_batchnorm=None,
    ):
        super(MLPGraphIndependent, self).__init__()
        self.node_types = node_types

        if node_in_dims is not None:
            self.node_mlps = {}
            for tpe in node_types:
                if tpe in node_in_dims:
                    mlp = MLP(
                        input_dim=node_in_dims[tpe],
                        fc_dims=list(node_fc_dims[tpe]) + [node_out_dims[tpe]],
                        dropout_p=dropout_p,
                        use_batchnorm=use_batchnorm,
                    )
                    self.node_mlps[tpe] = mlp

            self.node_mlps = nn.ModuleDict(self.node_mlps)
        else:
            self.node_mlps = None

        if edge_in_dims is not None:
            self.edge_mlps = {}
            for i, tpe1 in enumerate(node_types):
                for tpe2 in node_types[i:]:
                    key = f"{tpe1}-{tpe2}"
                    if key in edge_in_dims:
                        mlp = MLP(
                            input_dim=edge_in_dims[key],
                            fc_dims=list(edge_fc_dims[key]) + [edge_out_dims[key]],
                            dropout_p=dropout_p,
                            use_batchnorm=use_batchnorm,
                        )
                        self.edge_mlps[key] = mlp
            self.edge_mlps = nn.ModuleDict(self.edge_mlps)
        else:
            self.edge_mlps = None

    def forward(
        self,
        edge_feats: Optional[Dict[EdgeType, torch.Tensor]] = None,
        nodes_feats: Optional[Dict[NodeType, torch.Tensor]] = None,
    ):

        if self.node_mlps is not None:
            out_node_feats = {}
            for tpe in self.node_types:
                if tpe in self.node_mlps and tpe in nodes_feats:
                    out_node_feats[tpe] = self.node_mlps[tpe](nodes_feats[tpe])
                elif tpe in nodes_feats:
                    # We just pass through the node features for this
                    # particular node type
                    out_node_feats[tpe] = nodes_feats[tpe]
        else:
            out_node_feats = nodes_feats

        if self.edge_mlps is not None:
            out_edge_feats = {}
            for i, tpe1 in enumerate(self.node_types):
                for tpe2 in self.node_types[i:]:
                    key = f"{tpe1}-{tpe2}"
                    if key in self.edge_mlps and key in edge_feats:
                        out_edge_feats[key] = self.edge_mlps[key](edge_feats[key])
                    elif key in edge_feats:
                        # We just pass through the edge features for this
                        # particular edge type
                        out_edge_feats[key] = edge_feats[key]
        else:
            out_edge_feats = edge_feats

        return out_edge_feats, out_node_feats
