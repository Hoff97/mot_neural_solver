from typing import List, Optional, Tuple

import torch
from mot_neural_solver.models.combined.message_model import MessageModel
from mot_neural_solver.models.combined.meta_layer import MetaLayer
from mot_neural_solver.models.combined.mlp import MLPGraphIndependent
from mot_neural_solver.models.combined.node_type_model import NodeTypeModel
from mot_neural_solver.models.combined.time_aware_message_model import (
    TimeAwareMessageModel,
)
from mot_neural_solver.models.mlp import MLP
from mot_neural_solver.models.mpn import EdgeModel
from torch import nn
from torch_scatter import scatter_add, scatter_max, scatter_mean


class MOTMPNet(nn.Module):
    """
    Main Model Class. Contains all the components of the model. It consists of of several networks:
    - Encoder networks:
        - N encoder MLPs for the N different node types
        - N^2 encoder MLPs for the N^2 different combinations of edges
    TODO:
    - 1 edge classifier MLP that performs binary classification over the Message Passing Network's output.
    """

    def __init__(self, model_params, bb_encoder=None):
        super(MOTMPNet, self).__init__()

        self.node_cnn = bb_encoder
        self.model_params = model_params

        self.node_types = model_params["node_types"]

        # Define Encoder and Classifier Networks
        encoder_feats_dict = model_params["encoder_feats_dict"]
        classifier_feats_dict = model_params["classifier_feats_dict"]

        self.encoder = MLPGraphIndependent(
            node_types=self.node_types, **encoder_feats_dict
        )
        self.classifier = MLPGraphIndependent(
            node_types=self.node_types, **classifier_feats_dict
        )

        # Define the 'Core' message passing network (i.e. node and edge update models)
        self.MPNet = self._build_core_MPNet(
            model_params=model_params, encoder_feats_dict=encoder_feats_dict
        )

        self.num_enc_steps = model_params["num_enc_steps"]
        self.num_class_steps = model_params["num_class_steps"]

    def _build_core_MPNet(self, model_params, encoder_feats_dict):
        """
        Builds the core part of the Message Passing Network: Node Update and Edge Update models.
        Args:
            model_params: dictionary contaning all model hyperparameters
            encoder_feats_dict: dictionary containing the hyperparameters for the initial node/edge encoder
        """

        # Define an aggregation operator for nodes to 'gather' messages from incident edges
        node_agg_fn = model_params["node_agg_fn"]
        assert node_agg_fn.lower() in (
            "mean",
            "max",
            "sum",
        ), "node_agg_fn can only be 'max', 'mean' or 'sum'."

        if node_agg_fn == "mean":
            node_agg_fn = lambda out, row, x_size: scatter_mean(
                out, row, dim=0, dim_size=x_size
            )

        elif node_agg_fn == "max":
            node_agg_fn = lambda out, row, x_size: scatter_max(
                out, row, dim=0, dim_size=x_size
            )[0]

        elif node_agg_fn == "sum":
            node_agg_fn = lambda out, row, x_size: scatter_add(
                out, row, dim=0, dim_size=x_size
            )

        # Define all MLPs involved in the graph network
        # For both nodes and edges, the initial encoded features (i.e. output of self.encoder) can either be
        # reattached or not after each Message Passing Step. This affects MLPs input dimensions
        self.reattach_initial_nodes = model_params["reattach_initial_nodes"]
        self.reattach_initial_edges = model_params["reattach_initial_edges"]

        edge_factor = 2 if self.reattach_initial_edges else 1
        node_factor = 2 if self.reattach_initial_nodes else 1

        edge_models_in_dim = {}
        for i, tpe1 in enumerate(self.node_types):
            for tpe2 in self.node_types[i:]:
                key = f"{tpe1}-{tpe2}"
                node_dim = node_factor * (
                    encoder_feats_dict["node_out_dims"][tpe1]
                    + encoder_feats_dict["node_out_dims"][tpe2]
                )
                edge_dim = edge_factor * encoder_feats_dict["edge_out_dims"][key]
                edge_models_in_dim[key] = node_dim + edge_dim

        node_models_in_dim = {}
        for i, tpe in enumerate(self.node_types):
            node_dim = node_factor * encoder_feats_dict["node_out_dims"][tpe]
            edge_dim = 0
            for tpe2 in self.node_types[:i]:
                key = f"{tpe2}-{tpe}"
                edge_dim += encoder_feats_dict["edge_out_dims"][key]
            for tpe2 in self.node_types[i:]:
                key = f"{tpe}-{tpe2}"
                edge_dim += encoder_feats_dict["edge_out_dims"][key]
            node_models_in_dim[tpe] = node_dim + edge_dim

        # Define all MLPs used within the MPN
        edge_model_feats_dict = model_params["edge_model_feats_dict"]
        node_model_feats_dict = model_params["node_model_feats_dict"]

        edge_models = {}
        for i, tpe1 in enumerate(self.node_types):
            for tpe2 in self.node_types[i:]:
                key = f"{tpe1}-{tpe2}"
                # The fc_dims are different for every type of node combination
                # while dropout and batchnorm are either used for all or none
                # of the MLPs
                if key in edge_model_feats_dict["fc_dims"]:
                    mlp = MLP(
                        input_dim=edge_models_in_dim[key],
                        fc_dims=edge_model_feats_dict["fc_dims"][key],
                        dropout_p=edge_model_feats_dict["dropout_p"],
                        use_batchnorm=edge_model_feats_dict["use_batchnorm"],
                    )
                    edge_model = EdgeModel(edge_mlp=mlp)
                    edge_models[key] = edge_model

        node_models = {}
        for tpe in self.node_types:
            node_type_config = node_model_feats_dict[tpe]

            message_modules = {}
            for tpe2 in self.node_types:
                if "time_aware" in node_type_config["message_modules"][tpe2]:
                    flow_in_mlp = MLP(
                        **node_type_config["message_modules"][tpe2]["time_aware"]
                    )
                    flow_out_mlp = MLP(
                        **node_type_config["message_modules"][tpe2]["time_aware"]
                    )
                    message_module = TimeAwareMessageModel(
                        flow_in_mlp, flow_out_mlp, node_agg_fn
                    )
                    message_modules[tpe2] = message_module
                else:
                    mlp = MLP(**node_type_config["message_modules"][tpe2])
                    # TODO: Enable use of time aware MP layer here
                    message_module = MessageModel(mlp, node_agg_fn)
                    message_modules[tpe2] = message_module

            update_module = MLP(**node_type_config["update_mlp"])

            node_type_model = NodeTypeModel(
                message_modules, update_module, tpe, self.node_types
            )
            node_models[tpe] = node_type_model

        # Define all MLPs used within the MPN
        return MetaLayer(
            edge_models=edge_models,
            node_type_models=node_models,
            node_types=self.node_types,
        )

    def forward(self, data):
        """
        Provides a fractional solution to the data association problem.
        First, node and edge features are independently encoded by the encoder network. Then, they are iteratively
        'combined' for a fixed number of steps via the Message Passing Network (self.MPNet). Finally, they are
        classified independently by the classifiernetwork.
        Args:
            data: object containing attribues
              - x: node features matrix for each node type
              - edge_index: for each edge type:
                tensor with shape [2, M], with M being the number of edges, indicating nonzero entries in the
                graph adjacency (i.e. edges) (i.e. sparse adjacency)
              - edge_attr: edge features matrix (sorted by edge apperance in edge_index)

        Returns:
            classified_edges: list of unnormalized node probabilites after each MP step
        """
        xs = {}
        edge_attrs = {}
        edge_indices = {}

        for i, tpe in enumerate(self.node_types):
            xs[tpe] = getattr(data, f"x_{tpe}")
            for tpe2 in self.node_types[i:]:
                key = f"{tpe}-{tpe2}"
                edge_attrs[key] = getattr(data, f"edge_attr_{key}")
                edge_indices[key] = getattr(data, f"edge_index_{key}")

        # Encoding features step
        latent_edge_feats, latent_node_feats = self.encoder(edge_attrs, xs)
        initial_edge_feats = {}
        for key in latent_edge_feats.keys():
            initial_edge_feats[key] = latent_edge_feats[key]
        initial_node_feats = {}
        for key in latent_node_feats.keys():
            initial_node_feats[key] = latent_node_feats[key]

        # During training, the feature vectors that the MPNetwork outputs for the  last self.num_class_steps message
        # passing steps are classified in order to compute the loss.
        first_class_step = self.num_enc_steps - self.num_class_steps + 1
        outputs_dict = {"classified_edges": []}
        for step in range(1, self.num_enc_steps + 1):

            # Reattach the initially encoded embeddings before the update
            if self.reattach_initial_edges:
                for key in initial_edge_feats.keys():
                    latent_edge_feats[key] = torch.cat(
                        (initial_edge_feats[key], latent_edge_feats[key]), dim=1
                    )
            if self.reattach_initial_nodes:
                for key in initial_node_feats.keys():
                    latent_node_feats[key] = torch.cat(
                        (initial_node_feats[key], latent_node_feats[key]), dim=1
                    )

            # Message Passing Step
            latent_node_feats, latent_edge_feats = self.MPNet(
                latent_node_feats, edge_indices, latent_edge_feats
            )

            if step >= first_class_step:
                # Classification Step
                dec_edge_feats, _ = self.classifier(latent_edge_feats)
                outputs_dict["classified_edges"].append(dec_edge_feats)

        if self.num_enc_steps == 0:
            dec_edge_feats, _ = self.classifier(latent_edge_feats)
            outputs_dict["classified_edges"].append(dec_edge_feats)

        return outputs_dict
