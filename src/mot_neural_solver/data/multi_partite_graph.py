from typing import Dict, List

import torch
from mot_neural_solver.data.mot_graph import Graph, MOTGraph
from mot_neural_solver.models.combined.types import EdgeType, NodeType

_data_attr_names = [
    "edge_attr",
    "edge_index",
    "x",
    "node_names",
    "edge_labels",
    "edge_preds",
    "reid_emb_dists",
]


class MultiGraph(Graph):
    def __init__(
        self,
        node_types: List[str],
        edge_attrs: Dict[EdgeType, torch.Tensor],
        edge_indices: Dict[EdgeType, torch.Tensor],
        xs: Dict[NodeType, torch.Tensor],
        **kwargs,
    ):
        super().__init__(**kwargs)

        for key in edge_indices.keys():
            setattr(self, f"edge_index_{key}", edge_indices[key])
        for key in edge_attrs.keys():
            setattr(self, f"edge_attr_{key}", edge_attrs[key])
        for key in xs.keys():
            setattr(self, f"x_{key}", xs[key])

        self.__node_types = node_types
        self.__edge_types = []
        for i, tpe1 in enumerate(self.__node_types):
            for tpe2 in self.__node_types[i:]:
                self.__edge_types.append(f"{tpe1}-{tpe2}")

        self.__data_attr_names = _data_attr_names
        for node_type in self.__node_types:
            self.__data_attr_names.append(f"x_{node_type}")
        for edge_type in self.__edge_types:
            self.__data_attr_names.append(f"edge_attr_{edge_type}")
            self.__data_attr_names.append(f"edge_index_{edge_type}")

    def _change_attrs_types(self, attr_change_fn):
        """
        Base method for all methods related to changing attribute types. Iterates over the attributes names in
        _data_attr_names, and changes its type via attr_change_fun

        Args:
            attr_change_fn: callable function to change a variable's type
        """

        for attr_name in self.__data_attr_names:
            if hasattr(self, attr_name):
                if getattr(self, attr_name) is not None:
                    old_attr_val = getattr(self, attr_name)
                    setattr(self, attr_name, attr_change_fn(old_attr_val))

    def __inc__(self, key: str, value):
        # Adopted from https://github.com/rusty1s/pytorch_geometric/issues/1210
        if key.startswith("edge_index_"):
            edge_type = key[11:].split("-")
            tpe1 = edge_type[0]
            tpe2 = edge_type[1]

            nodes_tpe1 = getattr(self, f"x_{tpe1}")
            nodes_tpe2 = getattr(self, f"x_{tpe2}")

            return torch.tensor([[nodes_tpe1.size(0)], [nodes_tpe2.size(0)]])
        else:
            return super(MultiGraph, self).__inc__(key, value)


class MultiMOTGraph(MOTGraph):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def construct_graph_object(self):
        super().construct_graph_object()

        graph_obj = self.graph_obj

        joints = self._get_joints().float()

        edge_attrs = {}
        edge_indices = {}
        xs = {}

        edge_attrs["bb-bb"] = graph_obj.edge_attr
        edge_indices["bb-bb"] = graph_obj.edge_index
        xs["bb"] = graph_obj.x

        xs["joint"] = joints.reshape((-1, 3))

        num_bb = xs["bb"].shape[0]
        num_joints_per_bb = joints.shape[1]

        joint_bb_edge_ix, joint_bb_edge_attr = self.get_bb_joint_edges(
            num_bb, num_joints_per_bb
        )

        edge_attrs["bb-joint"] = joint_bb_edge_attr
        edge_indices["bb-joint"] = joint_bb_edge_ix

        joint_joint_edge_ix, joint_joint_edge_attr = self.get_joint_joint_edges(
            num_bb, num_joints_per_bb, graph_obj.edge_index.cpu()
        )

        edge_attrs["joint-joint"] = joint_joint_edge_attr
        edge_indices["joint-joint"] = joint_joint_edge_ix

        # TODO: Test if we can get rid of edge_attr, edge_index, x, reid_emb_dists?
        # They are necessary right now so the tracking in the beginning works correctly
        reid_emb_dists = None
        if hasattr(graph_obj, "reid_emb_dists"):
            reid_emb_dists = graph_obj.reid_emb_dists
        self.graph_obj = MultiGraph(
            edge_attrs=edge_attrs,
            edge_indices=edge_indices,
            xs=xs,
            node_types=["bb", "joint"],
            edge_attr=graph_obj.edge_attr,
            edge_index=graph_obj.edge_index,
            x=graph_obj.x,
            reid_emb_dists=reid_emb_dists,
        )

        self.graph_obj.to(
            torch.device(
                "cuda" if torch.cuda.is_available() and self.inference_mode else "cpu"
            )
        )

    def get_bb_joint_edges(self, num_bb: int, num_joints_per_bb: int):
        joint_bb_edge_ix = torch.zeros(
            (2, num_bb * num_joints_per_bb), dtype=torch.long
        )
        # TODO: Add option to add edge attributes for bb-joint edges
        joint_bb_edge_attr = torch.zeros((num_bb * num_joints_per_bb, 0))

        joint_bb_edge_ix[1] = torch.arange(0, num_bb * num_joints_per_bb)
        joint_bb_edge_ix[0] = joint_bb_edge_ix[1] // num_joints_per_bb

        return joint_bb_edge_ix, joint_bb_edge_attr

    def get_joint_joint_edges(
        self, num_bb: int, num_joints_per_bb: int, bb_edge_indices: torch.Tensor
    ):
        # TODO: Debug this shit
        num_bb_edges = bb_edge_indices.shape[1]

        joint_joint_edge_ix = torch.zeros(
            (2, num_bb_edges * num_joints_per_bb), dtype=torch.long
        )

        joint_ids = (
            torch.arange(0, num_bb_edges * num_joints_per_bb) % num_joints_per_bb
        )

        bb_ids_1 = bb_edge_indices[0].reshape((-1, 1)).repeat((1, num_joints_per_bb))
        bb_ids_2 = bb_edge_indices[1].reshape((-1, 1)).repeat((1, num_joints_per_bb))

        joint_joint_edge_ix[0] = bb_ids_1.reshape((-1)) * num_joints_per_bb + joint_ids
        joint_joint_edge_ix[1] = bb_ids_2.reshape((-1)) * num_joints_per_bb + joint_ids

        # TODO: Add option to add edge attributes for joint-joint edges
        joint_joint_edge_attr = torch.zeros((num_bb_edges * num_joints_per_bb, 0))

        return joint_joint_edge_ix, joint_joint_edge_attr
