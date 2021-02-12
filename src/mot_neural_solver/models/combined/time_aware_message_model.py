import torch
import torch.nn as nn


class TimeAwareMessageModel(nn.Module):
    """
    Class used to peform the node update during Neural mwssage passing
    """

    def __init__(self, flow_in_mlp, flow_out_mlp, node_agg_fn):
        super(TimeAwareMessageModel, self).__init__()

        self.flow_in_mlp = flow_in_mlp
        self.flow_out_mlp = flow_out_mlp
        self.node_agg_fn = node_agg_fn

    def forward(self, _, x, edge_index, edge_attr):
        row, col = edge_index
        flow_out_mask = row < col
        flow_out_row, flow_out_col = row[flow_out_mask], col[flow_out_mask]
        flow_out_input = torch.cat([x[flow_out_col], edge_attr[flow_out_mask]], dim=1)
        flow_out = self.flow_out_mlp(flow_out_input)
        flow_out = self.node_agg_fn(flow_out, flow_out_row, x.size(0))

        flow_in_mask = row > col
        flow_in_row, flow_in_col = row[flow_in_mask], col[flow_in_mask]
        flow_in_input = torch.cat([x[flow_in_col], edge_attr[flow_in_mask]], dim=1)
        flow_in = self.flow_in_mlp(flow_in_input)

        flow_in = self.node_agg_fn(flow_in, flow_in_row, x.size(0))
        flow = torch.cat((flow_in, flow_out), dim=1)

        return flow
