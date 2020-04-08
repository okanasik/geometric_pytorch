import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F


class GCNNet(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GCNNet, self).__init__()
        self.conv1 = GCNConv(num_features, 16, cached=False, normalize=True)
        self.conv2 = GCNConv(16, 16, cached=False, normalize=True)

        self.reg_params = self.conv1.parameters()
        self.non_reg_params = self.conv2.parameters()

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)