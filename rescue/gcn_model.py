import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F


class GCNNet(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GCNNet, self).__init__()
        self.conv1 = GCNConv(num_features, 128, cached=False, normalize=True)
        self.conv2 = GCNConv(128, 128, cached=False, normalize=True)
        self.conv3 = GCNConv(128, 64, cached=False, normalize=True)
        self.conv4 = GCNConv(64, num_classes, cached=False, normalize=True)
        self.p1 = 0.2
        self.p2 = 0.2
        self.p3 = 0.2

        # self.reg_params = self.conv1.parameters()
        # self.non_reg_params = self.conv4.parameters()

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, p=self.p1, training=self.training)
        x = F.relu(self.conv2(x, edge_index, edge_weight))
        x = F.dropout(x, p=self.p2, training=self.training)
        x = F.relu(self.conv3(x, edge_index, edge_weight))
        x = F.dropout(x, p=self.p3, training=self.training)
        x = self.conv4(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)