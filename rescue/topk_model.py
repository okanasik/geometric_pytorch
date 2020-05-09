import torch
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, TopKPooling, GATConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp


class TopKNet(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(TopKNet, self).__init__()
        self.bn1 = torch.nn.BatchNorm1d(num_features=num_features)

        self.conv1 = GATConv(num_features, 256)
        self.bn2 = torch.nn.BatchNorm1d(num_features=256)
        self.pool1 = TopKPooling(256, ratio=0.8)

        self.conv2 = GATConv(256, 256)
        self.bn3 = torch.nn.BatchNorm1d(num_features=256)
        self.pool2 = TopKPooling(256, ratio=0.8)

        self.conv3 = GATConv(256, 256)
        self.bn4 = torch.nn.BatchNorm1d(num_features=256)
        self.pool3 = TopKPooling(256, ratio=0.8)

        self.lin1 = torch.nn.Linear(512, 256)
        self.bn5 = torch.nn.BatchNorm1d(num_features=256)
        self.lin2 = torch.nn.Linear(256, 128)
        self.bn6 = torch.nn.BatchNorm1d(num_features=128)
        self.lin3 = torch.nn.Linear(128, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.bn1(x)
        x = F.relu(self.bn2(self.conv1(x, edge_index)))
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.bn3(self.conv2(x, edge_index)))
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.bn4(self.conv3(x, edge_index)))
        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3

        x = F.relu(self.bn5(self.lin1(x)))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.bn6(self.lin2(x)))
        x = self.lin3(x)
        return x
