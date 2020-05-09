import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F


class GCNNet(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GCNNet, self).__init__()
        self.bn1 = torch.nn.BatchNorm1d(num_features=num_features)
        self.conv1 = GCNConv(num_features, 128, cached=False, normalize=True)
        self.bn2 = torch.nn.BatchNorm1d(num_features=128)
        self.conv2 = GCNConv(128, 128, cached=False, normalize=True)
        self.bn3 = torch.nn.BatchNorm1d(num_features=128)
        self.conv3 = GCNConv(128, 64, cached=False, normalize=True)
        self.bn4 = torch.nn.BatchNorm1d(num_features=64)
        self.conv4 = GCNConv(64, 32, cached=False, normalize=True)

        self.bn5 = torch.nn.BatchNorm1d(num_features=32)
        self.conv5 = GCNConv(32, 64, cached=False, normalize=True)
        self.bn6 = torch.nn.BatchNorm1d(num_features=64)
        self.conv6 = GCNConv(64, 128, cached=False, normalize=True)
        self.bn7 = torch.nn.BatchNorm1d(num_features=128)
        self.conv7 = GCNConv(128, 64, cached=False, normalize=True)
        self.bn8 = torch.nn.BatchNorm1d(num_features=64)
        self.conv8 = GCNConv(64, 32, cached=False, normalize=True)
        self.p = 0.5

        self.bn9 = torch.nn.BatchNorm1d(num_features=32)
        self.lin1 = torch.nn.Linear(32, 64)
        self.bn10 = torch.nn.BatchNorm1d(num_features=64)
        self.lin2 = torch.nn.Linear(64, num_classes)

        # self.reg_params = self.conv1.parameters()
        # self.non_reg_params = self.conv4.parameters()

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = self.bn1(x)
        x = F.relu(self.bn2(self.conv1(x, edge_index, edge_weight)))
        x = F.dropout(x, p=self.p, training=self.training)
        x = F.relu(self.bn3(self.conv2(x, edge_index, edge_weight)))
        x = F.dropout(x, p=self.p, training=self.training)
        x = F.relu(self.bn4(self.conv3(x, edge_index, edge_weight)))
        x = F.dropout(x, p=self.p, training=self.training)
        x = F.relu(self.bn5(self.conv4(x, edge_index, edge_weight)))
        x = F.dropout(x, p=self.p, training=self.training)
        x = F.relu(self.bn6(self.conv5(x, edge_index, edge_weight)))
        x = F.dropout(x, p=self.p, training=self.training)
        x = F.relu(self.bn7(self.conv6(x, edge_index, edge_weight)))
        x = F.dropout(x, p=self.p, training=self.training)
        x = F.relu(self.bn8(self.conv7(x, edge_index, edge_weight)))
        x = self.bn9(self.conv8(x, edge_index, edge_weight))
        x = self.lin1(F.relu(x))
        x = self.bn10(x)
        x = self.lin2(F.relu(x))
        return x
        # return F.log_softmax(x, dim=1)