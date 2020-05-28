from rescue.models.model import Model
import torch
from torch_geometric.nn import TopKPooling, GATConv
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp


class TestModel(Model):
    def __init__(self, num_features, num_classes):
        super(TestModel, self).__init__()
        self.name = "test_model"
        self.version = "v1"
        self.num_features = num_features
        self.num_classes = num_classes

        self.bn1 = torch.nn.BatchNorm1d(num_features=num_features)

        self.conv1 = GATConv(num_features, 128)
        self.bn2 = torch.nn.BatchNorm1d(num_features=128)
        self.pool1 = TopKPooling(128, ratio=0.8)

        self.lin1 = torch.nn.Linear(256, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.bn1(x)
        x = F.relu(self.bn2(self.conv1(x, edge_index)))
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = self.lin1(x)
        return x





