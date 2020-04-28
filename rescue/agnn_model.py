import torch
import torch.nn.functional as F
from torch_geometric.nn import AGNNConv


class AGNNNet(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(AGNNNet, self).__init__()
        self.lin1 = torch.nn.Linear(num_features, 64)
        self.conv1 = AGNNConv(requires_grad=True)
        self.conv2 = AGNNConv(requires_grad=True)
        self.lin2 = torch.nn.Linear(64, num_classes)

    def forward(self, data):
        x = F.dropout(data.x, training=self.training)
        x = F.relu(self.lin1(x))
        x = self.conv1(x, data.edge_index)
        x = self.conv2(x, data.edge_index)
        x = F.dropout(x, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)