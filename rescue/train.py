import torch
import torch.nn.functional as F
from torch_geometric.data.dataloader import DataLoader
from rescue_dataset import RescueDataset
from gcn_model import GCNNet
from topk_model import TopKNet

node_classification = False
dataset = RescueDataset("/home/okan/rescuesim/rcrs-server/dataset", "firebrigade", comp="robocup2019", scenario="test",
                        team="ait", node_classification=node_classification)
train_loader = DataLoader(dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(dataset, batch_size=128, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = GCNNet(dataset.num_features, dataset.num_classes).to(device)
model = TopKNet(dataset.num_features, dataset.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)


def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        data = data.to(device)
        output = model(data)
        loss = F.nll_loss(output, data.y)
        loss.backward()
        total_loss += loss.item()
        optimizer.step()
    return total_loss / len(dataset)


def test():
    model.eval()
    correct = 0
    for data in test_loader:
        data = data.to(device)
        output = model(data)
        pred = output.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(dataset)


for epoch in range(1, 501):
    loss = train()
    accuracy = test()
    log = 'Epoch: {:03d}, Train Loss: {:.4f} Accuracy: {:.4f}'
    print(log.format(epoch, loss, accuracy))
