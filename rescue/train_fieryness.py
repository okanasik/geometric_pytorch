import torch
import torch.nn.functional as F
from torch_geometric.data.dataloader import DataLoader
from rescue_dataset import RescueDataset
from topk_model import TopKNet
from gcn_model import GCNNet
from agnn_model import AGNNNet
from soft_assignment_loss import soft_assignment_loss

# parameters
node_classification = True
batch_size = 1

dataset = RescueDataset("/home/okan/rescuesim/rcrs-server/dataset", "firebrigade", comp="robocup2019", scenario="test2",
                        team="ait", node_classification=node_classification)

test_dataset = RescueDataset("/home/okan/rescuesim/rcrs-server/dataset", "firebrigade", comp="robocup2019", scenario="test3",
                        team="ait", node_classification=node_classification)

train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCNNet(dataset.num_features, dataset.num_classes)
# model = AGNNNet(dataset.num_features, dataset.num_classes).to(device)

model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)


def get_fieryness_target(fieryness_values, size, device):
    nodes_with_fieryness = [idx for idx, fieryness in enumerate(fieryness_values) if
                            fieryness >= 1 and fieryness <= 3]
    num_fieryness = len(nodes_with_fieryness)
    target = torch.zeros(size, device=device, dtype=torch.long)
    target[nodes_with_fieryness] = 1
    return target, num_fieryness

def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        data = data.to(device)
        output = model(data)
        target, num_fieryness = get_fieryness_target(data.x[:, 5].view(-1), data.y.size(), device)
        class_weight = torch.tensor([num_fieryness/data.num_nodes, (data.num_nodes-num_fieryness)/data.num_nodes], dtype=torch.float32, device=device)
        loss = F.cross_entropy(output, target, weight=class_weight)
        # loss = F.cross_entropy(output, target)
        loss.backward()
        total_loss += loss.item()
        optimizer.step()
    return total_loss / len(dataset)


def test():
    model.eval()
    correct = 0
    total = 0
    total_fieryness = 0
    for data in test_loader:
        data = data.to(device)
        output = model(data)
        pred = output.max(dim=1)[1]
        target, num_fieryness = get_fieryness_target(data.x[:, 5].view(-1), data.y.size(), device)
        this_correct = pred.eq(target).sum().item()
        correct += this_correct
        total += data.num_nodes
        total_fieryness += num_fieryness
    return correct / total, total_fieryness / total


for epoch in range(1, 1001):
    loss = train()
    accuracy, fieryness_ratio = test()
    log = 'Epoch: {:03d}, Train Loss: {:.8f} Test Accuracy: {:.8f} Fieryness Ratio: {}'
    print(log.format(epoch, loss, accuracy, fieryness_ratio))
    if epoch % 100 == 0:
        model_filename = 'gcn_fieryness.pth'
        torch.save(model.state_dict(), model_filename)
        print("Model is saved as " + model_filename)

# accuracy = test()
# log = 'Epoch: {:03d}, Train Loss: {:.8f} Accuracy: {:.8f}'
# print(log.format(0, 0, accuracy))