import torch
import torch.nn.functional as F
from torch_geometric.data.dataloader import DataLoader
from rescue_dataset import RescueDataset
from topk_model import TopKNet
from gcn_model import GCNNet
from agnn_model import AGNNNet
from soft_assignment_loss import soft_assignment_loss

# parameters
node_classification = False
batch_size = 64

dataset = RescueDataset("/home/okan/rescuesim/rcrs-server/dataset", "firebrigade", comp="robocup2019", scenario="test2",
                        team="ait", node_classification=node_classification)

train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
if node_classification:
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False)
else:
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = GCNNet(dataset.num_features, dataset.num_classes)
# optimizer = torch.optim.Adam([
#     dict(params=model.reg_params, weight_decay=5e-4),
#     dict(params=model.non_reg_params, weight_decay=0)
# ], lr=0.01)

model = TopKNet(dataset.num_features, dataset.num_classes)
# model.load_state_dict(torch.load('./topk_model_test4.pth'))
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

# model = AGNNNet(dataset.num_features, dataset.num_classes).to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        data = data.to(device)
        output = model(data)
        if node_classification:
            class_weight = torch.tensor([data.num_graphs/data.num_nodes, (data.num_nodes-data.num_graphs)/data.num_nodes], device=device)
            loss = F.cross_entropy(output, data.y, weight=class_weight)
            # loss = soft_assignment_loss(output, data, device)
        else:
            loss = F.cross_entropy(output, data.y)
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
        if node_classification:
            pred = output.max(1)[1]
            pred_index = torch.argmax(pred)
            y_index = torch.argmax(data.y)
            correct += int(y_index == pred_index)
        else:
            pred = output.max(dim=1)[1]
            correct += pred.eq(data.y).sum().item()
    return correct / len(dataset)


for epoch in range(1, 1001):
    loss = train()
    accuracy = test()
    log = 'Epoch: {:03d}, Train Loss: {:.8f} Train Accuracy: {:.8f}'
    print(log.format(epoch, loss, accuracy))

# accuracy = test()
# log = 'Epoch: {:03d}, Train Loss: {:.8f} Accuracy: {:.8f}'
# print(log.format(0, 0, accuracy))

# torch.save(model.state_dict(), './topk_model_test4.pth')