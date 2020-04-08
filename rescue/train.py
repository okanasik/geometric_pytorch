import torch
import torch.nn.functional as F
from torch_geometric.data.dataloader import DataLoader
from rescue_dataset import RescueDataset
from gcn_model import GCNNet

dataset = RescueDataset("/home/okan/rescuesim/rcrs-server/dataset", "firebrigade", "robocup2019", "test", "ait")
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset, batch_size=1, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCNNet(dataset.num_features, dataset.num_classes).to(device)
optimizer = torch.optim.Adam([
    dict(params=model.reg_params, weight_decay=5e-4),
    dict(params=model.non_reg_params, weight_decay=0)
], lr=0.01)


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
        pred = output.max(1)[1]
        # print("pred:{}".format(pred))
        # print("true:{}".format(data.y))
        y_index = torch.argmax(data.y)
        # print("pred:{} true:{}".format(pred, y_index))
        pred_index = torch.argmax(pred)
        correct += int(y_index == pred_index)
    return float(correct) / len(dataset)


for epoch in range(1, 201):
    loss = train()
    accuracy = test()
    log = 'Epoch: {:03d}, Train Loss: {:.4f} Accuracy: {:.4f}'
    print(log.format(epoch, loss, accuracy))
