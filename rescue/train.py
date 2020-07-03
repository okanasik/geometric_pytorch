import torch
import torch.nn.functional as F
from torch_geometric.data.dataloader import DataLoader
from dataset.rescue_dataset import RescueDataset
from dataset.inmemory_rescue_dataset import InMemoryRescueDataset
from dataset.rescue_dataset_list import RescueDatasetList
from models.topk_model import TopKNet
import test_model

# parameters
node_classification = False
batch_size = 256
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train():
    model.train()
    total_loss = 0
    total_graph = 0
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
        total_graph += data.num_graphs
    return total_loss / total_graph


if __name__ == '__main__':
    # import analyze_dataset
    # train_dataset, test_dataset = get_datasets()
    # train_dataset, test_dataset = analyze_dataset.get_notrandom_notnull(train_dataset, test_dataset,
    #                                                                     node_classification=node_classification)
    train_dataset = InMemoryRescueDataset([], node_classification=node_classification)
    train_dataset.load('dataset/train_notnull_notrandom_dataset.pt', device=device)

    test_dataset = InMemoryRescueDataset([], node_classification=node_classification)
    test_dataset.load('dataset/test_notnull_notrandom_dataset.pt', device=device)
    print(len(train_dataset))
    print(len(test_dataset))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    if node_classification:
        testtrain_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    else:
        testtrain_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    model_filename = "topk_gat_test_notnull_notrandom.pt"
    # model = GCNNet(dataset.num_features, dataset.num_classes)
    model = TopKNet(train_dataset.num_features, train_dataset.num_classes)
    model.load_state_dict(torch.load(model_filename))
    # model = AGNNNet(dataset.num_features, dataset.num_classes)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
    for epoch in range(190, 1001):
        loss = train()
        if epoch % 10 == 0:
            train_accuracy = test_model.test_model(testtrain_loader, model, node_classification, device)
            test_accuracy = test_model.test_model(test_loader, model, node_classification, device)
            log = 'Epoch: {:03d}, Train Loss: {:.8f} Train Accuracy: {:.8f} Test Accuracy: {:.8f}'
            print(log.format(epoch, loss, train_accuracy, test_accuracy))
            print("Saving model topk")
            torch.save(model.state_dict(), model_filename)
            print('Model: ' +  model_filename + ' is saved.')