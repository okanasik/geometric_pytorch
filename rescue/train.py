import torch
import torch.nn.functional as F
from torch_geometric.data.dataloader import DataLoader
from rescue_dataset import RescueDataset
from inmemory_rescue_dataset import InMemoryRescueDataset
from rescue_dataset_list import RescueDatasetList
from models.topk_model import TopKNet
import test_model

# parameters
node_classification = False
batch_size = 256


def get_datasets():
    # train dataset
    train_datasets = []
    for i in range(50,300):
        dataset = RescueDataset("/home/okan/rescuesim/rcrs-server/dataset", "firebrigade", comp="robocup2019",
                                   scenario="test"+str(i),
                                team="ait", node_classification=node_classification, read_info_map=True)
        train_datasets.append(dataset)
    train_dataset_list = RescueDatasetList(train_datasets)


    # test dataset
    test_datasets = []
    for i in range(300, 400):
        dataset = RescueDataset("/home/okan/rescuesim/rcrs-server/dataset", "firebrigade", comp="robocup2019",
                                scenario="test" + str(i),
                                team="ait", node_classification=node_classification, read_info_map=True)
        test_datasets.append(dataset)
    test_dataset_list = RescueDatasetList(test_datasets)

    print('Train dataset len:{}'.format(len(train_dataset_list)))
    print('Test dataset len:{}'.format(len(test_dataset_list)))

    return train_dataset_list, test_dataset_list


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
    train_dataset.load('train_dataset.pt')

    test_dataset = InMemoryRescueDataset([], node_classification=node_classification)
    test_dataset.load('test_dataset.pt')
    print(len(train_dataset))
    print(len(test_dataset))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    if node_classification:
        testtrain_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    else:
        testtrain_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model = GCNNet(dataset.num_features, dataset.num_classes)
    model = TopKNet(train_dataset.num_features, train_dataset.num_classes)
    model.load_state_dict(torch.load('./topk_test_gat.pth'))
    # model = AGNNNet(dataset.num_features, dataset.num_classes)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
    for epoch in range(0, 1001):
        loss = train()
        if epoch % 10 == 0:
            train_accuracy = test_model.test_model(testtrain_loader, model, node_classification, device)
            test_accuracy = test_model.test_model(test_loader, model, node_classification, device)
            log = 'Epoch: {:03d}, Train Loss: {:.8f} Train Accuracy: {:.8f} Test Accuracy: {:.8f}'
            print(log.format(epoch, loss, train_accuracy, test_accuracy))
            print("Saving model topk")
            torch.save(model.state_dict(), './topk_test_gat.pth')
            print('Model: ' +  './topk_test_gat.pth' + ' is saved.')

# accuracy = test()
# log = 'Epoch: {:03d}, Train Loss: {:.8f} Accuracy: {:.8f}'
# print(log.format(0, 0, accuracy))

# torch.save(model.state_dict(), './topk_model_test4.pth')