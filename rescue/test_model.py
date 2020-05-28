import torch
from inmemory_rescue_dataset import InMemoryRescueDataset
from torch_geometric.data.dataloader import DataLoader
from models.model import Model


def test_model(data_loader, model, node_classification, device):
    model.eval()
    correct = 0
    total_graph = 0
    for data in data_loader:
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
        total_graph += data.num_graphs
    return correct / total_graph


def test_model_building_on_fire_selection(data_loader, model, node_classification, device):
    model.eval()
    on_fire_count = 0
    correct = 0
    total_graph = 0
    for data in data_loader:
        data = data.to(device)
        output = model(data)
        if node_classification:
            pass
        else:
            pred = output.max(dim=1)[1]
            # print("pred:{}".format(pred))
            # print("true:{}".format(data.y))
            buildings_on_fire = torch.nonzero(data.x[:, 5]).view(-1)
            # print("buildings_on_fire:{}".format(buildings_on_fire))
            # print("in_the_list:{}".format((pred in buildings_on_fire)))
            if pred in buildings_on_fire:
                on_fire_count += 1
            correct += pred.eq(data.y).sum().item()
            total_graph += data.num_graphs
    return correct / total_graph, on_fire_count / total_graph


if __name__ == '__main__':
    batch_size = 1
    node_classification = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_dataset = InMemoryRescueDataset([], node_classification=node_classification)
    test_dataset.load('test_dataset.pt')
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    model = Model.load("models/topk_test_gat.pth")
    model = model.to(device)
    accuracy, on_fire_accuracy = test_model_building_on_fire_selection(test_dataloader, model, node_classification, device)
    print("Accuracy:{} On Fire Accuracy:{}".format(accuracy, on_fire_accuracy))