import torch
from dataset.inmemory_rescue_dataset import InMemoryRescueDataset
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


def convert_and_save_model():
    test_dataset = InMemoryRescueDataset([], node_classification=False)
    test_dataset.load('dataset/test_notnull_notrandom_dataset.pt')

    from rescue.models.topk_model import TopKNet
    topk_model = TopKNet(test_dataset.num_features, test_dataset.num_classes)

    model_filename = "topk_gat_test_notnull_notrandom.pt"
    topk_model.load_state_dict(torch.load(model_filename))
    topk_model.mysave("models/" + model_filename)

if __name__ == '__main__':
    batch_size = 1
    node_classification = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = InMemoryRescueDataset([], node_classification=node_classification)
    train_dataset.load('dataset/train_notnull_notrandom_dataset.pt')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)

    test_dataset = InMemoryRescueDataset([], node_classification=node_classification)
    test_dataset.load('dataset/test_notnull_notrandom_dataset.pt')
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    model = Model.load("models/topk_gat_test_notnull_notrandom.pt")
    model = model.to(device)
    accuracy, on_fire_accuracy = test_model_building_on_fire_selection(test_dataloader, model, node_classification, device)
    print("Accuracy:{} On Fire Accuracy:{}".format(accuracy, on_fire_accuracy))

    # accuracy = test_model(train_dataloader, model, node_classification, device)
    # print("Train Dataset Accuracy:{}".format(accuracy))
    #
    # accuracy = test_model(test_dataloader, model, node_classification, device)
    # print("Test Dataset Accuracy:{}".format(accuracy))

    # convert_and_save_model()