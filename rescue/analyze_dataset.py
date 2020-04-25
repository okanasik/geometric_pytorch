from rescue_dataset import RescueDataset


def create_graph_str(data):
    x = data.x.view((-1,))
    return "".join([str(int(v.item())) for v in x])


def memorization_model(dataset):
    memory_model = {}
    for data_item in dataset:
        data_str = create_graph_str(data_item)
        memory_model[data_str] = data_item.y.item()
    return memory_model


def test_model(dataset, mem_model):
    accuracy = 0
    for data_item in dataset:
        data_str = create_graph_str(data_item)
        if mem_model[data_str] == data_item.y.item():
            accuracy += 1

    print("Accuracy of memory model is {}".format((accuracy/len(dataset))))


def test_memorization_model():
    dataset = RescueDataset("/home/okan/rescuesim/rcrs-server/dataset", "firebrigade", comp="robocup2019",
                            scenario="test4", team="ait", node_classification=False)
    mem_model = memorization_model(dataset)
    test_model(dataset, mem_model)


if __name__ == "__main__":
    test_memorization_model()
