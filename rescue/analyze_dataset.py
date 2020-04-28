from rescue_dataset import RescueDataset
import os.path


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
                            scenario="test2", team="ait", node_classification=False)
    mem_model = memorization_model(dataset)
    test_model(dataset, mem_model)


def test_random_target_selection():
    dataset = RescueDataset("/home/okan/rescuesim/rcrs-server/dataset", "firebrigade", comp="robocup2019",
                            scenario="test2", team="ait", node_classification=False, read_info_map=True)
    num_random = 0
    for data_item in dataset:
        if 'type' in data_item.info_map:
            if data_item.info_map['type'] == 'random':
                num_random += 1

    print("{:.2f}% of selection is random".format(100*(num_random/len(dataset))))


def test_fieryness_target_selection():
    dataset = RescueDataset("/home/okan/rescuesim/rcrs-server/dataset", "firebrigade", comp="robocup2019",
                            scenario="test2", team="ait", node_classification=False)

    num_in_fieryness = 0
    for data_item in dataset:
        nodes_with_fieryness = data_item.x[:,5].nonzero().view(-1)
        print(nodes_with_fieryness)
        print(data_item.y.item())
        if data_item.y.item() in nodes_with_fieryness:
            num_in_fieryness += 1
    print("{:.2f}% of targets from buildings with fieryness".format(100 * (num_in_fieryness / len(dataset))))


def print_raw_fieryness_and_target():
    dataset = RescueDataset("/home/okan/rescuesim/rcrs-server/dataset", "firebrigade", comp="robocup2019",
                            scenario="test2", team="ait", node_classification=False)
    num_in_fieryness = 0
    burning_fieryness_values = {1, 2, 3}
    for raw_file_name in dataset.raw_file_names:
        full_filename = os.path.join(dataset.root, raw_file_name)
        json_data = RescueDataset.read_raw_json_file(full_filename)
        buildings_on_fire = set()
        for frame in json_data['frames']:
            if frame['time'] < 4:
                continue
            nodes = frame['change']['nodes']
            for node_id_str in nodes:
                if 'fieryness' in nodes[node_id_str]:
                    if nodes[node_id_str]['fieryness'] in burning_fieryness_values:
                        buildings_on_fire.add(nodes[node_id_str]['id'])
                    else:
                        if nodes[node_id_str]['id'] in buildings_on_fire: buildings_on_fire.remove(nodes[node_id_str]['id'])

            if frame['action']['targetId'] in buildings_on_fire:
                num_in_fieryness += 1
            print(buildings_on_fire)
            print("target:" + str(frame['action']['targetId']))

    print("{:.2f}% of targets from buildings with fieryness".format(100 * (num_in_fieryness / len(dataset))))


if __name__ == "__main__":
    # test_memorization_model()
    # test_random_target_selection()
    # test_fieryness_target_selection()
    print_raw_fieryness_and_target()