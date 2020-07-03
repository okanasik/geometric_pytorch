from dataset.rescue_dataset import RescueDataset
from dataset.inmemory_rescue_dataset import InMemoryRescueDataset
import os.path
import rescue.dataset.dataset_manager as dataset_manager


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
    # dataset = RescueDataset("/home/okan/rescuesim/rcrs-server/dataset", "firebrigade", comp="robocup2019",
    #                         scenario="test2", team="ait", node_classification=False, read_info_map=True)
    import train
    train_dataset, test_dataset = train.get_datasets()

    num_random = 0
    for data_item in train_dataset:
        if 'type' in data_item.info_map:
            if data_item.info_map['type'] == 'random':
                num_random += 1

    print("{:.2f}% of selection is random".format(100*(num_random/len(train_dataset))))


def test_fieryness_target_selection():
    dataset = RescueDataset("/home/okan/rescuesim/rcrs-server/dataset", "firebrigade", comp="robocup2019",
                            scenario="test2", team="ait", node_classification=False)

    num_in_fieryness = 0
    for data_item in dataset:
        fieryness_values = data_item.x[:,5].view(-1)
        nodes_with_fieryness = [idx for idx, fieryness in enumerate(fieryness_values) if fieryness >= 1 and fieryness <= 3]
        # print(nodes_with_fieryness)
        # print(data_item.y.item())
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


def test_null_target_selection():
    import train
    train_dataset, test_dataset = train.get_datasets()
    count_null = 0
    for idx, data_item in enumerate(train_dataset):
        # print("graph_{}:{}".format(idx, data_item.y.item()))
        if data_item.y.item() == 0:
            count_null += 1

    print("{:.2f}% of targets are null".format(100 * (count_null / len(train_dataset))))


def get_notrandom_notnull(train_dataset, test_dataset, node_classification):
    from dataset.inmemory_rescue_dataset import InMemoryRescueDataset

    data_list = []
    for data_item in train_dataset:
        if data_item.y.item() == 0:
            continue
        if data_item.info_map['type'] == 'random':
            continue

        data_list.append(data_item)

    for data_item in test_dataset:
        if data_item.y.item() == 0:
            continue
        if data_item.info_map['type'] == 'random':
            continue

        data_list.append(data_item)

    import random
    random.shuffle(data_list)

    train_data_list = data_list[:int(0.75*len(data_list))]
    print(len(train_data_list))
    test_data_list = data_list[int(0.75*len(data_list)):]
    print(len(test_data_list))

    return InMemoryRescueDataset(train_data_list, node_classification=node_classification), \
           InMemoryRescueDataset(test_data_list, node_classification=node_classification)


def get_notrandom(dataset, node_classification):
    data_list = []
    for data_item in dataset:
        if data_item.info_map['type'] == 'random':
            continue

        data_list.append(data_item)

    return InMemoryRescueDataset(data_list, node_classification=node_classification)


def get_notnull(dataset, node_classification):
    if node_classification:
        import sys
        print("get_notnull only works for non node classification")
        sys.exit(1)

    data_list = []
    for data_item in dataset:
        if data_item.y.item() == 0:
            continue
        data_list.append(data_item)

    return InMemoryRescueDataset(data_list, node_classification=node_classification)


def increase_key(class_count, key):
    if key in class_count:
        class_count[key] += 1
    else:
        class_count[key] = 1


def calculate_class_distribution(dataset):
    class_count = {}
    for i in range(len(dataset)):
        data = dataset[i]
        if dataset.node_classification:
            for node_class in data.y:
                increase_key(class_count, node_class)
        else:
            increase_key(class_count, data.y.item())

    # normalize
    total_classes = 0
    for class_idx in class_count:
        total_classes += class_count[class_idx]

    for class_idx in class_count:
        class_count[class_idx] = class_count[class_idx] / total_classes

    return {k: v for k, v in sorted(class_count.items(), key=lambda item: item[1], reverse=True)}


def save_for_inmemory_dataset():
    scn_list = ["test"+str(i) for i in range(400)]
    train_dataset = dataset_manager.get_dataset("/home/okan/rescuesim/rcrs-server/dataset", scn_list, read_map_info=True,
                                node_classification=False, agent_type="firebrigade", scn_dir="test", team="ait")

    scn_list = ["test" + str(i) for i in range(400, 500)]
    test_dataset = dataset_manager.get_dataset("/home/okan/rescuesim/rcrs-server/dataset", scn_list, read_map_info=True,
                                node_classification=False, agent_type="firebrigade", scn_dir="test", team="ait")
    print(len(train_dataset))
    print(len(test_dataset))

    train_notnull_dataset = get_notnull(train_dataset, train_dataset.node_classification)
    train_notnull_dataset.save("train_notnull_dataset.pt")

    test_notnull_dataset = get_notnull(test_dataset, test_dataset.node_classification)
    test_notnull_dataset.save("test_notnull_dataset.pt")

    train_notnull_notrandom_dataset = get_notrandom(train_notnull_dataset, train_notnull_dataset.node_classification)
    train_notnull_notrandom_dataset.save("train_notnull_notrandom_dataset.pt")

    test_notnull_notrandom_dataset = get_notrandom(test_notnull_dataset, test_notnull_dataset.node_classification)
    test_notnull_notrandom_dataset.save("test_notnull_notrandom_dataset.pt")


if __name__ == "__main__":
    # test_memorization_model()
    # test_random_target_selection()
    # test_fieryness_target_selection()
    # print_raw_fieryness_and_target()
    # test_null_target_selection()
    # save_for_inmemory_dataset()
    train_dataset = InMemoryRescueDataset([])
    train_dataset.load("train_notnull_notrandom_dataset.pt")
    print(len(train_dataset))

    test_dataset = InMemoryRescueDataset([])
    test_dataset.load("test_notnull_notrandom_dataset.pt")
    print(len(test_dataset))

