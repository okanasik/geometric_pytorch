import json
import os
import os.path as osp
import random
import torch
from torch_geometric.data import Dataset
import rescue.dataset.data_adapter as data_adapter


class RescueDataset(Dataset):
    def __init__(self, root, agent_type, comp=None, scenario=None, team=None, node_classification=False,
                 start_datetime=None, end_datetime=None, read_info_map=False, transform=None, pre_transform=None,
                 max_cache_size=100):
        self.comp = comp
        self.scenario = scenario
        self.team = team
        self.agent_type = agent_type
        self.cache = {}
        self.max_cache_size = max_cache_size
        self.node_classification = node_classification
        self.start_datetime = start_datetime
        self.end_datetime = end_datetime
        self.read_info_map = read_info_map

        self.metadata = {}

        super(RescueDataset, self).__init__(root, transform, pre_transform)
        self.index_to_filename, self.index_to_inner_index, self.graph_count = \
            data_adapter.create_index_lookup(self.metadata)

    @property
    def dataset_pattern(self):
        pattern = ""
        if self.comp:
            pattern += self.comp
        if self.scenario:
            pattern += "_" + self.scenario
        if self.team:
            pattern += "_" + self.team
        pattern += "_" + self.agent_type
        return pattern

    @property
    def raw_file_names(self):
        filenames = [filename for filename in os.listdir(self.raw_dir)
                     if self.dataset_pattern in filename and filename[-3:] == "zip" and
                     data_adapter.is_datetime_valid(self.start_datetime, self.end_datetime, filename)]
        return sorted(filenames)

    @property
    def raw_dir(self):
        return self.root

    @property
    def processed_file_names(self):
        return ["data.pt", "pre_filter.pt", "pre_transform.pt"]

    @property
    def num_classes(self):
        r"""The number of classes in the dataset."""
        # whether node is selected as target or not
        if self.node_classification:
            return 2
        else:
            data = self.get(0)
            return data.x.size(0)

    def len(self):
        return self.graph_count

    def get(self, idx):
        if idx < 0 or idx > self.len():
            raise IndexError()

        filename = self.index_to_filename[idx]
        data_list = self.read_raw_dataset_file(filename)
        data = data_list[self.index_to_inner_index[idx]]
        if self.pre_transform is not None:
            data = self.pre_transform(data)
        return data

    def process(self):
        self.read_metadata()
        # print(self.raw_file_names)
        # print(self.dataset_pattern)
        for filename in self.raw_file_names:
            if filename in self.metadata:
                # print("skipping:" + filename + " already processed.")
                continue
            # print("Processing: " + filename)
            full_filename = osp.join(self.raw_dir, filename)
            json_data = data_adapter.read_raw_json_file(full_filename)
            num_graph = data_adapter.get_num_graph(json_data)
            self.add_metadata(filename, "num_graph", num_graph)
        self.save_metadata()

    def read_raw_dataset_file(self, filename):
        if filename in self.cache:
            return self.cache[filename]

        full_filename = osp.join(self.raw_dir, filename)
        json_data = data_adapter.read_raw_json_file(full_filename)
        data_list = data_adapter.create_graph_data(json_data, self.node_classification, self.read_info_map)
        if len(self.cache) > self.max_cache_size:
            del self.cache[random.choice(self.cache.keys())]
        self.cache[filename] = data_list
        return data_list

    def read_metadata(self):
        for file_name in self.raw_file_names:
            metadata_filename = file_name.replace('.zip', '_metadata.json')
            metadata_filename = osp.join(self.root, metadata_filename)
            if osp.exists(metadata_filename):
                with open(metadata_filename) as json_file:
                    self.metadata[file_name] = json.load(json_file)

    def add_metadata(self, filename, data_key, data_value):
        if filename not in self.metadata:
            self.metadata[filename] = {}
        self.metadata[filename][data_key] = data_value

    def save_metadata(self):
        for filename_key in self.metadata:
            metadata_filename = filename_key.replace('.zip', '_metadata.json')
            with open(osp.join(self.root, metadata_filename), "w") as json_file:
                json.dump(self.metadata[filename_key], json_file)


if __name__ == "__main__":
    # dataset = RescueDataset("/home/okan/rescuesim/rcrs-server/dataset", "firebrigade", comp="robocup2019",
    #                         scenario="test2", team="ait", node_classification=False)
    # print(dataset.calculate_class_distribution())
    # # print(dataset[1001])
    # print(len(dataset))
    # print(dataset[10])
    # print(dataset.num_classes)

    from dataset.inmemory_rescue_dataset import InMemoryRescueDataset
    from torch_geometric.data.dataloader import DataLoader
    test_dataset = InMemoryRescueDataset([], node_classification=False)
    test_dataset.load('test_dataset.pt')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    for test_data in test_loader:
        print(test_data)
        print(test_data.batch)
        print(test_data.batch.size())
        print(test_data.x.size(0))
        print(test_data.batch.dtype)
        batch_indexes = torch.zeros(test_data.x.size(0), dtype=torch.long)
        print(batch_indexes)