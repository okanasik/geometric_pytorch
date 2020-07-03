from dataset.rescue_dataset import RescueDataset
import rescue.dataset.data_adapter as data_adapter


class RescueDatasetList(RescueDataset):
    def __init__(self, dataset_list, max_cache_size=10000):
        metadata = {}
        for dataset in dataset_list:
            for key in dataset.metadata:
                metadata[key] = dataset.metadata[key]

        self.index_to_filename, self.index_to_inner_index, self.graph_count = \
            data_adapter.create_index_lookup(metadata)

        self.cache = {}
        self.max_cache_size = max_cache_size
        self.node_classification = dataset_list[0].node_classification
        self.root = dataset_list[0].root
        self.read_info_map = dataset_list[0].read_info_map
        self.pre_transform = None
        self.transform = None
        self.__indices__ = None

    def get(self, idx):
        if idx < 0 or idx > self.len():
            raise IndexError()

        filename = self.index_to_filename[idx]
        data_list = self.read_raw_dataset_file(filename)
        data = data_list[self.index_to_inner_index[idx]]
        if self.pre_transform is not None:
            data = self.pre_transform(data)
        return data

    def len(self):
        return self.graph_count

    @property
    def num_classes(self):
        r"""The number of classes in the dataset."""
        # whether node is selected as target or not
        if self.node_classification:
            return 2
        else:
            data = self.get(0)
            return data.x.size(0)


if __name__ == '__main__':
    dataset = RescueDataset("/home/okan/rescuesim/rcrs-server/dataset.old", "firebrigade", comp="robocup2019",
                            scenario="test2",
                            team="ait", node_classification=False)
    dataset1 = RescueDataset("/home/okan/rescuesim/rcrs-server/dataset.old", "firebrigade", comp="robocup2019",
                                 scenario="test3",
                                 team="ait", node_classification=False)

    print(len(dataset))
    print(len(dataset1))
    dataset_list = RescueDatasetList([dataset, dataset1])
    print(len(dataset_list))
