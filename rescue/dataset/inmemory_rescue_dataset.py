from torch_geometric.data import Dataset
import torch


class InMemoryRescueDataset(Dataset):
    def __init__(self, data_list, node_classification=False):
        self.data_list = data_list
        self.node_classification = node_classification

        self.pre_transform = None
        self.transform = None
        self.__indices__ = None

    def get(self, idx):
        if idx < 0 or idx > self.len():
            raise IndexError()

        return self.data_list[idx]

    def len(self):
        return len(self.data_list)

    @property
    def num_classes(self):
        r"""The number of classes in the dataset."""
        # whether node is selected as target or not
        if self.node_classification:
            return 2
        else:
            data = self.get(0)
            return data.x.size(0)

    def save(self, file_name):
        torch.save(self.data_list, file_name)

    def load(self, file_name, device="cpu"):
        self.data_list = torch.load(file_name)

        # if gpu copy all data to gpu
        # if device != "cpu":
        #     for i in range(len(self.data_list)):
        #         self.data_list[i] = self.data_list[i].to(device)
