from rescue.dataset.rescue_dataset_list import RescueDatasetList
from rescue.dataset.rescue_dataset import RescueDataset


if __name__ == "__main__":
    rescuedataset1 = RescueDataset("./data", "firebrigade", comp="test",
                            scenario="test0",
                            team="ait", node_classification=False)
    print(len(rescuedataset1))

    rescuedataset2 = RescueDataset("./data", "firebrigade", comp="test",
                                   scenario="test1",
                                   team="ait", node_classification=False)
    print(len(rescuedataset2))

    dataset_all = RescueDatasetList([rescuedataset1, rescuedataset2])
    print(len(dataset_all))

    for data in dataset_all:
        print(data.x[4][5])