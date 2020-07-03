from rescue.dataset.rescue_dataset import RescueDataset


if __name__ == "__main__":
    rescuedataset = RescueDataset("./data", "firebrigade", comp="test",
                                   scenario="test0",
                                   team="ait", node_classification=False)
    print(rescuedataset.dataset_pattern)
    print(rescuedataset.raw_file_names)
    print(len(rescuedataset))
    for i in range(len(rescuedataset)):
        print(rescuedataset[i].x[4][5])