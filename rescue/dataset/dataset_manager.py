from rescue.dataset.rescue_dataset import RescueDataset
from rescue.dataset.rescue_dataset_list import RescueDatasetList


def get_dataset(dataset_dir, scenario_list, node_classification=False, read_map_info=False, agent_type="firebrigade",
                     scn_dir="test", team="ait"):
    datasets = []
    for scn in scenario_list:
        dataset = RescueDataset(dataset_dir, agent_type, comp=scn_dir,
                                   scenario=scn, team=team,
                                node_classification=node_classification, read_info_map=read_map_info)
        print(dataset.dataset_pattern)
        print(len(dataset))
        datasets.append(dataset)
    dataset_list = RescueDatasetList(datasets)
    print('Dataset len:{}'.format(len(dataset_list)))

    return dataset_list