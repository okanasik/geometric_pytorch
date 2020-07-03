import os
from zipfile import ZipFile
import json
from torch_geometric.datasets import Planetoid


def get_scenario_files(base_dir, file_pattern):
    return [base_dir + os.path.sep + filename for filename in os.listdir(base_dir) if file_pattern in filename]


def read_data_file(filename):
    zipfile = ZipFile(filename)
    json_data = zipfile.read(zipfile.namelist()[0])
    json_object = json.loads(json_data)
    return json_object


def read_scenario_data(base_dir, competition, scenario):
    file_pattern = competition + "_" + scenario
    data_files = get_scenario_files(base_dir, file_pattern)
    episode = []
    for filename in data_files:
        json_object = read_data_file(filename)
        # if json_object["agent"]["agentType"] == agent_type:
        episode.append(json_object)
        break
    return episode


if __name__ == "__main__":
    base_dir = "/home/okan/rescuesim/rcrs-server/dataset"
    comp = "robocup2019"
    scenario = "Kobe1"
    # episodes = read_scenario_data(base_dir, comp, scenario)
    # rescue_data = RescueData(episodes)

    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    print(dataset[0].edge_index)
    print(dataset[0].y.shape)

    # for frame in episodes[0]["frames"]:
    #     print(frame["time"])
    #     print(frame["action"])