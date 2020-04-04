import json
import os
import os.path as osp
import random
from zipfile import ZipFile

import matplotlib.pyplot as plt
import networkx as nx
import torch
from agent_type import AgentType
from torch_geometric.data import Data
from torch_geometric.data import Dataset
from torch_geometric.utils.convert import to_networkx


class RescueDataset(Dataset):
    def __init__(self, root, agent_type, comp=None, scenario=None, team=None, transform=None, pre_transform=None):
        self.comp = comp
        self.scenario = scenario
        self.team = team
        self.agent_type = agent_type
        self.cache = {}
        self.max_cache_size = 100
        super(RescueDataset, self).__init__(root, transform, pre_transform)

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
    def metadata_filename(self):
        return self.dataset_pattern + "_metadata.json"

    @property
    def raw_file_names(self):
        filenames = [filename for filename in os.listdir(self.raw_dir) if self.dataset_pattern in filename and ".zip" in filename]
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
        data = self.get(0)
        return data.x.size(0)+1

    def len(self):
        count = 0
        for filename_key in self.metadata:
            count += self.metadata[filename_key]["num_graph"]
        return count

    def get(self, idx):
        if idx < 0 or idx > self.len():
            raise IndexError()
        filenames = sorted(self.metadata.keys())
        for filename in filenames:
            count = self.metadata[filename]["num_graph"]
            if idx < count:
                data_list = self.read_raw_dataset_file(filename)
                data = data_list[idx]
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                return data
            else:
                idx -= count

    def process(self):
        self.process_metadata()
        print(self.raw_file_names)
        print(self.dataset_pattern)
        for filename in self.raw_file_names:
            if filename in self.metadata:
                print("skipping:" + filename + " already processed.")
                continue
            print("Processing: " + filename)
            full_filename = osp.join(self.raw_dir, filename)
            json_data = self.read_raw_json_file(full_filename)
            num_graph = self.get_num_graph(json_data)
            self.add_metadata(filename, "num_graph", num_graph)
        self.save_metadata()

    def read_raw_dataset_file(self, filename):
        if filename in self.cache:
            return self.cache[filename]

        full_filename = osp.join(self.raw_dir, filename)
        json_data = self.read_raw_json_file(full_filename)
        data_list = self.create_graph_data(json_data)
        if len(self.cache) > self.max_cache_size:
            del self.cache[random.sample(self.cache.keys())]
        self.cache[filename] = data_list
        return data_list


    def get_num_graph(self, json_data):
        return len(json_data["frames"])

    def process_metadata(self):
        self.metadata = {}
        full_name = osp.join(self.root, self.metadata_filename)
        if osp.exists(full_name):
            with open(full_name) as json_file:
                self.metadata = json.load(json_file)

    def add_metadata(self, filename, data_key, data_value):
        self.metadata[filename] = {}
        self.metadata[filename][data_key] = data_value

    def save_metadata(self):
        with open(osp.join(self.root, self.metadata_filename), "w") as json_file:
            json.dump(self.metadata, json_file)

    def create_graph_data(self, json_data):
        data_list = []
        node_indexes, node_ids = self.create_node_indexes(json_data["graph"])
        edge_indexes, edge_attr = self.create_edges(json_data["graph"], node_indexes)
        node_poses = self.create_node_poses(json_data["graph"], node_ids)

        agent_id = json_data["agent"]["agentId"]
        agent_pos_id = self.get_agent_pos_id(json_data, agent_id, self.get_agent_list_name(json_data["agent"]["agentType"]))

        ambulance_pos_dict = self.create_agent_pos_dict(json_data["ambulances"])
        firebrigade_pos_dict = self.create_agent_pos_dict(json_data["firebrigades"])
        police_pos_dict = self.create_agent_pos_dict(json_data["polices"])

        fb_node_counts = self.get_agent_counts(ambulance_pos_dict)
        amb_node_counts = self.get_agent_counts(firebrigade_pos_dict)
        police_node_counts = self.get_agent_counts(police_pos_dict)

        civ_pos_dict = {} # civ_id -> pos_id
        civ_node_counts = {} # pos_id -> count

        # feature row: static_features, dynamic_features
        # static_features: is_refuge, is_gasstation, is_building, area*floors(volume)
        # dynamic_features: is_agent_here, fieryness, brokennes, repair_cost, num_fbs, num_ambs, num_polices, num_civilians

        size_feature = 12
        x_features = torch.zeros(len(node_ids), size_feature, dtype=torch.float32)
        node_idx = 0
        for node_id in node_ids:
            node = json_data["graph"]["nodes"][str(node_id)]

            # set static features
            if node["type"] == "REFUGE":
                x_features[node_idx][0] = 1

            if node["type"] == "GASSTATION":
                x_features[node_idx][1] = 1

            if node["type"] == "BUILDING" or node["type"] == "GASSTATION" or node["type"] == "AMBULANCECENTRE" or\
                node["type"] == "FIRESTATION" or node["type"] == "POLICEOFFICE" or node["type"] == "REFUGE":
                x_features[node_idx][2] = 1

            if "area" in node:
                x_features[node_idx][3] = node["area"]*node["floors"]

            # dynamic features
            # current agent position
            if agent_pos_id == node_id:
                x_features[node_idx][4] = 1

            # set fieryness as zero for the first timestep
            x_features[node_idx][5] = 0

            # set brokennes as zero for the first timestep
            x_features[node_idx][6] = 0

            # set repair_cost as zero for the first timestep
            x_features[node_idx][7] = 0

            if node_id in amb_node_counts:
                x_features[node_idx][8] = amb_node_counts[node_id]

            if node_id in fb_node_counts:
                x_features[node_idx][9] = fb_node_counts[node_id]

            if node_id in police_node_counts:
                x_features[node_idx][10] = police_node_counts[node_id]

            # num_civilians
            x_features[node_idx][11] = 0

            node_idx += 1

        # update features and output tensors frame by frame
        for frame in json_data["frames"]:
            nodes = frame["change"]["nodes"]
            agent_pos_id = self.get_agent_pos_id(frame["change"], agent_id, self.get_agent_list_name(json_data["agent"]["agentType"]))

            # update agent positions
            self.update_agent_positions(ambulance_pos_dict, amb_node_counts, frame["change"]["ambulances"])
            self.update_agent_positions(firebrigade_pos_dict, fb_node_counts, frame["change"]["firebrigades"])
            self.update_agent_positions(police_pos_dict, police_node_counts, frame["change"]["polices"])
            self.update_agent_positions(civ_pos_dict, civ_node_counts, frame["change"]["civilians"])

            for node_id_str in nodes:
                node_idx = node_indexes[int(node_id_str)]
                node = nodes[node_id_str]

                # set agent position
                if node_id_str == str(agent_pos_id):
                    x_features[node_idx][4] = 1

                # set fieryness
                if "fieryness" in node:
                    x_features[node_idx][5] = node["fieryness"]

                # set brokennes
                if "brokennes" in node:
                    x_features[node_idx][6] = node["brokennes"]

                # set repair cost
                if "repairCost" in node:
                    x_features[node_idx][7] = node["repairCost"]

                node_id = int(node_id_str)

                if node_id in amb_node_counts:
                    x_features[node_idx][8] = amb_node_counts[node_id]

                if node_id in fb_node_counts:
                    x_features[node_idx][9] = fb_node_counts[node_id]

                if node_id in police_node_counts:
                    x_features[node_idx][10] = police_node_counts[node_id]

                if node_id in civ_node_counts:
                    x_features[node_idx][11] = civ_node_counts[node_id]

            y_val = len(node_ids)
            if frame["action"]["type"] != "NULL":
                y_val = node_indexes[frame["action"]["targetId"]]

            frame_data = Data(x=x_features.clone(), edge_index=edge_indexes, edge_attr=edge_attr, y=y_val, pos=node_poses)
            data_list.append(frame_data)

        return data_list

    @staticmethod
    def get_agent_list_name(agent_type):
        if agent_type == AgentType.AMBULANCE:
            return "ambulances"
        elif agent_type == AgentType.FIREBRIGADE:
            return "firebrigades"
        elif agent_type == AgentType.POLICE:
            return "polices"

    @staticmethod
    def get_agent_pos_id(json_data, curr_agent_id, agent_type):
        agent_list = json_data[agent_type]
        for agent_id in agent_list:
            if agent_id == str(curr_agent_id):
                return agent_list[agent_id]["posId"]

    @staticmethod
    def create_agent_pos_dict(agent_dict):
        pos_dict = {}
        for agent_id_str in agent_dict:
            pos_dict[agent_id_str] = agent_dict[agent_id_str]["posId"]
        return pos_dict


    @staticmethod
    def get_agent_counts(agent_pos_dict):
        pos_count = {}
        for agent_id in agent_pos_dict:
            pos_id = agent_pos_dict[agent_id]
            if pos_id in pos_count:
                pos_count[pos_id] += 1
            else:
                pos_count[pos_id] = 1
        return pos_count

    @staticmethod
    def update_agent_positions(agent_pos_dict, count_dict, agent_dict):
        for agent_id_str in agent_dict:
            if agent_id_str in agent_pos_dict and agent_pos_dict[agent_id_str] in count_dict:
                count_dict[agent_pos_dict[agent_id_str]] -= 1

            agent_pos_dict[agent_id_str] = agent_dict[agent_id_str]["posId"]
            if agent_pos_dict[agent_id_str] in count_dict:
                count_dict[agent_pos_dict[agent_id_str]] += 1
            else:
                count_dict[agent_pos_dict[agent_id_str]] = 1

    @staticmethod
    def get_agent_ids(episode, agent_type):
        ids = [int(agent_id) for agent_id in episode[agent_type]]
        return sorted(ids)

    @staticmethod
    def create_node_poses(graph, node_ids):
        node_poses = torch.empty(len(graph["nodes"]), 2, dtype=torch.long)
        idx = 0
        for node_id in node_ids:
            node = graph["nodes"][str(node_id)]
            node_poses[idx][0] = node["x"]
            node_poses[idx][1] = node["y"]
            idx += 1
        return node_poses

    @staticmethod
    def create_edges(graph, node_indexes):
        edge_indexes = torch.empty(2, len(graph["edges"]), dtype=torch.long)
        edge_attr = torch.empty(len(graph["edges"]), dtype=torch.int32)
        idx = 0
        for edge in graph["edges"]:
            edge_attr[idx] = edge["weight"]
            edge_indexes[0][idx] = node_indexes[edge["from"]]
            edge_indexes[1][idx] = node_indexes[edge["to"]]
            idx += 1
        return edge_indexes, edge_attr

    @staticmethod
    def create_node_indexes(graph):
        node_indexes = {}
        node_ids = sorted([int(node_id) for node_id in graph["nodes"]])
        index = 0
        for node_id in node_ids:
            node_indexes[node_id] = index
            index += 1
        return node_indexes, node_ids

    @staticmethod
    def read_raw_json_file(filename):
        zipfile = ZipFile(filename)
        json_data = zipfile.read(zipfile.namelist()[0])
        json_object = json.loads(json_data)
        return json_object


def visualize(graphdata):
    nxgraph = to_networkx(graphdata)
    plt.figure(1, figsize=(24,20))
    nx.draw(nxgraph, cmap=plt.get_cmap("Set1"), node_size=35, linewidths=6)
    plt.show()

if __name__ == "__main__":
    dataset = RescueDataset("/home/okan/rescuesim/rcrs-server/dataset", "firebrigade", "robocup2019", "Kobe1", "ait")
    print(dataset[1001])