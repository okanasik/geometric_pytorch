import json
import os
import os.path as osp
import random
from zipfile import ZipFile
import torch
from agent_type import AgentType
from torch_geometric.data import Data
from torch_geometric.data import Dataset
import utils


class RescueDataset(Dataset):
    def __init__(self, root, agent_type, comp=None, scenario=None, team=None, node_classification=False,
                 start_datetime=None, end_datetime=None, read_info_map=False, transform=None, pre_transform=None):
        self.comp = comp
        self.scenario = scenario
        self.team = team
        self.agent_type = agent_type
        self.cache = {}
        self.max_cache_size = 100
        self.node_classification = node_classification
        self.start_datetime = start_datetime
        self.end_datetime = end_datetime
        self.metadata = {}
        self.read_info_map = read_info_map
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
    def raw_file_names(self):
        filenames = [filename for filename in os.listdir(self.raw_dir) if self.dataset_pattern in filename and filename[-3:] == "zip" and
                     self.is_datetime_valid(self.start_datetime, self.end_datetime, filename)]
        return sorted(filenames)

    @staticmethod
    def is_datetime_valid(start_datetime, end_datetime, filename):
        file_datetime = filename[filename.rfind('_')+1:filename.rfind('.')]
        file_datetime = utils.convert_to_datetime(file_datetime)

        if start_datetime:
            start_datetime = utils.convert_to_datetime(start_datetime)
            diff = file_datetime - start_datetime
            # note that the second is inclusive
            if diff.total_seconds() < 0:
                return False

        if end_datetime:
            end_datetime = utils.convert_to_datetime(end_datetime)
            diff = end_datetime - file_datetime
            # note that the second is inclusive
            if diff.total_seconds() < 0:
                return False

        return True



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
        # remove the first 3 frames
        return len(json_data["frames"])-3

    def process_metadata(self):
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

    def create_graph_data(self, json_data):
        null_node_id = self.add_null_node(json_data)
        data_list = []
        node_indexes, node_ids = self.create_node_indexes(json_data["graph"])
        edge_indexes, edge_attr = self.create_edges(json_data["graph"], node_indexes)
        node_poses = self.create_node_poses(json_data["graph"], node_ids)

        agent_id = json_data["agent"]["agentId"]
        agent_pos_id = self.get_agent_pos_id(json_data, agent_id, self.get_agent_list_name(json_data["agent"]["agentType"]))

        ambulance_pos_dict = self.create_agent_pos_dict(json_data["ambulances"])
        firebrigade_pos_dict = self.create_agent_pos_dict(json_data["firebrigades"])
        police_pos_dict = self.create_agent_pos_dict(json_data["polices"])
        civ_pos_dict = {}  # civ_id -> pos_id

        amb_node_counts = self.get_agent_counts(firebrigade_pos_dict)
        fb_node_counts = self.get_agent_counts(ambulance_pos_dict)
        police_node_counts = self.get_agent_counts(police_pos_dict)
        civ_node_counts = {} # pos_id -> count

        last_amb_count_poses = amb_node_counts.keys()
        last_fb_count_poses = fb_node_counts.keys()
        last_police_count_poses = police_node_counts.keys()
        last_civ_count_poses = civ_node_counts.keys()

        last_target_id = None
        if self.node_classification:
            y_val = torch.zeros(len(node_ids), dtype=torch.long) # set class of each node to 0
        else:
            y_val = torch.tensor([0], dtype=torch.long) # the class of the target

        # feature row: static_features, dynamic_features, new_features
        # static_features: is_refuge, is_gasstation, is_building, area*floors(volume)
        # dynamic_features: is_agent_here, fieryness, brokennes, repair_cost, num_fbs, num_ambs, num_polices, num_civilians
        # new_features: x pos, y pos

        size_feature = 14
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
                # calculate volumes
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

            # x and y position of the node
            x_features[node_idx][12] = node["x"]
            x_features[node_idx][13] = node["y"]

            node_idx += 1

        last_agent_pos_id = agent_pos_id

        # update features and output tensors frame by frame
        for frame in json_data["frames"]:
            nodes = frame["change"]["nodes"]
            agent_pos_id = self.get_agent_pos_id(frame["change"], agent_id, self.get_agent_list_name(json_data["agent"]["agentType"]))

            if last_agent_pos_id is not None:
                x_features[node_indexes[last_agent_pos_id]][4] = 0

            last_agent_pos_id = agent_pos_id

            # update agent positions
            self.update_agent_positions(ambulance_pos_dict, amb_node_counts, frame["change"]["ambulances"])
            self.update_agent_positions(firebrigade_pos_dict, fb_node_counts, frame["change"]["firebrigades"])
            self.update_agent_positions(police_pos_dict, police_node_counts, frame["change"]["polices"])
            self.update_agent_positions(civ_pos_dict, civ_node_counts, frame["change"]["civilians"])

            # reset previous count poses
            self.reset_count_poses(last_amb_count_poses, x_features, node_indexes, 8)
            self.reset_count_poses(last_fb_count_poses, x_features, node_indexes, 9)
            self.reset_count_poses(last_police_count_poses, x_features, node_indexes, 10)
            self.reset_count_poses(last_civ_count_poses, x_features, node_indexes, 11)

            last_amb_count_poses = amb_node_counts.keys()
            last_fb_count_poses = fb_node_counts.keys()
            last_police_count_poses = police_node_counts.keys()
            last_civ_count_poses = civ_node_counts.keys()


            for node_id_str in nodes:
                node_idx = node_indexes[int(node_id_str)]
                node = nodes[node_id_str]

                # set current agent position
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

            # reset previous node target
            if self.node_classification:
                if last_target_id is not None:
                    y_val[last_target_id] = 0

                if frame["action"]["type"] == "NULL":
                    last_target_id = node_indexes[null_node_id]
                    y_val[last_target_id] = 1
                else:
                    last_target_id = node_indexes[frame["action"]["targetId"]]
                    y_val[last_target_id] = 1
            else:
                if frame["action"]["type"] == "NULL":
                    y_val[0] = node_indexes[null_node_id]
                else:
                    y_val[0] = node_indexes[frame["action"]["targetId"]]

            # skip first 3 time steps
            if frame["time"] <= 3:
                continue

            frame_data = Data(x=x_features.clone(), edge_index=edge_indexes, edge_attr=edge_attr,
                              y=y_val.clone(), pos=node_poses)
            if self.read_info_map:
                if 'infoMap' in frame:
                    frame_data['info_map'] = frame['infoMap']
            data_list.append(frame_data)

        return data_list

    @staticmethod
    def reset_count_poses(last_pos_ids, x_features, node_indexes, col_idx):
        for pos_id in last_pos_ids:
            # only civilian in a building or on road
            # we do not handle the case where the civilian is being carried by an ambulance
            if pos_id in node_indexes:
                x_features[node_indexes[pos_id]][col_idx] = 0

    def calculate_class_distribution(self):
        class_count = {}
        if self.node_classification:
            class_count = {0:0, 1:0}
            for i in range(self.len()):
                data = self.get(i)
                for class_idx in data.y:
                    class_count[class_idx.item()] += 1
        else:
            for i in range(self.len()):
                data = self.get(i)
                if data.y.item() in class_count:
                    class_count[data.y.item()] += 1
                else:
                    class_count[data.y.item()] = 1

        # normalize
        total_classes = 0
        for class_idx in class_count:
            total_classes += class_count[class_idx]

        for class_idx in class_count:
            class_count[class_idx] = class_count[class_idx] / total_classes

        return {k:v for k, v in sorted(class_count.items(), key=lambda item: item[1], reverse=True)}

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
        edge_attr = torch.empty(len(graph["edges"]), dtype=torch.float)
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

    @staticmethod
    def add_null_node(json_data):
        # find empty node id
        null_id = 0
        while str(null_id) in json_data["graph"]["nodes"]:
            null_id += 1
        # add null node as a building with dummy values
        json_data["graph"]["nodes"][str(null_id)] = {"area":1, "floors":1, "fieryness":0, "brokennes":0, "id": null_id, "type": "BUILDING", "x": 0, "y": 0}
        # add edges between each node and null node
        # for node_id in json_data["graph"]["nodes"]:
        #     json_data["graph"]["edges"].append({"from": int(node_id), "to": null_id, "weight": 1})
        #     json_data["graph"]["edges"].append({"from": null_id, "to": int(node_id), "weight": 1})
        return null_id


if __name__ == "__main__":
    dataset = RescueDataset("/home/okan/rescuesim/rcrs-server/dataset", "firebrigade", comp="robocup2019",
                            scenario="test2", team="ait", node_classification=False)
    print(dataset.calculate_class_distribution())
    # print(dataset[1001])
    print(len(dataset))
    print(dataset[10])
    print(dataset.num_classes)