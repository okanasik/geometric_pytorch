import rescue.utils as utils
from zipfile import ZipFile
import json
import torch
from torch_geometric.data import Data
from rescue.dataset.agent_type import AgentType


def is_datetime_valid(start_datetime, end_datetime, filename):
    file_datetime = filename[filename.rfind('_') + 1:filename.rfind('.')]
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


def create_index_lookup(metadata):
    current_count = 0
    index_to_filename = {}
    index_to_inner_index = {}
    for i, filename in enumerate(sorted(metadata.keys())):
        for idx in range(current_count, current_count + metadata[filename]["num_graph"]):
            index_to_filename[idx] = filename
            index_to_inner_index[idx] = idx - current_count
        current_count += metadata[filename]["num_graph"]
    return index_to_filename, index_to_inner_index, current_count


def add_null_node(graph):
    # find empty node id
    null_id = 0
    while str(null_id) in graph["nodes"]:
        null_id += 1
    # add null node as a building with dummy values
    graph["nodes"][str(null_id)] = {"area": 1, "floors": 1, "fieryness": 0, "brokennes": 0, "id": null_id,
                                    "type": "BUILDING", "x": 0, "y": 0}
    # add edges between each node and null node
    # for node_id in json_data["graph"]["nodes"]:
    #     json_data["graph"]["edges"].append({"from": int(node_id), "to": null_id, "weight": 1})
    #     json_data["graph"]["edges"].append({"from": null_id, "to": int(node_id), "weight": 1})
    return null_id


def create_node_indexes(graph):
    node_ids = sorted([int(node_id) for node_id in graph["nodes"]])
    node_indexes = {node_id:i for i, node_id in enumerate(node_ids)}
    return node_indexes, node_ids


def read_raw_json_file(filename):
    zipfile = ZipFile(filename)
    json_data = zipfile.read(zipfile.namelist()[0])
    json_object = json.loads(json_data)
    return json_object


def add_batch(data):
    batch_indexes = torch.zeros(data.x.size(0), dtype=torch.long)
    data.batch = batch_indexes


def create_edges(graph, node_indexes):
    edge_indexes = torch.empty(2, len(graph["edges"]), dtype=torch.long)
    edge_attr = torch.empty(len(graph["edges"]), dtype=torch.float)

    for i, edge in enumerate(graph["edges"]):
        edge_attr[i] = edge["weight"]
        edge_indexes[0][i] = node_indexes[edge["from"]]
        edge_indexes[1][i] = node_indexes[edge["to"]]
    return edge_indexes, edge_attr


def convert_to_graph_data(graph, ambulances, firebrigades, polices, civilians, agent):
    null_node_id = add_null_node(graph)
    node_indexes, node_ids = create_node_indexes(graph)
    edge_indexes, edge_attr = create_edges(graph, node_indexes)

    amb_node_counts = count_agents(ambulances)
    fb_node_counts = count_agents(firebrigades)
    police_node_counts = count_agents(polices)
    civ_node_counts = count_agents(civilians)

    x_features = cal_features(graph, node_ids, agent["posId"], amb_node_counts,
                               fb_node_counts, police_node_counts, civ_node_counts)
    data = Data(x=x_features, edge_index=edge_indexes, edge_attr=edge_attr)
    return data, node_indexes, node_ids


def count_agents(agents):
    pos_dict = {}
    for agent_id in agents:
        pos_id = agents[agent_id]["posId"]
        if pos_id in pos_dict:
            pos_dict[pos_id] += 1
        else:
            pos_dict[pos_id] = 1
    return pos_dict


def get_agent_ids(episode, agent_type):
    ids = [int(agent_id) for agent_id in episode[agent_type]]
    return sorted(ids)


def create_node_poses(graph, node_ids):
    node_poses = torch.empty(len(graph["nodes"]), 2, dtype=torch.long)
    idx = 0
    for node_id in node_ids:
        node = graph["nodes"][str(node_id)]
        node_poses[idx][0] = node["x"]
        node_poses[idx][1] = node["y"]
        idx += 1
    return node_poses


def get_agent_list_name(agent_type):
    if agent_type == AgentType.AMBULANCE:
        return "ambulances"
    elif agent_type == AgentType.FIREBRIGADE:
        return "firebrigades"
    elif agent_type == AgentType.POLICE:
        return "polices"


def get_agent_pos_id(json_data, curr_agent_id, agent_type):
    agent_list = json_data[agent_type]
    for agent_id in agent_list:
        if agent_id == str(curr_agent_id):
            return agent_list[agent_id]["posId"]


def create_agent_pos_dict(agent_dict):
    pos_dict = {}
    for agent_id_str in agent_dict:
        pos_dict[agent_id_str] = agent_dict[agent_id_str]["posId"]
    return pos_dict


def get_agent_counts(agent_pos_dict):
    pos_count = {}
    for agent_id in agent_pos_dict:
        pos_id = agent_pos_dict[agent_id]
        if pos_id in pos_count:
            pos_count[pos_id] += 1
        else:
            pos_count[pos_id] = 1
    return pos_count


def update_agent_positions(agent_pos_dict, count_dict, agent_dict):
    for agent_id_str in agent_dict:
        if agent_id_str in agent_pos_dict and agent_pos_dict[agent_id_str] in count_dict:
            count_dict[agent_pos_dict[agent_id_str]] -= 1

        agent_pos_dict[agent_id_str] = agent_dict[agent_id_str]["posId"]
        if agent_pos_dict[agent_id_str] in count_dict:
            count_dict[agent_pos_dict[agent_id_str]] += 1
        else:
            count_dict[agent_pos_dict[agent_id_str]] = 1


def cal_features(graph, node_ids, agent_pos_id, amb_node_counts, fb_node_counts,
                 police_node_counts, civ_node_counts):
    size_feature = 14
    x_features = torch.zeros(len(node_ids), size_feature, dtype=torch.float32)
    for i, node_id in enumerate(node_ids):
        node = graph["nodes"][str(node_id)]

        # set static features
        if node["type"] == "REFUGE":
            x_features[i][0] = 1

        if node["type"] == "GASSTATION":
            x_features[i][1] = 1

        if node["type"] == "BUILDING" or node["type"] == "GASSTATION" or node["type"] == "AMBULANCECENTRE" or \
                node["type"] == "FIRESTATION" or node["type"] == "POLICEOFFICE" or node["type"] == "REFUGE":
            x_features[i][2] = 1

        if "area" in node:
            # calculate volumes
            x_features[i][3] = node["area"] * node["floors"]

        # dynamic features
        # current agent position
        if agent_pos_id == node_id:
            x_features[i][4] = 1

        # set fieryness as zero for the first timestep
        if "fieryness" in node:
            x_features[i][5] = node["fieryness"]
        else:
            x_features[i][5] = 0

        # set brokennes as zero for the first timestep
        if "brokennes" in node:
            x_features[i][6] = node["brokennes"]
        else:
            x_features[i][6] = 0

        # set repair_cost as zero for the first timestep
        if "repairCost" in node:
            x_features[i][7] = node["repairCost"]
        else:
            x_features[i][7] = 0

        if node_id in amb_node_counts:
            x_features[i][8] = amb_node_counts[node_id]

        if node_id in fb_node_counts:
            x_features[i][9] = fb_node_counts[node_id]

        if node_id in police_node_counts:
            x_features[i][10] = police_node_counts[node_id]

        if node_id in civ_node_counts:
            x_features[i][11] = civ_node_counts[node_id]

        # x and y position of the node
        x_features[i][12] = node["x"]
        x_features[i][13] = node["y"]

    return x_features


def reset_count_poses(last_pos_ids, x_features, node_indexes, col_idx):
    for pos_id in last_pos_ids:
        # only civilian in a building or on road
        # we do not handle the case where the civilian is being carried by an ambulance
        if pos_id in node_indexes:
            x_features[node_indexes[pos_id]][col_idx] = 0


def create_graph_data(json_data, node_classification, read_info_map):
    null_node_id = add_null_node(json_data["graph"])
    data_list = []
    node_indexes, node_ids = create_node_indexes(json_data["graph"])
    edge_indexes, edge_attr = create_edges(json_data["graph"], node_indexes)
    node_poses = create_node_poses(json_data["graph"], node_ids)

    agent_id = json_data["agent"]["agentId"]
    agent_pos_id = get_agent_pos_id(json_data, agent_id, get_agent_list_name(json_data["agent"]["agentType"]))

    ambulance_pos_dict = create_agent_pos_dict(json_data["ambulances"])
    firebrigade_pos_dict = create_agent_pos_dict(json_data["firebrigades"])
    police_pos_dict = create_agent_pos_dict(json_data["polices"])
    civ_pos_dict = {}  # civ_id -> pos_id

    amb_node_counts = get_agent_counts(firebrigade_pos_dict)
    fb_node_counts = get_agent_counts(ambulance_pos_dict)
    police_node_counts = get_agent_counts(police_pos_dict)
    civ_node_counts = {}  # pos_id -> count

    last_amb_count_poses = amb_node_counts.keys()
    last_fb_count_poses = fb_node_counts.keys()
    last_police_count_poses = police_node_counts.keys()
    last_civ_count_poses = civ_node_counts.keys()

    last_target_id = None
    if node_classification:
        y_val = torch.zeros(len(node_ids), dtype=torch.long)  # set class of each node to 0
    else:
        y_val = torch.tensor([0], dtype=torch.long)  # the class of the target

    # feature row: static_features, dynamic_features, new_features
    # static_features: is_refuge, is_gasstation, is_building, area*floors(volume)
    # dynamic_features: is_agent_here, fieryness, brokennes, repair_cost, num_fbs, num_ambs, num_polices, num_civilians
    # new_features: x pos, y pos

    x_features = cal_features(json_data["graph"], node_ids, agent_pos_id, amb_node_counts, fb_node_counts,
                      police_node_counts, civ_node_counts)
    last_agent_pos_id = agent_pos_id

    # update features and output tensors frame by frame
    for frame in json_data["frames"]:
        nodes = frame["change"]["nodes"]
        agent_pos_id = get_agent_pos_id(frame["change"], agent_id,
                                             get_agent_list_name(json_data["agent"]["agentType"]))

        if last_agent_pos_id is not None:
            x_features[node_indexes[last_agent_pos_id]][4] = 0

        last_agent_pos_id = agent_pos_id

        # update agent positions
        update_agent_positions(ambulance_pos_dict, amb_node_counts, frame["change"]["ambulances"])
        update_agent_positions(firebrigade_pos_dict, fb_node_counts, frame["change"]["firebrigades"])
        update_agent_positions(police_pos_dict, police_node_counts, frame["change"]["polices"])
        update_agent_positions(civ_pos_dict, civ_node_counts, frame["change"]["civilians"])

        # reset previous count poses
        reset_count_poses(last_amb_count_poses, x_features, node_indexes, 8)
        reset_count_poses(last_fb_count_poses, x_features, node_indexes, 9)
        reset_count_poses(last_police_count_poses, x_features, node_indexes, 10)
        reset_count_poses(last_civ_count_poses, x_features, node_indexes, 11)

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
        if node_classification:
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
        if read_info_map:
            if 'infoMap' in frame:
                frame_data['info_map'] = frame['infoMap']
        data_list.append(frame_data)

    return data_list


def get_num_graph(json_data):
    # remove the first 3 frames if it is 1st, 2nd or 3rd frame
    skip_count = 0
    for frame in json_data["frames"]:
        if frame["time"] == 1 or frame["time"] == 2 or frame["time"] == 3:
            skip_count += 1
        else:
            break

    return len(json_data["frames"]) - skip_count

