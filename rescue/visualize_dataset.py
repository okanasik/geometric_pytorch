from torch_geometric.utils import to_networkx
import networkx as nx
from dataset.rescue_dataset import RescueDataset
import matplotlib.pyplot as plt


def create_pos_dict(graph_data):
    pos_dict = {}
    pos_idx = 0
    for node in graph_data.x:
        pos_dict[pos_idx] = (node[12].item(), node[13].item())
        pos_idx += 1
    return pos_dict


def create_node_colors(graph_data):
    # find agent position
    node_colors = ['#808080']*graph_data.num_nodes

    refuge_color = '#00ff00'
    road_color = '#808080'
    building_color = '#0000ff'
    agent_pos_color = '#00FFFF'
    target_pos_color = '#FF00FF'
    blocked_color = '#000000'

    # NORMAL=0;
    # HEATING=1;
    # BURNING=2;
    # INFERNO=3
    # EXTINGUISHED=5;
    # BURNED_DOWN=4;

    fieryness1_color = '#FFD859' # heating
    fieryness2_color = '#EC7325' # burning
    fieryness3_color = '#CB0E0E' # inferno
    fieryness4_color = '#490620' # burnt_down

    for idx, node in enumerate(graph_data.x):
        if node[2].item() == 1:
            node_colors[idx] = building_color

        if node[0].item() == 1:
            node_colors[idx] = refuge_color

        # roads have zero volumes
        if node[3].item() == 0:
            node_colors[idx] = road_color

        if node[7].item() > 0:
            node_colors[idx] = blocked_color

        if node[5].item() == 1:
            node_colors[idx] = fieryness1_color
        elif node[5].item() == 2:
            node_colors[idx] = fieryness2_color
        elif node[5].item() == 3:
            node_colors[idx] = fieryness3_color
        elif node[5].item() == 4:
            node_colors[idx] = fieryness4_color

        if node[4].item() == 1:
            node_colors[idx] = agent_pos_color

    node_colors[graph_data.y.item()] = target_pos_color

    return node_colors


def play_dataset(dataset, pause=0.2):
    plt.ion()
    fig_conf = plt.gcf()
    fig_conf.set_size_inches(12, 10)
    for idx, graph_data in enumerate(dataset):
        print("Drawing figure:{}".format(idx))
        graph = to_networkx(graph_data).to_undirected()

        # remove the null node that is connected to all nodes
        # graph.remove_node(0)

        pos_dict = create_pos_dict(graph_data)
        node_colors = create_node_colors(graph_data)

        # node_color can also be a list of colors, one color for each node
        nx.draw(graph, pos_dict, node_size=80, node_color=node_colors, width=1.0)

        plt.show()
        plt.pause(pause)
        plt.clf()


if __name__ == "__main__":
    dataset = RescueDataset("/home/okan/rescuesim/rcrs-server/dataset", "firebrigade", comp="test",
                            scenario="test2", team="ait", node_classification=False)
    play_dataset(dataset, pause=0.1)




