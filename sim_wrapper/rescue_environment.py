from sim_wrapper.scenario_runner import ScenarioRunner
import os
import threading
from conn.pika_rpc_server import PikaRPCServer
from rescue.dataset.scenario_manager import get_random_seed
from rescue.dataset.scenario_manager import set_random_seed
import random
from rescue.dataset import data_adapter
import torch
from torch.distributions.categorical import Categorical


class RescueEnvironment(object):
    def __init__(self):
        self.scn_folder = None
        self.scn_name = None

        self.observations = {}
        self.actions = {}
        self.rewards = {}

        self.model = None

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.current_sim = -1

        self.building_ids = []
        self.building_indexes = []

        self.rpc_server = PikaRPCServer()
        self.rpc_server.connect()
        self.rpc_server.listen_queue("observation_rl", self.observation_callback)

        # listen in a thread
        self.rpc_thread = threading.Thread(target=self.rpc_server.start_listening)
        self.rpc_thread.start()

    def set_scenario(self, scn_folder, scn_name):
        self.scn_folder = scn_folder
        self.scn_name = scn_name

    def collect_data(self, model, num_sim):
        self.observations = {} # agent_id:list of observations
        self.actions = {} # agent_id: list of actions
        self.rewards = {}
        self.building_ids = []
        self.building_indexes = []

        self.model = model
        self.model = model.to(self.device)
        self.model.eval()

        for i in range(num_sim):
            self.current_sim = i
            self.observations[self.current_sim] = {}
            self.actions[self.current_sim] = {}
            scn_path = os.path.join(self.scn_folder, self.scn_name)
            current_rnd_seed = get_random_seed(scn_path)
            new_rnd_seed = random.randrange(1000000)
            set_random_seed(scn_path, new_rnd_seed)
            scenario_runner = ScenarioRunner("aitrl", "false", scn_path, viewer="")
            final_score, scores = scenario_runner.run()
            set_random_seed(scn_path, current_rnd_seed)
            sum_rew = sum(scores[3:]) # we ignore the first three time steps
            self.rewards[self.current_sim] = [sum_rew] * len(scores[3:]) # we ignore the first three time steps

        # rearrange observations, actions and rewards of all simulations as single lists
        all_observations = []
        all_actions = []
        all_rewards = []
        for sim_idx in self.observations:
            rewards = self.rewards[sim_idx]
            for agent_id in self.observations[sim_idx]:
                all_observations += self.observations[sim_idx][agent_id]
                all_actions += self.actions[sim_idx][agent_id]
                all_rewards += rewards

        return all_observations, all_actions, all_rewards

    # def get_buildings_on_fire(self, feature_matrix):
    #     fieryness_values = feature_matrix[:, 5].view(-1)
    #     nodes_with_fieryness = [idx for idx, fieryness in enumerate(fieryness_values) if
    #                             fieryness >= 1 and fieryness <= 3]
    #     return nodes_with_fieryness

    def observation_callback(self, json_object):
        # ignore first three time steps, return just any building from the graph
        if json_object["time"] <= 3:
            # calculate the building ids
            if len(self.building_indexes) == 0:
                data_adapter.add_null_node(json_object["graph"])
                node_indexes, node_ids = data_adapter.create_node_indexes(json_object["graph"])
                self.building_ids = [int(node_id) for node_id in json_object["graph"]["nodes"]
                                     if json_object["graph"]["nodes"][node_id]["type"] == "BUILDING" and
                                     int(node_id) != 0]
                self.building_indexes = [node_indexes[b_id] for b_id in self.building_ids]
            return self.building_ids[0]

        data, node_indexes, node_ids = data_adapter.convert_to_graph_data(json_object["graph"],
                                                                          json_object["ambulances"],
                                                                          json_object["firebrigades"],
                                                                          json_object["polices"],
                                                                          json_object["civilians"],
                                                                          json_object["agent"])

        data_adapter.add_batch(data)
        data = data.to(self.device)
        output = self.model(data)
        del data.batch
        # make sure that we sample only the building
        action_dist = Categorical(logits=output[0][self.building_indexes])
        sample = action_dist.sample()
        sampled_building_index = self.building_indexes[sample.item()]
        target_id = node_ids[sampled_building_index]
        self.add_action(json_object["agent"]["id"], sampled_building_index) # save as the node index not node id

        data = data.to("cpu")
        self.add_observation(json_object["agent"]["id"], data)

        return target_id

    def add_observation(self, agent_id, data):
        if agent_id not in self.observations[self.current_sim]:
            self.observations[self.current_sim][agent_id] = []
        self.observations[self.current_sim][agent_id].append(data)

    def add_action(self, agent_id, target_index):
        if agent_id not in self.actions[self.current_sim]:
            self.actions[self.current_sim][agent_id] = []
        self.actions[self.current_sim][agent_id].append(target_index)

    def close_rpc_server(self):
        try:
            self.rpc_server.stop_listening()
        except:
            pass



if __name__ == "__main__":
    from rescue.models.model import Model
    rescue_env = RescueEnvironment()
    rescue_env.set_scenario("/home/okan/rescuesim/scenarios/test", "test400")

    model = Model.load("../rescue/models/topk_gat_test_notnull.pt")
    obs, acts, rewards = rescue_env.collect_data(model, 1)

    print(len(obs))
    print(obs)
    print(len(acts))
    print(acts)
    print(len(rewards))
    print(rewards)

    rescue_env.close_rpc_server()




