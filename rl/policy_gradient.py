from sim_wrapper.rescue_environment import RescueEnvironment
import random
from rescue.models.topk_model import TopKNet
from torch.optim import Adam
import torch
from rescue.dataset.inmemory_rescue_dataset import InMemoryRescueDataset
from torch_geometric.data.dataloader import DataLoader
from torch.distributions.categorical import Categorical
from rescue.models.model import Model


def compute_loss(model, obs, acts, rewards):
    dataset = InMemoryRescueDataset(obs, node_classification=False)
    dataloader = DataLoader(dataset, batch_size=len(obs), shuffle=False)
    for data_item in dataloader:
        output = model(data_item)
        action_dist = Categorical(logits=output)
        log_prob = action_dist.log_prob(acts)
    return -(log_prob * rewards).mean()


def train_one_epoch(num_sim, model, optimizer):
    rescue_env = RescueEnvironment()
    epoch_obs = []
    epoch_acts = []
    epoch_rewards = []
    for i in range(num_sim):
        # rnd_scn_index = random.randrange(400)  # choose from training scenarios
        rnd_scn_index = 0
        print("scn_index:{}".format(rnd_scn_index))
        rescue_env.set_scenario("/home/okan/rescuesim/scenarios/test", "test{}".format(rnd_scn_index))
        obs, acts, rewards = rescue_env.collect_data(model, 5)
        # add to current epoch data
        epoch_obs += obs
        epoch_acts += acts
        epoch_rewards += rewards
    rescue_env.close_rpc_server()

    print("policy gradient step")
    model.train()
    optimizer.zero_grad()
    acts_tensor = torch.tensor(epoch_acts, dtype=torch.int32)
    rewards_tensor = torch.tensor(epoch_rewards, dtype=torch.float32)
    model = model.to("cpu")
    loss = compute_loss(model, epoch_obs, acts_tensor, rewards_tensor)
    loss.backward()
    optimizer.step()

    return loss, rewards_tensor.mean()


def measure_performance(model):
    model.eval()
    pass


def train_policy_gradient():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # num_features = 14
    # num_of_outputs -> num_of_nodes+null_node = 96
    model = TopKNet(14, 96).to(device)
    # model.load_state_dict(torch.load("../rescue/models/topk_gat_test_notnull.pt"))
    # model = Model.load("../rescue/models/topk_gat_test_notnull.pt")
    optimizer = Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    # performance_rewards = []
    model_filename = "topk_gat_test0_rl.pt"
    for i in range(1001):
        loss, mean_reward = train_one_epoch(1, model, optimizer)
        print("epoch:{} reward:{} loss:{}".format(i, mean_reward, loss))
        if i%10 == 0:
            torch.save(model.state_dict(), model_filename)
        # avg_reward = measure_performance(model)
        # performance_rewards.append(avg_reward)


if __name__ == "__main__":
    train_policy_gradient()