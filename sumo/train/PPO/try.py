import os
CURRENT_PATH = os.getcwd()

import pygame
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import sklearn
import torch
from torch.nn import functional as F
import torch.optim as optim

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from olympics_engine.scenario import wrestling
from olympics_engine.generator import create_scenario


class Expertise_NN(torch.nn.Module):
    def __init__(self):
        super(Expertise_NN, self).__init__()
        self.Conv = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
        self.BN = torch.nn.BatchNorm2d(num_features=1)
        self.Pool = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.NN1 = torch.nn.Linear(1600, 400)
        self.NN2 = torch.nn.Linear(400, 128)
        self.NN3 = torch.nn.Linear(128, 64)
        self.NN4 = torch.nn.Linear(64, 2)


    def forward(self, input):
        x = self.Conv(input)
        x = self.BN(x)
        x = F.relu(x)
        x = self.Pool(x)
        x = F.dropout(x, p=0.2)
        x = torch.flatten(input, start_dim=1)

        x = self.NN1(x)
        x = F.relu(x)
        x = self.NN2(x)
        x = F.relu(x)
        x = self.NN3(x)
        x = F.relu(x)
        x = self.NN4(x)

        x[:, 0] = torch.tanh(x[:, 0])*150 + 50
        x[:, 1] = torch.tanh(x[:, 1])*30

        return x

class Expertise:
    def __init__(self, device, path=None):
        self.device = device


        self.model = Expertise_NN().to(device)
        self.model.load_state_dict(torch.load(CURRENT_PATH + '/parameter/expertise.pt', map_location=self.device))
        #self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()

    def take_action(self, state):
        state = torch.tensor([[state]], dtype=torch.float).to(self.device)
        action = self.model(state)
        return action[0]

class PolicyNetContinuous(torch.nn.Module):
    def __init__(self):
        super(PolicyNetContinuous, self).__init__()
        self.Conv = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
        self.BN = torch.nn.BatchNorm2d(num_features=1)
        self.Pool = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.NN1 = torch.nn.Linear(1600, 400)
        self.NN2 = torch.nn.Linear(400, 128)
        self.NN3 = torch.nn.Linear(128, 64)
        self.NN4 = torch.nn.Linear(64, 2)


    def forward(self, input):
        x = self.Conv(input)
        x = self.BN(x)
        x = F.relu(x)
        x = self.Pool(x)
        x = F.dropout(x, p=0.2)
        x = torch.flatten(input, start_dim=1)

        x = self.NN1(x)
        x = F.relu(x)
        x = self.NN2(x)
        x = F.relu(x)
        x = self.NN3(x)
        x = F.relu(x)
        x = self.NN4(x)

        x[:, 0] = torch.tanh(x[:, 0])*150 + 50
        x[:, 1] = torch.tanh(x[:, 1])*30

        return x


class ValueNet(torch.nn.Module):
    def __init__(self):
        super(ValueNet, self).__init__()
        self.Conv = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
        self.BN = torch.nn.BatchNorm2d(num_features=1)
        self.Pool = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.NN1 = torch.nn.Linear(1600, 400)
        self.NN2 = torch.nn.Linear(400, 128)
        self.NN3 = torch.nn.Linear(128, 64)
        self.NN4 = torch.nn.Linear(64, 1)


    def forward(self, input):
        x = self.Conv(input)
        x = self.BN(x)
        x = F.relu(x)
        x = self.Pool(x)
        x = F.dropout(x, p=0.2)
        x = torch.flatten(input, start_dim=1)

        x = self.NN1(x)
        x = F.relu(x)
        x = self.NN2(x)
        x = F.relu(x)
        x = self.NN3(x)
        x = F.relu(x)
        x = self.NN4(x)

        return x

def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)

class PPOContinuous:
    ''' 处理连续动作的PPO算法 '''
    def __init__(self, actor_lr, critic_lr, lmbda, epochs, eps, gamma, device):
        self.actor = PolicyNetContinuous().to(device)
        self.critic = ValueNet().to(device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.device = device

    def take_action(self, state):
        state = torch.tensor([[state]], dtype=torch.float).to(self.device)
        action = self.actor(state)
        return action[0]

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)
        rewards = (rewards + 8.0) / 8.0  # 和TRPO一样,对奖励进行修改,方便训练
        td_target = rewards + self.gamma * self.critic(next_states)[0] * (1 -
                                                                       dones)
        td_delta = td_target - self.critic(states)[0]
        advantage = compute_advantage(td_delta.cpu()).to(self.device)
        mu, std = self.actor(states)
        action_dists = torch.distributions.Normal(mu.detach(), std.detach())
        # 动作是正态分布
        old_log_probs = action_dists.log_prob(actions)

        for _ in range(self.epochs):
            mu, std = self.actor(states)
            action_dists = torch.distributions.Normal(mu, std)
            log_probs = action_dists.log_prob(actions)
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            critic_loss = torch.mean(
                F.mse_loss(self.critic(states), td_target.detach()))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

    def compute_advantage(self, td_delta):
        td_delta = td_delta.detach().numpy()
        advantage_list = []
        advantage = 0.0
        for delta in td_delta[::-1]:
            advantage = self.gamma * self.lmbda * advantage + delta
            advantage_list.append(advantage)
        advantage_list.reverse()
        return torch.tensor(advantage_list, dtype=torch.float)

def train_on_policy_agent(env, new_agent, expertise_agent, num_episodes):
    #team_0 是训练的智能体，team_1 是陪练的智能体
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                state = env.reset()
                done = False
                RENDER = False
                while not done:

                    action_team_0 = new_agent.take_action(state[0]['agent_obs'])
                    action_team_1 = expertise_agent.take_action(state[1]['agent_obs'])

                    next_state, reward, done, _ = env.step([action_team_0, action_team_1])
                    state = next_state
                    episode_return += reward[0]
                return_list.append(episode_return)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list

actor_lr = 1e-4
critic_lr = 5e-3
num_episodes = 2000
hidden_dim = 128
gamma = 0.9
lmbda = 0.9
epochs = 10
eps = 0.2
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")

envmap =  create_scenario('wrestling')        #load map config
env = wrestling(envmap)


agent = PPOContinuous(actor_lr, critic_lr, lmbda, epochs, eps, gamma, device)
expert_agent = Expertise(device)


return_list = train_on_policy_agent(env, agent, expert_agent, num_episodes)

episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('PPO on {}'.format('wrestling'))
plt.savefig(CURRENT_PATH+"try.png")