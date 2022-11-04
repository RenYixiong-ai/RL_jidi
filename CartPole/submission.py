import os
from pathlib import Path
import torch
from torch.nn import functional as F
from torch.distributions import Categorical

import numpy as np

current_path = Path(__file__).resolve().parent
model_path = os.path.join(current_path, 'actor.pt')

class Actor(torch.nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = torch.nn.Linear(self.state_size, 128)
        self.linear2 = torch.nn.Linear(128, 256)
        self.linear3 = torch.nn.Linear(256, self.action_size)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        output = self.linear3(output)
        distribution = Categorical(F.softmax(output))
        return distribution

model = Actor(4, 2)
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()

def my_controller(observation, action_space, is_act_continuous=True):

    obs_array = torch.tensor(observation['obs'], dtype=torch.float32).view(1, -1)
    action = model(obs_array).sample()

    each = [0] * 2
    each[int(action.item())] = 1
    return [each]