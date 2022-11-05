import os
from pathlib import Path
import torch
from torch.nn import functional as F

import numpy as np

current_path = Path(__file__).resolve().parent
model_path = os.path.join(current_path, 'paramet.pt')

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.Conv = torch.nn.Conv2d(in_channels=1, out_channels=1 ,kernel_size=3, padding=1)
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


model = CNN()
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()

def my_controller(observation, action_space, is_act_continuous=True):

    obs_array = torch.tensor([[observation['obs']['agent_obs']]]).float()
    action = model(obs_array)

    return [[action[0][0].item()], [action[0][1].item()]]

