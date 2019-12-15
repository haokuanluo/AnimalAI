import yaml
from animalai.envs.brain import BrainParameters

import gym, os
import random
import time
import numpy as np
import math
import torch
import torch.nn as nn

import torch.nn.functional as F

from torch.autograd import Variable


from animalai.envs.gym.environment import AnimalAIEnv
from animalai.envs.arena_config import ArenaConfig

class Policy(nn.Module):
    def __init__(self, num_inputs, action_space):
        super(Policy, self).__init__()

        self.conv1 = nn.Conv2d(num_inputs, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(32)

        self.lstm = nn.LSTMCell( 1152, 256)

        self.critic_linear = nn.Linear(256, 1)
        # The actor layer
        self.actor_linear = nn.Linear(256, action_space)

        self.save_actions = []
        self.rewards = []

    def forward(self, x):
        x, (hx, cx) = x

        # x = x.permute(0,3,1,2)
        # x.unsqueeze_(0)

        # x = x.float() / 255

        x = F.relu(self.bn1(self.conv1(x)))
        # x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        # x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = x.view(x.size(0), -1)

        hx, cx = self.lstm(x, (hx, cx))
        x = hx

        action_score = self.actor_linear(x)
        state_value = self.critic_linear(x)

        return state_value, action_score, (hx, cx)

class Agent(object):

    def __init__(self):
        """
         Load your agent here and initialize anything needed
        """
        # You can specify the resolution your agent takes as input, for example set resolution=128 to
        # have visual inputs of size 128*128*3 (if this attribute is omitted it defaults to 84)
        self.args = {'mn': '/aaio/data/animal_a3c_1',
                     'action_space': 3,
                     }
        self.resolution = 84
        
        # Load the configuration and model using ABSOLUTE PATHS
        self.model_path = '/aaio/data/1-Food/Learner'

        self.model = Policy(3, 3)
        self.model.load_state_dict(torch.load(self.args['mn'],map_location = torch.device('cpu')))

        self.state = None
        self.hx = Variable(torch.zeros(1, 256).float())
        self.cx = Variable(torch.zeros(1, 256).float())

    def reset(self, t=250):
        """
        Reset is called before each episode begins
        Leave blank if nothing needs to happen there
        :param t the number of timesteps in the episode
        """
        self.hx = Variable(torch.zeros(1, 256).float())
        self.cx = Variable(torch.zeros(1, 256).float())

    def transform_action(self,action):
        action = action[0][0]
        if action == 0:
            return [1,0]
        elif action == 1:
            return [0,1]
        else:
            return [0,2]

    def step(self, obs, reward, done, info):
        """
        :param obs: agent's observation of the current environment
        :param reward: amount of reward returned after previous action
        :param done: whether the episode has ended.
        :param info: contains auxiliary diagnostic information, including BrainInfo.
        :return: the action to take, a list or size 2
        """

        state = obs[0]

        state = state.transpose(2, 0, 1)
        with torch.no_grad():
            state = torch.tensor(state).unsqueeze(0).float()
        value, logit, (self.hx, self.cx) = self.model((Variable(state), (self.hx, self.cx)))
        self.cx = Variable(self.cx.data)
        self.hx = Variable(self.hx.data)
        prob = F.softmax(logit)
        action = prob.multinomial(num_samples=1).data
        action = self.transform_action( action.cpu().numpy())


        return action
