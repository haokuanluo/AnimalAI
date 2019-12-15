import torchvision.utils as vutils
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import time
import copy
import math
from torch.autograd import Variable

import pickle
from scipy.spatial import distance

device = 'cuda' if torch.cuda.is_available() else 'cpu'
final_dim = 1152
k = 34
f = 32
d = 6
k_ = 32
w = 8



class relation_Policy(nn.Module):
    def __init__(self, num_inputs, action_space, gpu_id=0):
        super(relation_Policy, self).__init__()
        self.gpu_id = gpu_id
        if device == 'cuda':
            torch.cuda.set_device(gpu_id)

        self.conv1 = nn.Conv2d(num_inputs, f, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(f)
        self.conv2 = nn.Conv2d(f, f, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(f)
        self.conv3 = nn.Conv2d(f, f, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(f)
        self.conv4 = nn.Conv2d(f, f, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(f)

        self.q = nn.Linear(k, k_)
        self.k = nn.Linear(k, k_)
        self.v = nn.Linear(k, k_)

        self.MLP1 = nn.Sequential(
            nn.Linear(k_, 32),
            nn.ReLU(),
            nn.Linear(32, k_)
        )

        self.MLP2 = nn.Sequential(
            nn.Linear(k_, 400),
            nn.ReLU(),
            nn.Linear(400, final_dim)
        )

        self.lstm = nn.LSTMCell(final_dim, 256)

        self.critic_linear = nn.Linear(256, 1)
        # The actor layer
        self.actor_linear = nn.Linear(256, action_space)

        self.save_actions = []
        self.rewards = []

    def forward(self, x):
        x, (hx, cx) = x

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        # x now f * 5 * 5
        x = x[0].permute(1, 2, 0)
        x = x.reshape((d * d, f))  # now d*d, f

        pos = torch.zeros((d * d, 2)).float()
        if device == 'cuda':
            pos = pos.cuda()
        id = 0
        for i in range(d):
            for j in range(d):
                pos[id, 0] = (i / d) * 2 - 1.0
                pos[id, 1] = (j / d) * 2 - 1.0
        x = torch.cat([x, pos], dim=1)
        # print(x.shape,'should be ',d*d,f+2)
        q = self.q(x)
        k = self.k(x)
        k = k.permute(1, 0)  # now k is 32 * dd
        v = self.v(x)
        # each of them d*d * k_ (32)

        # multi-head part
        div = math.sqrt(k_)
        A1 = torch.mm(F.softmax(torch.mm(q[:, :w], k[:w, :]) / div, dim=1), v[:, :w])  # dd*dd dd*32
        A2 = torch.mm(F.softmax(torch.mm(q[:, w:w * 2], k[w:w * 2, :]) / div), v[:, w:w * 2])
        A3 = torch.mm(F.softmax(torch.mm(q[:, w * 2:w * 3], k[w * 2:w * 3, :]) / div), v[:, w * 2:w * 3])
        A4 = torch.mm(F.softmax(torch.mm(q[:, w * 3:w * 4], k[w * 3:w * 4, :]) / div), v[:, w * 3:w * 4])
        # each of them dim dd * w
        A = torch.cat([A1, A2, A3, A4], dim=1)
        # print(A.shape,'should be',d*d,w*4)
        # A is dd * k_ (32)
        A = self.MLP1(A)

        A, _ = torch.max(A, 0)  # now A is k_ (1 D)
        A = A.unsqueeze(0)
        x = self.MLP2(A)

        # x = x.view(x.size(0), -1)

        hx, cx = self.lstm(x, (hx, cx))
        x = hx

        action_score = self.actor_linear(x)
        state_value = self.critic_linear(x)

        return state_value, action_score, (hx, cx)


class shared_parameter_Policy(nn.Module):
    def __init__(self, num_inputs, action_space, gpu_id=0):
        super(shared_parameter_Policy, self).__init__()
        self.gpu_id = gpu_id
        if device == 'cuda':
            torch.cuda.set_device(gpu_id)



        self.q = nn.Linear(k, k_)
        self.k = nn.Linear(k, k_)
        self.v = nn.Linear(k, k_)

        self.MLP1 = nn.Sequential(
            nn.Linear(k_, 32),
            nn.ReLU(),
            nn.Linear(32, k_)
        )

        self.MLP2 = nn.Sequential(
            nn.Linear(k_, 400),
            nn.ReLU(),
            nn.Linear(400, final_dim)
        )

        self.lstm = nn.LSTMCell(final_dim, 256)

        self.critic_linear = nn.Linear(256, 1)
        # The actor layer
        self.actor_linear = nn.Linear(256, action_space)

        self.save_actions = []
        self.rewards = []

    def forward(self, x):
        x, (hx, cx) = x


        # x now f * 5 * 5
        x = x[0].permute(1, 2, 0)
        x = x.reshape((d * d, f))  # now d*d, f

        pos = torch.zeros((d * d, 2)).float()
        if device == 'cuda':
            pos = pos.cuda()
        id = 0
        for i in range(d):
            for j in range(d):
                pos[id, 0] = (i / d) * 2 - 1.0
                pos[id, 1] = (j / d) * 2 - 1.0
        x = torch.cat([x, pos], dim=1)
        # print(x.shape,'should be ',d*d,f+2)
        q = self.q(x)
        k = self.k(x)
        k = k.permute(1, 0)  # now k is 32 * dd
        v = self.v(x)
        # each of them d*d * k_ (32)

        # multi-head part
        div = math.sqrt(k_)
        A1 = torch.mm(F.softmax(torch.mm(q[:, :w], k[:w, :]) / div, dim=1), v[:, :w])  # dd*dd dd*32
        A2 = torch.mm(F.softmax(torch.mm(q[:, w:w * 2], k[w:w * 2, :]) / div), v[:, w:w * 2])
        A3 = torch.mm(F.softmax(torch.mm(q[:, w * 2:w * 3], k[w * 2:w * 3, :]) / div), v[:, w * 2:w * 3])
        A4 = torch.mm(F.softmax(torch.mm(q[:, w * 3:w * 4], k[w * 3:w * 4, :]) / div), v[:, w * 3:w * 4])
        # each of them dim dd * w
        A = torch.cat([A1, A2, A3, A4], dim=1)
        # print(A.shape,'should be',d*d,w*4)
        # A is dd * k_ (32)
        A = self.MLP1(A)

        A, _ = torch.max(A, 0)  # now A is k_ (1 D)
        A = A.unsqueeze(0)
        x = self.MLP2(A)

        # x = x.view(x.size(0), -1)

        hx, cx = self.lstm(x, (hx, cx))
        x = hx

        action_score = self.actor_linear(x)
        state_value = self.critic_linear(x)

        return state_value, action_score, (hx, cx)


class h_Policy(nn.Module):
    def __init__(self, num_inputs, action_space,gpu_id = 1):
        super(h_Policy, self).__init__()

        self.conv1 = nn.Conv2d(num_inputs, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(32)
        self.gpu_id = gpu_id
        self.lstm = nn.LSTMCell(final_dim, 256)
        if device == 'cuda':
            with torch.cuda.device(self.gpu_id):
                self.cx = Variable(torch.zeros(1, 256).float().cuda())
                self.hx = Variable(torch.zeros(1, 256).float().cuda())
        else:
            self.cx = Variable(torch.zeros(1, 256).float())
            self.hx = Variable(torch.zeros(1, 256).float())

        self.critic_linear = nn.Linear(257, 1)
        # The actor layer
        self.actor_linear = nn.Linear(257, action_space)

        self.save_actions = []
        self.rewards = []

    def forward(self, x):

        x,v = x  # v is a 2d tensor ([1,1])

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
        self.hx, self.cx = self.lstm(x, (self.hx, self.cx))
        x = self.hx
        x = torch.cat([x,v],dim = 1)
        action_score = self.actor_linear(x)
        state_value = self.critic_linear(x)

        return state_value, action_score

    def reset(self):
        if device == 'cuda':
            with torch.cuda.device(self.gpu_id):
                self.cx = Variable(torch.zeros(1, 256).float().cuda())
                self.hx = Variable(torch.zeros(1, 256).float().cuda())
        else:
            self.cx = Variable(torch.zeros(1, 256).float())
            self.hx = Variable(torch.zeros(1, 256).float())


class nolstm_Policy(nn.Module):
    def __init__(self, num_inputs, action_space,gpu_id = 1):
        super(nolstm_Policy, self).__init__()

        self.conv1 = nn.Conv2d(num_inputs, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(32)
        self.gpu_id = gpu_id
        self.fc = nn.Linear(final_dim, 256)


        self.critic_linear = nn.Linear(256, 1)
        # The actor layer
        self.actor_linear = nn.Linear(256, action_space)

        self.save_actions = []
        self.rewards = []

    def forward(self, x):
        #x, (hx, cx) = x
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

        x = F.relu(self.fc(x))
        action_score = self.actor_linear(x)
        state_value = self.critic_linear(x)

        return state_value, action_score

    def reset(self):
        if device == 'cuda':
            with torch.cuda.device(self.gpu_id):
                self.cx = Variable(torch.zeros(1, 256).float().cuda())
                self.hx = Variable(torch.zeros(1, 256).float().cuda())
        else:
            self.cx = Variable(torch.zeros(1, 256).float())
            self.hx = Variable(torch.zeros(1, 256).float())


class Policy(nn.Module):
    def __init__(self, num_inputs, action_space,gpu_id = 1):
        super(Policy, self).__init__()

        self.conv1 = nn.Conv2d(num_inputs, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(32)
        self.gpu_id = gpu_id
        self.lstm = nn.LSTMCell(final_dim, 256)
        if device == 'cuda':
            with torch.cuda.device(self.gpu_id):
                self.cx = Variable(torch.zeros(1, 256).float().cuda())
                self.hx = Variable(torch.zeros(1, 256).float().cuda())
        else:
            self.cx = Variable(torch.zeros(1, 256).float())
            self.hx = Variable(torch.zeros(1, 256).float())

        self.critic_linear = nn.Linear(256, 1)
        # The actor layer
        self.actor_linear = nn.Linear(256, action_space)

        self.save_actions = []
        self.rewards = []

    def forward(self, x):
        #x, (hx, cx) = x
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

        self.hx, self.cx = self.lstm(x, (self.hx, self.cx))
        x = self.hx
        action_score = self.actor_linear(x)
        state_value = self.critic_linear(x)

        return state_value, action_score

    def reset(self):
        if device == 'cuda':
            with torch.cuda.device(self.gpu_id):
                self.cx = Variable(torch.zeros(1, 256).float().cuda())
                self.hx = Variable(torch.zeros(1, 256).float().cuda())
        else:
            self.cx = Variable(torch.zeros(1, 256).float())
            self.hx = Variable(torch.zeros(1, 256).float())


class Inverse(nn.Module):
    def __init__(self, num_inputs, action_space):
        super(Inverse, self).__init__()

        self.conv1 = nn.Conv2d(num_inputs, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(32)
        self.action_space = action_space

        # self.embed = nn.Linear(288*2, 256)
        # self.output = nn.Linear(256,action_space)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        #return x
        x = x.view(x.size(0), -1)

        # x = F.relu(self.embed(x))
        # x = self.output(x)
        return x


class mapping(nn.Module):
    def __init__(self, action_space, final_dim):
        super(mapping, self).__init__()

        self.embed = nn.Linear(final_dim * 2, 256)
        self.output = nn.Linear(256, action_space)

    def forward(self, x):
        x = F.relu(self.embed(x))
        x = self.output(x)
        return F.log_softmax(x, dim=1)


class prediction(nn.Module):
    def __init__(self, action_space, final_dim):
        super(prediction, self).__init__()

        self.f1 = nn.Linear(final_dim + action_space, 256)
        self.f2 = nn.Linear(256, final_dim)

    def forward(self, x):
        x = F.relu(self.f1(x))
        x = self.f2(x)
        return x

