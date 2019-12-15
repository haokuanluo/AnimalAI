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

Conv_W = 3
PAD = 2 * (int((Conv_W -1) / 2), )
EMB_DIM = 32
final_dim = 1152
if device == 'cuda':
    torch.cuda.set_device(1)



class random_project(nn.Module):
    def __init__(self, num_inputs, gpu_id = 1):
        super(random_project, self).__init__()

        self.conv1 = nn.Conv2d(num_inputs, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(32)
        self.gpu_id = gpu_id

        self.random_project = nn.Linear(final_dim, 256)


    def forward(self, x):
        #x, (hx, cx) = x
        # x = x.permute(0,3,1,2)
        # x.unsqueeze_(0)

        # x = x.float() / 255
        with torch.no_grad():
            x = F.relu(self.bn1(self.conv1(x)))
            # x = F.max_pool2d(x, 2, 2)
            x = F.relu(self.bn2(self.conv2(x)))
            # x = F.max_pool2d(x, 2, 2)
            x = F.relu(self.bn3(self.conv3(x)))
            x = F.relu(self.bn4(self.conv4(x)))
            x = x.view(x.size(0), -1)

            x = self.random_project(x)
            return x



class forward(nn.Module):
    def __init__(self, num_inputs, gpu_id = 1):
        super(forward, self).__init__()

        self.conv1 = nn.Conv2d(num_inputs, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(32)
        self.gpu_id = gpu_id

        self.fc = nn.Linear(final_dim,256)
        self.embed_phi = nn.Linear(256, 256)
        self.obstacle_predict = nn.Linear(256, 1)
        # The actor layer
        self.reward_predict = nn.Linear(256, 1)



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
        phi = self.embed_phi(x)
        obs = F.sigmoid(self.obstacle_predict(x))
        reward = F.sigmoid(self.reward_predict(x))

        return phi,obs,reward




class union(nn.Module):  # old -> new
    def __init__(self):
        super(union, self).__init__()

    def forward(self, cfold, cf):

        c, f = cf  # both 4 D tensor
        cold,fold = cfold
        if cold is None:
            loss = 0
            return cf,loss

        newf = (fold*cold + f*c)/(cold+c+0.0001)
        newc = cold + c

        #calculate loss
        loss = c.detach()*((fold-f.detach())**2) + cold * c.detach()*((fold.detach()-f.detach())**2)# this doesn't balance the value of c
        loss = loss - cold * (c.detach()*((fold.detach()-f.detach())**2)).mean()
        return (newc,newf),loss



class transform(nn.Module):   # old -> new
    def __init__(self):
        super(transform, self).__init__()

    def forward(self,cf,theta,velo):
        c,f = cf  # both 4 D tensor
        #print('ggggggggg',c.size(),f.size())
        csize = c.size()
        # turn, theta in radians

        rotation_matrix = torch.tensor([
            [math.cos(theta), math.sin(-theta), 0],
            [math.sin(theta), math.cos(theta), 0]
        ], dtype=torch.float).to(device)
        grid = F.affine_grid(rotation_matrix.unsqueeze(0), csize).to(device)
        c = F.grid_sample(c, grid,mode='bilinear')
        f = F.grid_sample(f, grid, mode='bilinear')

        #velo

        translation_matrix = torch.tensor([
                [1, 0, velo[0]*0.002],
                [0, 1, -velo[2]*0.002]
            ], dtype=torch.float).to(device)
        grid = F.affine_grid(translation_matrix.unsqueeze(0), csize)
        c = F.grid_sample(c, grid, mode='bilinear')
        f = F.grid_sample(f, grid, mode='bilinear')
        return (c,f)

class phi(nn.Module):
    def __init__(self, n_chan,variational = False,CC=32,LL=21,WW = 21):
        super(phi, self).__init__()
        # 1 channel input to 2 channel output of first time print and written
        self.CC,self.LL,self.WW = CC,LL,WW
        self.conv1 = nn.Conv2d(n_chan, 8, Conv_W, padding=PAD)
        self.conv2 = nn.Conv2d(8, 16, Conv_W, padding=PAD)
        self.conv3 = nn.Conv2d(16, 32, Conv_W, padding=PAD)

        self.dense_enc = nn.Linear(CC * LL * WW, 100)

        # variational bits
        self.fc_mu = nn.Linear(100, EMB_DIM)
        self.fc_logvar = nn.Linear(100, EMB_DIM)

        self.dense_dec = nn.Linear(EMB_DIM, CC * LL * WW)

        self.deconv3 = torch.nn.ConvTranspose2d(32, 16, Conv_W, padding=PAD)
        self.deconv2 = torch.nn.ConvTranspose2d(16, 8, Conv_W, padding=PAD)
        self.deconv1 = torch.nn.ConvTranspose2d(8, 2, Conv_W, padding=PAD)

        self.pool = nn.MaxPool2d(2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2)

        self.variational = variational


    def forward(self, x):
        # conv1
        x = F.relu(self.conv1(x))
        size1 = x.size()
        # x, idx1 = self.pool(x)

        # conv2
        x = F.relu(self.conv2(x))
        size2 = x.size()
        x, idx2 = self.pool(x)
        # print('size2=',size2)

        # conv3
        x = F.relu(self.conv3(x))
        size3 = x.size()
        x, idx3 = self.pool(x)

        # =================================================
        # reached the middle layer, some dense
        x = x.view(-1, self.CC * self.LL * self.WW)
        # x = x.view(-1,8*60*60)
        x = torch.relu(self.dense_enc(x))

        mu  = self.fc_mu(x)
        if self.variational:
            logvar = self.fc_logvar(x)
            x = self.reparameterize(mu, logvar)
        else:
            x = mu
        x = F.relu(self.dense_dec(x))
        x = x.view(-1, self.CC, self.LL, self.WW)
        # =================================================

        # deconv3
        x = self.unpool(x, idx3, size3)
        x = F.relu(self.deconv3(x))

        # deconv2
        x = self.unpool(x, idx2, size2)
        x = F.relu(self.deconv2(x))

        # deconv1
        # x = self.unpool(x, idx1, size1)
        x = self.deconv1(x)
        # x = torch.sigmoid(self.deconv1(x))
        return F.relu(x[:,0,:,:]).unsqueeze(0),x[:,1,:,:].unsqueeze(0)

    # def decode(self, x):



    # VAE MAGIC =================

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def learn_once(self, imgs):
        img_rec, mu, logvar = self(imgs)

        self.opt.zero_grad()

        L2_LOSS = ((img_rec - imgs) ** 2).mean()
        KLD_LOSS = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # print (L2_LOSS, KLD_LOSS)
        loss = L2_LOSS + KLD_LOSS * 0.01
        loss.backward()

        for param in self.parameters():
            param.grad.data.clamp_(-1, 1)

        self.opt.step()
        return loss


final_dim = 1152
k = 34
f = 32
d = 6
k_ = 32
w = 8

