import gym, os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import time
import pickle

from torch.autograd import Variable

from torch.multiprocessing import Process
import torch.multiprocessing as mp
from model import relation_Policy
import copy
from PIL import Image
import torchvision.transforms as T
from animalai.envs.gym.environment import AnimalAIEnv
from animalai.envs.arena_config import ArenaConfig

env_path = '../env/AnimalAI'
worker_id = random.randint(1, 100)
arena_config_in = ArenaConfig('configs/1-Food.yaml')
base_dir = 'models/dopamine'
gin_files = ['configs/rainbow.gin']

def create_env_fn():
    env = AnimalAIEnv(environment_filename=env_path,
                      worker_id=worker_id,
                      n_arenas=1,
                      arenas_configurations=arena_config_in,
                      docker_training=False,
                      retro=False,
                      inference=False,
                      resolution = 80)
    return env







#Hyperparameters

epoch = 4
eps_clip = 0.2

grid = 3
action_space = grid*grid    # 9 action space means we have 8 different directions and an idle state

global total_step
total_step = 0
window = 20

device = 'cuda'
#device = 'cpu'
torch.cuda.set_device(1)



def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None and device == 'cpu':
            return

        shared_param._grad = param.grad


#resiz = T.Compose([T.ToPILImage(),
#                    T.Resize((80,80)),
#                    T.ToTensor()])


class atari_env(object):   # to change env: extra punishment
    def __init__(self,env_id,rank=0,save_img = False):
        self.env = create_env_fn()
        self.observation_space = self.env.observation_space
        self.total_step = 0
        self.rank = rank
        self.ball = None
        self.first_frame = None
        self.img = []
        self.save_img = save_img
        self.env_id = rank
        self.total_it = 0

    def transform_action(self,action):
        action = action[0][0]
        return [action%3,action//3]





    def step(self,action):


    ######### We perform the action once, and let the ball slide on its own for "window" frames
        action = self.transform_action(action)
        a,b,c,d = self.env.step(action)






        self.total_step = self.total_step + 1
        if self.total_step%10 == 0:
            print("reward, action, done, total_step, thread is ",b,action,c,self.total_step,self.env_id)





        a = a[0]

        if self.save_img:
            self.img.append(a)
            if self.total_step % 50 == 0:
                pickle.dump(self.img, open('imgg.p', 'wb'))


        a = a.transpose(2,0,1)

        with torch.no_grad():
            a = torch.tensor(a).unsqueeze(0).float().to(device)
        if self.first_frame is None:
            self.first_frame = copy.deepcopy(a).to(device)
        #print(a)
        return a,b,c,d


    def reset(self):
        a = self.env.reset()[0]
        #print(a.shape)
        a = a.transpose(2, 0, 1)

        with torch.no_grad():
            a = torch.tensor(a).unsqueeze(0).float().to(device)
        return a

    def render(self):
        self.env.render()

    def seed(self,seed):
        
        self.env.seed(seed)

    def close(self):
        self.env.close()






class Agent(object):
    def __init__(self, model, env, args, state,gpu_id = 0):
        self.model = model
        self.env = env
        self.current_life = 0
        self.state = state
        self.hx = None
        self.cx = None
        self.eps_len = 0
        self.args = args
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        self.old_state = []
        self.old_actions = []
        self.done = True
        self.info = None
        self.reward = 0
        self.gpu_id = gpu_id
        self.old_model = None

    def save_old_model(self):
        if self.old_model is None:
            self.old_model = copy.deepcopy(self.model)

        with torch.cuda.device(self.args['gpu_ids'][0]):
            self.old_model = self.old_model.cuda()
            self.old_model.load_state_dict(self.model.state_dict())




    def action_train(self):
        if self.done:
            print('i am done')
            with torch.cuda.device(self.gpu_id):
                self.cx = Variable(torch.zeros(1, 256).float().to(device))
                self.hx = Variable(torch.zeros(1, 256).float().to(device))
        else:

            self.cx = Variable(self.cx.data)
            self.hx = Variable(self.hx.data)
        self.old_state.append(self.state)
        with torch.cuda.device(self.gpu_id):
            value, logit, (self.hx, self.cx) = self.old_model((Variable(self.state), (self.hx, self.cx)))
        prob = F.softmax(logit)
        log_prob = F.log_softmax(logit)
        entropy = -(log_prob * prob).sum(1)
        self.entropies.append(entropy)
        action = prob.multinomial(num_samples = 1).data
        self.old_actions.append(Variable(action))
        log_prob = log_prob.gather(1, Variable(action))
        self.state, self.reward, self.done, self.info = self.env.step(action.cpu().numpy())
        #with torch.cuda.device(self.gpu_id):
        #    self.state = torch.from_numpy(state).float().to(device)
        self.eps_len += 1
        self.done = self.done or self.eps_len >= self.args['M']
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.rewards.append(self.reward)

        return self

    def action_replay(self,state,action):
        if self.done:
            with torch.cuda.device(self.gpu_id):
                self.cx = Variable(torch.zeros(1, 256).float().to(device))
                self.hx = Variable(torch.zeros(1, 256).float().to(device))
            self.done = False
        else:

            self.cx = Variable(self.cx.data)
            self.hx = Variable(self.hx.data)

        with torch.cuda.device(self.gpu_id):
            value, logit, (self.hx, self.cx) = self.model((Variable(state), (self.hx, self.cx)))
        prob = F.softmax(logit)
        log_prob = F.log_softmax(logit)
        entropy = -(log_prob * prob).sum(1)
        log_prob = log_prob.gather(1, Variable(action))
        #self.done = self.done or self.eps_len >= self.args['M']
        return log_prob,value,entropy



    def clear_actions(self):
        del self.values
        del self.log_probs
        del self.rewards
        del self.entropies
        del self.old_state
        del self.old_actions
        self.old_actions = []
        self.old_state = []
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        return self
loss = []



def train(args, optimizer, rank, shared_model,step_loss):

    print("start training thread ",rank)
    gpu_id = args['gpu_ids'][rank]

    torch.manual_seed(args['seed'] + rank)
    torch.cuda.manual_seed(args['seed'] + rank)
    env = atari_env(args['env_ids'],rank=rank)

    value_weight = 1

    if True:
        env.seed(args['seed'] + rank)
        if optimizer == None:
            optimizer = optim.Adam(shared_model.parameters(), lr=args['LR'])

        player = Agent(None, env, args, None, gpu_id)
        player.model = copy.deepcopy(shared_model)
        player.save_old_model()

        with torch.cuda.device(gpu_id):
            player.model = player.model.cuda() if gpu_id >= 0 else player.model

        player.model.train()


        player.state = player.env.reset()

        while True:

            with torch.cuda.device(gpu_id):
                player.model.load_state_dict(shared_model.state_dict())
                player.save_old_model()
            with torch.cuda.device(gpu_id):
                player.model = player.model.cuda() if gpu_id >= 0 else player.model

            steps = 0
            for step in range(args['NS']):
                steps = steps + 1
                player.action_train()

                if player.done:
                    break
            if player.reward < args['mr']:
                steps = args['M']
            if player.done:
                player.eps_len = 0
                player.current_life = 0
                player.state = player.env.reset()

            rewards = []
            discounted_reward = 0
            reward_sum = 0
            for i in reversed(range(len(player.rewards))):
                if i != 0:
                    player.rewards[i] = player.rewards[i] - player.rewards[i-1]
                reward_sum = reward_sum + player.rewards[i]
                discounted_reward = player.rewards[i]+args['G']*discounted_reward
                rewards.insert(0, discounted_reward)

            rewards = torch.tensor(rewards).to(device)
            #rewards =  rewards * 20

            epoch = 4
            for kk in range(epoch):
                player.done = True

                policy_loss = 0
                value_loss = 0

                for i in range(len(player.rewards)):
                    logprobs, state_values, dist_entropy = player.action_replay(player.old_state[i],
                                                                                player.old_actions[i])
                    ratios = torch.exp(logprobs - player.log_probs[i].detach())
                    advantages = rewards[i] - state_values.detach()

                    surr1 = ratios * advantages
                    surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * advantages
                    policy_loss = policy_loss -torch.min(surr1, surr2)  - 0.005 * dist_entropy
                    value_loss = value_loss + nn.MSELoss()(state_values, rewards[i])

                optimizer.zero_grad()
                policy_wei = 1
                value_wei = 0.5

                if kk == epoch-1:
                    print("reward_sum, episode length, average reward, thread: ",reward_sum, len(player.rewards), reward_sum / len(player.rewards), args['env_ids'][rank])
                    print("policy loss, value loss",policy_loss, value_loss)
                    step_loss = pickle.load(open(args['loss'], 'rb'))
                    step_loss.append((steps, reward_sum, reward_sum / len(player.rewards),
                                      policy_loss.cpu().detach().numpy()[0][0],
                                      float(value_loss.cpu().detach().numpy()),player.env.env_id))
                    print("summary of step loss",step_loss[-50:-1])
                    pickle.dump(step_loss, open(args['loss'], 'wb'))

                (policy_wei * policy_loss + value_wei * value_loss).backward()

                #with torch.cuda.device(gpu_id):
                #    torch.cuda.synchronize()

                ensure_shared_grads(player.model, shared_model)
                optimizer.step()
                # new/old/shared model's problem
                player.model.load_state_dict(shared_model.state_dict())
                player.done = True




            player.clear_actions()
            torch.save(shared_model.state_dict(), args['mn'])



def loadarguments():
    global env_conf
    global env
    global setup_json
    global shared_model
    global saved_state
    global optimizer
    global torch
    global step_loss

    #step_loss = pickle.load(open('TDW_A2C_step_loss_reward_sum.p', 'rb'))
    step_loss = []
    if args['L'] == False:
        pickle.dump(step_loss, open(args['loss'], 'wb'))

    torch.set_default_tensor_type('torch.FloatTensor')
    torch.manual_seed(args['seed'])

    the_gpu_id = 1
    shared_model = relation_Policy(3, action_space)
    if device == 'cuda' and True:
        with torch.cuda.device(the_gpu_id):
            shared_model.cuda()
    if args['L']:
        with torch.cuda.device(the_gpu_id):
            shared_model.load_state_dict(torch.load(args['lmn']))

    shared_model.share_memory()

    if args['SO']:
        if args['OPT'] == 'Adam':
            optimizer = None
    else:
        optimizer = None

args = {'LR': 0.0002, "G":0.98, "T":1.00,"NS":200,"M":200,'W':4,   ###############
         "seed":92,'LMD':'/modeldata/','SMD':'/modeldata/','ENV':'gym_tdw:tdw_puzzle_7-v0','L':False,'SO':False,'OPT':'Adam',
        'gpu_ids':[1,1,1,1,1,1,1,1],'env_ids':[0,1,2,3,4,5,6,8],'mr':1,'mn':'tdw_ppo_a3c_model_relation','lmn':'tdw_ppo_a3c_model_relation','loss':'relation_TDW_reward.p'}

if __name__ == '__main__':
    torch.backends.cudnn.benchmark = False
    processes = []
    loadarguments()
    torch.manual_seed(args['seed'])
    torch.cuda.manual_seed(args['seed'])

    mp.set_start_method('spawn')
    #train(args,optimizer,0,shared_model)



    for rank in range(0, args['W']):
        p = Process(
            target=train, args=(args, optimizer, rank, shared_model,step_loss))
        p.start()
        processes.append(p)
        time.sleep(10)
    for p in processes:
        p.join()
        time.sleep(10)
