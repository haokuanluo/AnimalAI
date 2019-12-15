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
from model import Policy, Inverse, mapping, prediction
import copy
from PIL import Image
import torchvision.transforms as T









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


resiz = T.Compose([T.ToPILImage(),
                    T.Resize((80,80)),
                    T.ToTensor()])


class atari_env(object):
    def __init__(self,env_id,rank=0,save_img = False):
        self.env = gym.make('gym_tdw:tdw_puzzle_1_proc-v0', no_of_cubes=15, pro_gen_puzzle_no=2)
        self.env.set_observation(True)
        self.observation_space = np.array([1,2,3])
        self.total_step = 0
        self.rank = rank
        self.ball = None
        self.first_frame = None
        self.img = []
        self.save_img = save_img

    def transform_action(self,action):
        action = action[0][0]
        multiplier = 20.0
        return {'x': (action // grid) * multiplier - 20.0, 'z': (action % grid) * multiplier - 20.0}

    def aux_reward(self,state):

        return 0


    def step(self,action):


        action = self.transform_action(action)
        self.env.step(action)

        for step in range(window):
            a,b,c,d = self.env.step({'x':0,'z':0})   # observation, reward, done, info
            if c:     # if done == True or we have a reward

                break




        self.total_step = self.total_step + 1
        if self.total_step%10 == 0:
            print("reward, action, done, total_step, thread is ",b,action,c,self.total_step,self.rank)

        if c and b<1:   # extra punishment for jumping out of boundary   #
            b = b - 0.000001
        a = a['image']
        if self.save_img:
            self.img.append(a)
            if self.total_step % 50 == 0:
                pickle.dump(self.img, open('imgg.p', 'wb'))


        a = a.transpose(2,0,1)
        with torch.no_grad():
            a = resiz(a).unsqueeze(0).float().to(device)
        if self.first_frame is None:
            self.first_frame = copy.deepcopy(a).to(device)

        return a,b,c,d

    def reset(self):
        self.env.reset()
        if self.first_frame is not None:
            return self.first_frame
        with torch.no_grad():
            a = torch.zeros(1, 3,80,80).float().to(device)

        return a

    def render(self):
        self.env.render()

    def seed(self,seed):
        pass
        #self.env.seed(seed)

    def close(self):
        self.env.close()






class Agent(object):
    def __init__(self, model, env, args, state,inverse_model,mapping_model,prediction_model, gpu_id = 1):
        self.model = model
        self.inverse_model = inverse_model
        self.mapping_model = mapping_model
        self.prediction_model = prediction_model
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
        self.i_rewards = []
        self.entropies = []

        self.actions = []
        self.one_hot_actions = []
        self.next_states = []

        self.old_state = []
        self.old_actions = []
        self.done = True
        self.info = None
        self.reward = 0
        self.loss_3 = 0
        self.loss_5 = 0
        self.gpu_id = gpu_id
        self.old_model = None

    def save_old_model(self):
        if self.old_model is None:
            self.old_model = copy.deepcopy(self.model)

        with torch.cuda.device(1):
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
        with torch.cuda.device(self.gpu_id):
            value, logit, (self.hx, self.cx) = self.old_model((Variable(self.state), (self.hx, self.cx)))
        prob = F.softmax(logit)
        log_prob = F.log_softmax(logit)
        entropy = -(log_prob * prob).sum(1)
        self.entropies.append(entropy)
        action = prob.multinomial(num_samples = 1).data
        self.old_actions.append(Variable(action))
        log_prob = log_prob.gather(1, Variable(action))
        state, self.reward, self.done, self.info = self.env.step(action.cpu().numpy())
        self.old_state.append(self.state[0])
        self.next_states.append(state[0])
        self.actions.append(action[0][0])


        one_hot_a = torch.zeros(action_space).float().to(device)
        one_hot_a[action.cpu().numpy()] = 1

        self.one_hot_actions.append(one_hot_a)



        phi_states = self.inverse_model(self.state)
        phi_next_states = self.inverse_model(state)
        concat = torch.cat([phi_states, phi_next_states], dim=1)
        acs = self.mapping_model(concat)
        acs.to(device)
        player_actions = action[0]

        loss_3 = F.nll_loss(acs, player_actions)
        self.loss_3 = self.loss_3 + loss_3
        oha = one_hot_a.unsqueeze(0)
        ca = torch.cat([phi_states, oha], dim=1)

        predicted_states = self.prediction_model(ca)
        mse = nn.MSELoss()

        loss_5 = mse(predicted_states, phi_next_states)
        self.loss_5 = self.loss_5 + loss_5
        self.i_rewards.append(float(loss_5.detach().cpu().numpy()))



        self.eps_len += 1
        self.done = self.done or self.eps_len >= self.args['M']
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.rewards.append(self.reward)
        self.state = state

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
        state = state.unsqueeze(0)
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
        del self.i_rewards
        del self.actions
        del self.one_hot_actions
        del self.next_states

        self.old_actions = []
        self.old_state = []
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        self.i_rewards = []
        self.loss_3 = 0
        self.loss_5 = 0

        self.actions = []
        self.one_hot_actions = []
        self.next_states = []


        return self
loss = []



def train(args, optimizer, rank, shared_model,step_loss,inverse_model,mapping_model,prediction_model):

    print("start training thread ",rank)
    gpu_id = args['gpu_ids'][rank]

    torch.manual_seed(args['seed'] + rank)
    torch.cuda.manual_seed(args['seed'] + rank)
    env = atari_env(args['ENV'],rank=rank)

    value_weight = 1

    if True:
        env.seed(args['seed'] + rank)
        if optimizer == None:
            optimizer = optim.Adam(shared_model.parameters(), lr=args['LR'])
            params = list(list(inverse_model.parameters()) + list(
                mapping_model.parameters()) + list(prediction_model.parameters()))
            optimizer_curiosity = optim.Adam(params, lr=args['LR'])

        player = Agent(None, env, args, None,inverse_model,mapping_model,prediction_model, gpu_id)
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
                rewards.insert(0, discounted_reward)   # Note that the intri R is also multiplied by 20

            rewards = torch.tensor(rewards).to(device)
            rewards =  rewards * 20
            player.i_rewards = np.array(player.i_rewards)
            player.i_rewards = (player.i_rewards-np.mean(player.i_rewards))/(np.std(player.i_rewards)+0.0001)/8.0
            #player.i_rewards = player.i_rewards * 5
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
                    advantages = advantages + max(min(player.i_rewards[i],0.125),-0.125)
                    surr1 = ratios * advantages
                    surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * advantages
                    policy_loss = policy_loss -torch.min(surr1, surr2)  - 0.03 * dist_entropy
                    value_loss = value_loss + nn.MSELoss()(state_values, rewards[i])


                policy_wei = 1
                value_wei = 0.5

                if kk == epoch-1:
                    print("reward_sum, episode length, average reward, thread: ",reward_sum, len(player.rewards), reward_sum / len(player.rewards), rank)
                    print("policy loss, value loss",policy_loss, value_loss)
                    step_loss = pickle.load(open('TDW_curiosity_PPO_step_loss_reward_sum.p', 'rb'))
                    step_loss.append((steps, reward_sum, reward_sum / len(player.rewards),
                                      policy_loss.cpu().detach().numpy()[0][0],
                                      float(value_loss.cpu().detach().numpy()),float(player.loss_3.cpu().detach().numpy())/len(player.rewards),float(player.loss_5.cpu().detach().numpy())/len(player.rewards)))
                    print("summary of step loss",step_loss[-50:-1])
                    pickle.dump(step_loss, open('TDW_curiosity_PPO_step_loss_reward_sum.p', 'wb'))

                optimizer.zero_grad()
                (policy_wei * policy_loss + value_wei * value_loss).backward()

                #with torch.cuda.device(gpu_id):
                #    torch.cuda.synchronize()

                ensure_shared_grads(player.model, shared_model)
                optimizer.step()
                # new/old/shared model's problem
                player.model.load_state_dict(shared_model.state_dict())
                player.done = True
            epoch2 = 10
            states = torch.stack(player.old_state)
            next_states = torch.stack(player.next_states)
            player_actions = torch.stack(player.actions)
            player_actions.to(device)
            player.one_hot_actions = torch.stack(player.one_hot_actions)

            for kk in range(epoch2):
                # loss 3


                # print("stacked states shape",states.shape)


                phi_states = inverse_model(states)
                phi_next_states = inverse_model(next_states)
                # print(phi_states.shape)
                concat = torch.cat([phi_states, phi_next_states], dim=1)
                actions = mapping_model(concat)
                actions.to(device)


                loss_3 = F.nll_loss(actions, player_actions)

                concat_actions = torch.cat([phi_states, player.one_hot_actions.to(device)], dim=1)
                # print('concat action shape',concat_actions.shape)

                predicted_states = prediction_model(concat_actions)
                mse = nn.MSELoss()

                loss_5 = mse(predicted_states, phi_next_states)
                optimizer_curiosity.zero_grad()
                (0.8 * loss_3 + 0.2 * loss_5).backward()
                optimizer_curiosity.step()

            #####
            optimizer_curiosity.zero_grad()
            (0.8 * player.loss_3 + 0.2 * player.loss_5).backward()
            optimizer_curiosity.step()

            #####
            player.clear_actions()
            torch.save(shared_model.state_dict(), 'tdw_ppo_curiosity_model')
            torch.save(inverse_model.state_dict(),'tdw_ppo_curiosity_model_inverse')
            torch.save(mapping_model.state_dict(), 'tdw_ppo_curiosity_model_mapping')
            torch.save(prediction_model.state_dict(), 'tdw_ppo_curiosity_model_prediction')
            #####


def loadarguments():
    global env_conf
    global env
    global setup_json
    global shared_model
    global inverse_model
    global mapping_model
    global prediction_model
    global saved_state
    global optimizer
    global torch
    global step_loss

    #step_loss = pickle.load(open('TDW_A2C_step_loss_reward_sum.p', 'rb'))
    step_loss = []
    if args['L'] == False:
        pickle.dump(step_loss, open('TDW_curiosity_PPO_step_loss_reward_sum.p', 'wb'))

    torch.set_default_tensor_type('torch.FloatTensor')
    torch.manual_seed(args['seed'])

    the_gpu_id = 1
    shared_model = Policy(3, action_space)

    final_dim = 800

    inverse_model = Inverse(3, action_space)
    mapping_model = mapping(action_space, final_dim)
    prediction_model = prediction(action_space, final_dim)







    if device == 'cuda' and True:
        with torch.cuda.device(the_gpu_id):
            shared_model.cuda()
            inverse_model.cuda()
            mapping_model.cuda()
            prediction_model.cuda()



    shared_model.share_memory()
    inverse_model.share_memory()
    mapping_model.share_memory()
    prediction_model.share_memory()

    if args['L']:
        shared_model.load_state_dict(torch.load('tdw_ppo_curiosity_model'))
        inverse_model.load_state_dict(torch.load('tdw_ppo_curiosity_model_inverse'))
        mapping_model.load_state_dict(torch.load('tdw_ppo_curiosity_model_mapping'))
        prediction_model.load_state_dict(torch.load('tdw_ppo_curiosity_model_prediction'))

    optimizer = None

args = {'LR': 0.01, "G":0.87, "T":1.00,"NS":100,"M":100,'W':6,   ###############
         "seed":92,'LMD':'/modeldata/','SMD':'/modeldata/','ENV':'gym_tdw:tdw_puzzle_7-v0','L':False,'SO':False,'OPT':'Adam',
        'gpu_ids':[1,1,1,1,1,1,1],'mr':1}

if __name__ == '__main__':
    torch.backends.cudnn.benchmark = False
    processes = []
    loadarguments()
    torch.manual_seed(args['seed'])
    torch.cuda.manual_seed(args['seed'])
    mp.set_start_method('spawn')
    for rank in range(0, args['W']):
        p = Process(
            target=train, args=(args, optimizer, rank, shared_model,step_loss,inverse_model,mapping_model,prediction_model))
        p.start()
        processes.append(p)
        time.sleep(10)
    for p in processes:
        p.join()
        time.sleep(10)
