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
from model import Policy,shared_parameter_Policy, Inverse, mapping, prediction
import copy
from PIL import Image
import torchvision.transforms as T
from animalai.envs.gym.environment import AnimalAIEnv
from animalai.envs.arena_config import ArenaConfig






def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        shared_param._grad = param.grad




class atari_env(object):   # to change env: extra punishment
    def __init__(self,args,rank=0,save_img = False):

        self.total_step = 0
        self.rank = rank
        self.img = []
        self.save_img = save_img
        self.args = args
        self.grid = 3
        self.env_path = '../env/AnimalAI'
        self.worker_id = random.randint(1, 100)
        self.arena_config_in = ArenaConfig('configs/1-Food_my.yaml')
        self.base_dir = 'models/dopamine'
        self.gin_files = ['configs/rainbow.gin']
        self.env = self.create_env_fn()
        self.previous_velocity = [1,1]
        self.observation_space = self.env.observation_space
        self.velocity = {0:[1,0],
                         1:[0.7,0.7],
                         2:[0,1],
                         3:[-0.7,0.7],
                         4:[-1,0],
                         5:[-0.7,-0.7],
                         6:[0,-1],
                         7:[0.7,-0.7],
                         }




    def create_env_fn(self):
        env = AnimalAIEnv(environment_filename=self.env_path,
                          worker_id=self.worker_id,
                          n_arenas=1,
                          arenas_configurations=self.arena_config_in,
                          docker_training=False,
                          retro=False,
                          inference=self.args['inference'],
                          resolution=84)
        return env

    def turn_to_direction(self,action):  # parameter: a torch tensor
        action = action[0][0]
        desired_velocity = self.velocity[action]
        self.previous_velocity = np.array(self.previous_velocity)
        self.previous_velocity = self.previous_velocity / (np.linalg.norm(self.previous_velocity)+0.0001)

        cross_product = desired_velocity[0]*self.previous_velocity[1]-desired_velocity[1]*self.previous_velocity[0]
        angle = np.arcsin(abs(cross_product))
        # takes 80 turns to get pi angle

        if cross_product < 0:
            action = [1,2]
            num_it = int(angle/3.1415926 * 68)
            num_it = max(num_it,1)
            for i in range(num_it):
                if i % 3 == 0:
                    action = [1,2]
                else:
                    action = [0,2]
                state,reward,done,info = self.env.step(action)
                if done or reward > 0:
                    return state,reward,done,info
        else:
            action = [1, 1]
            num_it = int(angle / 3.1415926 * 68)
            num_it = max(num_it,1)
            for i in range(num_it):
                if i % 3 == 0:
                    action = [1,1]
                else:
                    action = [0,1]
                state, reward, done, info = self.env.step(action)
                if done or reward > 0:
                    return state, reward, done, info
        return state,reward,done,info



    def step_(self,action):

        # turn to the right direction
        state,reward,done,info = self.turn_to_direction(action)




        if (not (done or reward > 0)):
            for frames in range(self.args['window']):
                state, reward, done, info = self.env.step([1,0])
                if done or reward>0:
                    break


        self.total_step = self.total_step + 1
        if self.total_step%5000 == 0:
            print("reward, action, done, total_step, thread is ",reward,action,done,self.total_step,self.rank)
        self.previous_velocity = [state[1][0], state[1][2]]
        state = state[0]

        if self.save_img:
            self.img.append(state)
            if self.total_step % 500 == 0:
                pickle.dump(self.img, open('imgg.p', 'wb'))
        state = state.transpose(2,0,1)
        with torch.no_grad():
            state = torch.tensor(state).unsqueeze(0).float().to(device)
        if reward < 0 and reward > -0.1:
            reward = 0
        return state,reward,done,info

    def transform_action(self,action):
        action = action[0][0]
        if action == 0:
            return [1,0]
        elif action == 1:
            return [0,1]
        else:
            return [0,2]
    def step(self,action):

        # turn to the right direction
        action = self.transform_action(action)

        for frames in range(self.args['window']):
            state, reward, done, info = self.env.step(action)
            if done or reward > 0:
                break


        self.total_step = self.total_step + 1
        if self.total_step%500 == 0:
            print("reward, action, done, total_step, thread is ",reward,action,done,self.total_step,self.rank)
        state = state[0]


        state = state.transpose(2,0,1)
        with torch.no_grad():
            state = torch.tensor(state).unsqueeze(0).float().to(device)
        if reward < 0 and reward > -0.1:
            reward = 0
        return state,reward,done,info

    def reset(self):
        state = self.env.reset()[0]
        state = state.transpose(2, 0, 1)

        with torch.no_grad():
            state = torch.tensor(state).unsqueeze(0).float().to(device)
        return state

    def render(self):
        self.env.render()

    def seed(self,seed):

        self.env.seed(seed)

    def close(self):
        self.env.close()






class Agent(object):
    def __init__(self, model, env, args, state,gpu_id = 0,inverse_model=None,mapping_model=None,prediction_model=None):
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
        ##### for curiosity
        self.inverse_model = inverse_model
        self.mapping_model = mapping_model
        self.prediction_model = prediction_model

        self.i_rewards = []
        self.actions = []
        self.one_hot_actions = []
        self.next_states = []

        self.loss_3 = 0
        self.loss_5 = 0

    def save_old_model(self):
        if self.old_model is None:
            self.old_model = copy.deepcopy(self.model)

        if device == 'cuda':
            with torch.cuda.device(self.args['gpu_ids'][0]):
                self.old_model = self.old_model.cuda()
                self.old_model.load_state_dict(self.model.state_dict())

    def action_train(self):
        if self.done:
            print('i am done')
            if device == 'cuda':
                with torch.cuda.device(self.gpu_id):
                    self.cx = Variable(torch.zeros(1, 256).float().cuda())
                    self.hx = Variable(torch.zeros(1, 256).float().cuda())
            else:
                self.cx = Variable(torch.zeros(1, 256).float())
                self.hx = Variable(torch.zeros(1, 256).float())

        else:

            self.cx = Variable(self.cx.data)
            self.hx = Variable(self.hx.data)
        self.old_state.append(self.state[0])
        if device == 'cuda':
            with torch.cuda.device(self.gpu_id):
                phi_states = self.inverse_model(Variable(self.state))

                value, logit, (self.hx, self.cx) = self.old_model((phi_states, (self.hx, self.cx)))
        else:
            phi_states = self.inverse_model(Variable(self.state))

            value, logit, (self.hx, self.cx) = self.old_model((phi_states, (self.hx, self.cx)))

        prob = F.softmax(logit)
        log_prob = F.log_softmax(logit)
        entropy = -(log_prob * prob).sum(1)
        self.entropies.append(entropy)
        action = prob.multinomial(num_samples = 1).data
        self.old_actions.append(Variable(action))
        log_prob = log_prob.gather(1, Variable(action))
        state, self.reward, self.done, self.info = self.env.step(action.cpu().numpy())
        self.eps_len += 1
        self.done = self.done or self.eps_len >= self.args['M']
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.rewards.append(self.reward)
        # curiosity
        if args['curiosity']:
            self.next_states.append(state[0])
            self.actions.append(action[0][0])
            one_hot_a = torch.zeros(self.args['action_space']).float().to(device)
            one_hot_a[action.cpu().numpy()] = 1

            self.one_hot_actions.append(one_hot_a)

            phi_states = phi_states.view(phi_states.size(0), -1)
            phi_next_states = self.inverse_model(state)
            phi_next_states = phi_next_states.view(phi_next_states.size(0), -1)
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


        self.state = state

        return self

    def action_replay(self,state,action):
        state = state.unsqueeze(0)
        if self.done:
            if device == 'cuda':
                with torch.cuda.device(self.gpu_id):
                    self.cx = Variable(torch.zeros(1, 256).float().cuda())
                    self.hx = Variable(torch.zeros(1, 256).float().cuda())
            else:
                self.cx = Variable(torch.zeros(1, 256).float())
                self.hx = Variable(torch.zeros(1, 256).float())

            self.done = False
        else:

            self.cx = Variable(self.cx.data)
            self.hx = Variable(self.hx.data)
        if device == 'cuda':
            with torch.cuda.device(self.gpu_id):
                phi_states = self.inverse_model(Variable(state))
                value, logit, (self.hx, self.cx) = self.model((phi_states, (self.hx, self.cx)))
        else:
            phi_states = self.inverse_model(Variable(state))
            value, logit, (self.hx, self.cx) = self.model((phi_states, (self.hx, self.cx)))

        prob = F.softmax(logit)
        log_prob = F.log_softmax(logit)
        entropy = -(log_prob * prob).sum(1)
        log_prob = log_prob.gather(1, Variable(action))
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



def train(args, rank, shared_model,inverse_model=None,mapping_model=None,prediction_model=None):

    print("start training thread ",rank)
    gpu_id = args['gpu_ids'][rank]

    torch.manual_seed(args['seed'] + rank)
    torch.cuda.manual_seed(args['seed'] + rank)
    env = atari_env(args,rank=rank)

    env.seed(args['seed'] + rank)

    optimizer = optim.Adam(list(list(shared_model.parameters())+list(inverse_model.parameters())), lr=args['LR'])
    if args['curiosity']:
        params = list(list(inverse_model.parameters()) + list(
            mapping_model.parameters()) + list(prediction_model.parameters()))
        optimizer_curiosity = optim.Adam(params, lr=args['curiosity_LR'])

    player = Agent(None, env, args, None, gpu_id=gpu_id,
                   inverse_model=inverse_model,mapping_model=mapping_model,prediction_model=prediction_model)
    player.model = copy.deepcopy(shared_model)
    player.save_old_model()
    if device == 'cuda':
        with torch.cuda.device(gpu_id):
            player.model = player.model.cuda()

    player.model.train()
    player.state = player.env.reset()

    for iteration in range(args['training_iteration']):


        player.model.load_state_dict(shared_model.state_dict())
        player.save_old_model()    # update the parameter of the old model to new model


        steps = 0
        for step in range(args['NS']):   # experience collection
            steps = steps + 1
            player.action_train()

            if player.done:
                break


        if player.done:
            player.eps_len = 0
            player.current_life = 0
            player.state = player.env.reset()

        rewards = []
        discounted_reward = 0
        reward_sum = 0
        for i in reversed(range(len(player.rewards))):
            reward_sum = reward_sum + player.rewards[i]
            discounted_reward = player.rewards[i] + args['G'] * discounted_reward
            rewards.insert(0, discounted_reward)

        rewards = torch.tensor(rewards).to(device)
        if args['curiosity']:
            player.i_rewards = np.array(player.i_rewards)
            player.i_rewards = (player.i_rewards - np.mean(player.i_rewards)) / (
                        np.std(player.i_rewards) + 0.0001) / 16.0

        epoch = args['ppo_epoch']
        for epochs in range(epoch):  ## PPO iteration
            player.done = True

            policy_loss = 0
            value_loss = 0

            for i in range(len(player.rewards)):
                logprobs, state_values, dist_entropy = player.action_replay(player.old_state[i],
                                                                            player.old_actions[i])
                ratios = torch.exp(logprobs - player.log_probs[i].detach())
                advantages = rewards[i] - state_values.detach()
                if args['curiosity']:
                    advantages = advantages + max(min(player.i_rewards[i], 0.125), -0.125)
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - args['eps_clip'], 1 + args['eps_clip']) * advantages
                policy_loss = policy_loss - torch.min(surr1, surr2) - 0.01 * dist_entropy
                value_loss = value_loss + nn.MSELoss()(state_values, rewards[i])


            policy_wei = 1
            value_wei = 5

            if epochs == epoch - 1:
                print("reward_sum, episode length, average reward, thread: ", reward_sum, len(player.rewards),
                      reward_sum / len(player.rewards), rank)
                print("policy loss, value loss", policy_loss, value_loss)
                step_loss = pickle.load(open(args['loss'], 'rb'))
                step_loss.append((steps, reward_sum, reward_sum / len(player.rewards),
                                  policy_loss.cpu().detach().numpy()[0][0],
                                  float(value_loss.cpu().detach().numpy()),
                                  float(player.loss_3.cpu().detach().numpy())/len(player.rewards),
                                  float(player.loss_5.cpu().detach().numpy())/len(player.rewards)))
                print("summary of step loss", step_loss[-50:-1])
                pickle.dump(step_loss, open(args['loss'], 'wb'))
            optimizer.zero_grad()
            (policy_wei * policy_loss + value_wei * value_loss).backward()

            ensure_shared_grads(player.model, shared_model)
            optimizer.step()     ##################################################
            # new/old/shared model's problem
            player.model.load_state_dict(shared_model.state_dict())
            player.done = True
        if args['curiosity']:
            states = torch.stack(player.old_state)
            next_states = torch.stack(player.next_states)
            player_actions = torch.stack(player.actions)
            player_actions.to(device)
            player.one_hot_actions = torch.stack(player.one_hot_actions)

            for epochs in range(args['curiosity_epoch']):
                # loss 3

                # print("stacked states shape",states.shape)

                phi_states = inverse_model(states)
                phi_states = phi_states.view(phi_states.size(0), -1)
                phi_next_states = inverse_model(next_states)
                phi_next_states = phi_next_states.view(phi_next_states.size(0), -1)
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

        player.clear_actions()
        torch.save(shared_model.state_dict(), args['mn'])
        if args['curiosity']:
            model_name = args['curiosity_mn']
            torch.save(inverse_model.state_dict(), model_name+ '_inverse')
            torch.save(mapping_model.state_dict(), model_name+'_mapping')
            torch.save(prediction_model.state_dict(), model_name+'_prediction')


def loadarguments():
    step_loss = []
    if args['L'] == False:
        pickle.dump(step_loss, open(args['loss'], 'wb'))
    torch.set_default_tensor_type('torch.FloatTensor')
    torch.manual_seed(args['seed'])
    shared_model = shared_parameter_Policy(3, args['action_space'])

    if args['curiosity']:
        final_dim = 6*6*32
        inverse_model = Inverse(3, args['action_space'])
        mapping_model = mapping(args['action_space'], final_dim)
        prediction_model = prediction(args['action_space'], final_dim)
    else:
        inverse_model = None
        mapping_model = None
        prediction_model = None

    if device == 'cuda':
        with torch.cuda.device(1):
            shared_model.cuda()
            if args['curiosity']:
                inverse_model.cuda()
                mapping_model.cuda()
                prediction_model.cuda()

    if args['L']:
        if device == 'cpu':
            shared_model.load_state_dict(torch.load(args['lmn'],map_location = torch.device('cpu')))
            if args['curiosity']:
                model_name = args['curiosity_mn']
                inverse_model.load_state_dict(torch.load(model_name+'_inverse',map_location = torch.device('cpu')))
                mapping_model.load_state_dict(torch.load(model_name + '_mapping', map_location=torch.device('cpu')))
                prediction_model.load_state_dict(torch.load(model_name + '_prediction', map_location=torch.device('cpu')))

        else:
            shared_model.load_state_dict(torch.load(args['lmn']))
            if args['curiosity']:
                model_name = args['curiosity_mn']
                inverse_model.load_state_dict(torch.load(model_name+'_inverse'))
                mapping_model.load_state_dict(torch.load(model_name + '_mapping'))
                prediction_model.load_state_dict(torch.load(model_name + '_prediction'))
    shared_model.share_memory()
    if args['curiosity']:
        inverse_model.share_memory()
        mapping_model.share_memory()
        prediction_model.share_memory()
    return shared_model,inverse_model,mapping_model,prediction_model



#device W inference
device = 'cuda'
#device = 'cpu'
#### now your model's policy parameter is never loading!

args = {'LR': 0.002, "G":0.9, "T":1.00,"NS":500,"M":500,'W':5,   ###############
         "seed":92,'L':False,'window':1,'turns':3,
        'gpu_ids':[1,1,1,1,1,1,1,1],'mn':'share_parameter',
        'lmn':'share_parameter','loss':'share.p',
        'ppo_epoch':4,'eps_clip':0.2,'action_space':3,
        'training_iteration':1000000,'inference':False,
        'curiosity':True,'curiosity_epoch':10,
        'curiosity_mn':'shared_parameter_curiosity',
        'curiosity_LR':0.002}


if device == 'cuda':
    torch.cuda.set_device(1)
## LR: learning rate. 'G': reward decaying rate. "NS" abd "M": episode length
## "W": parralel agent in training. "seed": random seed. "L": whether to load previously-saved model
## "mr": maximum possible reward
## 'mn': save model path
## 'lmn': load model path
## 'loss': loss pickle file path
## ppo epoch: how many iterations of ppo to train after each episode
## eps_clip: PPO parameter, restricts how much model is allowed to change
## grid, action space: how many discrete actions to consider
## window: let the ball slide for this amount of frames after an action

if __name__ == '__main__':
    torch.backends.cudnn.benchmark = False
    processes = []
    shared_model,inverse_model,mapping_model,prediction_model = loadarguments()
    torch.manual_seed(args['seed'])
    torch.cuda.manual_seed(args['seed'])

    mp.set_start_method('spawn')
    #train(args,optimizer,0,shared_model)



    for rank in range(0, args['W']):
        p = Process(
            target=train, args=(args, rank, shared_model,inverse_model,mapping_model,prediction_model))
        p.start()
        processes.append(p)
        time.sleep(10)
    for p in processes:
        p.join()
        time.sleep(10)
