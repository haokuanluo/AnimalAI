from animalai.envs.gym.environment import AnimalAIEnv
from animalai.envs.arena_config import ArenaConfig
from dopamine.agents.rainbow import rainbow_agent
from dopamine.discrete_domains import run_experiment
import random
import numpy as np
import math
import copy
from PIL import Image
from scipy import ndimage as ndi
from scipy.spatial import ConvexHull

from scipy import ndimage
env_path = '../../env/AnimalAI'
worker_id = random.randint(1, 100)
arena_config_in = ArenaConfig('../configs/exampleTraining.yaml')

class semantic_mask(object):
    def __init__(self):
        self.green = np.array([129.0, 191.0, 65.0])
        self.yellow = np.array([100.0, 65.0, 5.0])
        self.red = np.array([185.0, 50.0, 50.0])
        self.blue = np.array([50.0, 77.0, 121.0])
        self.mean = np.zeros((3, 3))
        self.mean = self.mean + 1 / 9

        self.grey = np.array([120.0, 120.0, 120.0])
        self.grey2 = np.array([64,64,64])
        self.steps = 0
        self.pi = 3.1415926
        self.vision_angle = self.pi / 2

    def get_colours(self,pixels, colour):
        if np.array_equal(colour,self.blue):
            return np.all(pixels > np.minimum(colour * .95, colour - 5), axis=-1) & np.all(
                pixels < np.maximum(colour * 1.05, colour + 5), axis=-1)
        if np.array_equal(colour,self.grey) or np.array_equal(colour,self.grey2):
            a = np.all(pixels > np.minimum(colour * .8, colour - 25), axis=-1) & np.all(
                pixels < np.maximum(colour * 1.2, colour + 25), axis=-1)

            b = np.abs(pixels[:,:,0] - pixels[:,:,1]) + np.abs(pixels[:,:,1] - pixels[:,:,2])
            a = a & (b < 2)
            #print(a)
            return a
        return np.all(pixels > np.minimum(colour * .8, colour - 25), axis=-1) & np.all(
            pixels < np.maximum(colour * 1.2, colour + 25), axis=-1)


    def blur_(self,img):


        img = ndimage.convolve(img, self.mean, mode='constant', cval=0.0)
        return img

    def blur(self, img_):
        img = copy.deepcopy(img_)
        for i in range(3):
            img[i]=self.blur_(img[i])
        return img

    def grey_mask(self,img):
        #blured = self.blur(img)

        grey = (self.get_colours(img, self.grey) | self.get_colours(img, self.grey2))
        return grey
        #print(grey.shape)
        #exit()
        #img = Image.fromarray(img.astype(np.uint8))
        #img.save('crr/my' + str(self.steps) + '.png')

        #img = Image.fromarray(grey.astype(np.uint8))
        #img.save('cr/my' + str(self.steps) + '.png')
    def process(self,img):
        #print(self.steps, img[::5, ::5, 0], img[::5, ::5, 1], img[::5, ::5, 2])
        #self.grey_mask(img)
        self.steps = self.steps+1


class Agent(object):

    def __init__(self):
        """
         Load your agent here and initialize anything needed
         WARNING: any path to files you wish to access on the docker should be ABSOLUTE PATHS
        """
        self.green = np.array([129.0,191.0,65.0])
        self.yellow = np.array([100.0,65.0,5.0])
        self.red = np.array([185.0,50.0,50.0])
        self.blue = np.array([50.0, 77.0, 121.0])
        self.grey = np.array([115.0,115.0,115.0])
        self.blue_threshold = 50
        self.grey_threshold = 100
        self.best_dir = None
        self.explore_turns = 0

        self.mode = 'spin'
        self.spin_turns = 0
        self.steps = 0
        self.k = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
        self.mean = np.zeros((3,3))
        self.mean = self.mean + 1/9
        self.k2 = np.array([[1,0,-1],[0,0,0],[-1,0,1]])
        self.max_r = -100

        self.theta = 3.1415926 / 2
        self.pi = 3.1415926
        self.points = [np.array([0, 0]),np.array([0.1, 0]),np.array([0, 0.1])]
        self.convex_hull = ConvexHull(np.array(self.points), incremental=True)
        self.point = np.array([0, 0])
        self.attempt_point = None
        self.lowests = []
        self.distances = []

        self.fetch_steps = 0

        self.semantic_mask = semantic_mask()


    def reset(self, t=250):
        """
        Reset is called before each episode begins
        Leave blank if nothing needs to happen there
        :param t the number of timesteps in the episode
        """
        self.setmode('spin')
        self.steps = 0
        self.points = [np.array([0, 0]), np.array([0.1, 0]), np.array([0, 0.1])]
        self.convex_hull = ConvexHull(np.array(self.points), incremental=True)
        self.point = np.array([0, 0])


    def get_colours(self,pixels, colour):
        if np.array_equal(colour,self.blue):
            return np.all(pixels > np.minimum(colour * .95, colour - 5), axis=-1) & np.all(
                pixels < np.maximum(colour * 1.05, colour + 5), axis=-1)
        return np.all(pixels > np.minimum(colour * .8, colour - 25), axis=-1) & np.all(
            pixels < np.maximum(colour * 1.2, colour + 25), axis=-1)

    def inspect(self,img):
        green = self.semantic_mask.grey_mask(img)
        ag, bg = np.where(green == True)
        return len(ag)>100

    def fetch(self,obs):
        img = obs[0]
        green = self.semantic_mask.grey_mask(img)
        a, b = np.where(green == True)
        if len(a) > 0:
            # print('ye')
            self.spin_turns = 0
            if np.mean(b) > 42:
                action = [2, 1]
            elif np.mean(b) < 42:
                action = [2, 2]
            else:
                action = [2, 0]
        self.fetch_steps = self.fetch_steps + 1
        if obs[1][2] > -0.5 and self.fetch_steps > 5:
            self.setmode('forward')

        return action

    def spin(self,obs,reward,done,info):
        img = obs[0]


        action = [0, 1]
        self.spin_turns = self.spin_turns + 1
        if self.spin_turns > 360/6:
            self.setmode('explore')
        return action

    def check(self,obs):



        img = obs[0]
        blue = self.get_colours(img, self.blue)
        a, b = np.where(blue == True)
        if len(a) > 0:
            self.sky_appear = True
        else:
            if self.sky_appear == True:
                self.setmode('explore')

        self.spin_turns = self.spin_turns + 1
        if self.spin_turns > 360 / 6:
            self.setmode('explore')

        return self.check_turn

    def explore(self,obs,reward,done,info):
        self.explore_turns = self.explore_turns - 1
        if self.explore_turns > 0:
            return self.explore_turn
        if obs[1][0]>0.5:
            return [1,1]
        if obs[1][0]<-0.5:
            return [1,2]
        img = obs[0]
        blue = self.get_colours(img, self.blue)
        a, b = np.where(blue == True)
        if len(a) > 10:
            self.sky_mean = np.mean(b)
        if obs[1][2]<0.5 and self.explore_turns < -5:
            self.setmode('check',obs=obs)
            return [1,0]
        return [1,0]

    def valid_openning(self,obs):
        img = obs[0]
        blue = self.get_colours(img[:,41:45,:], self.blue)
        a, b = np.where(blue == True)
        #print(len(a))
        return len(a)>self.blue_threshold


    def getedge(self,img):
        edge = img
        edge = np.mean(edge, axis=-1)
        img = Image.fromarray(edge.astype(np.uint8))
        #img.save('crr/my' + str(self.steps) + '.png')
        edge = ndimage.convolve(edge, self.mean, mode='constant', cval=0.0)
        edge = ndimage.convolve(edge, self.k, mode='constant', cval=0.0)
        # edge = feature.canny(edge, sigma=3)*255
        print(np.mean(edge))
        ff = edge.astype(np.uint8)
        img = Image.fromarray(ff)
        #img.save('cr/my' + str(self.steps) + '.png')
        return edge

    def update_convexhull(self,obs):
        direction = np.array([math.cos(self.theta), math.sin(self.theta)])
        velocity = direction * obs[1][2]
        if np.linalg.norm(velocity) > 0.1:
            self.point = self.point + velocity
            self.points.append(self.point)

            #previous_area = self.convex_hull.area
            self.convex_hull.add_points(np.array([self.point]))
            #new_area = self.convex_hull.area
            #convex_reward = max(new_area - previous_area, 0)
            #convex_reward = math.sqrt(convex_reward)  # [0,4]

    def forward(self,obs):
        grey = self.semantic_mask.grey_mask(obs[0])
        a, b = np.where(grey == True)
        hehe = self.semantic_mask.grey_mask(obs[0][:,42:44,:])
        aa,bb = np.where(hehe == True)
        lowest = np.max(aa)
        self.lowests.append(lowest)
        self.distances.append(obs[1][2])
        img = Image.fromarray(obs[0].astype(np.uint8))
        img.save('crr/my' + str(self.steps) + '.png')
        img = Image.fromarray((hehe*255).astype(np.uint8))
        img.save('cr/my' + str(self.steps) + '.png')
        if len(self.lowests) > 5 and obs[1][2]<0.1:
            dista = 0
            for i in reversed(range(len(self.lowests))):
                dista = dista + self.distances[i]
                self.distances[i] = dista
            import pickle
            pickle.dump((self.lowests,self.distances),open('dist.p','wb'))
            exit()
        return [1,0] if obs[1][2]<0.5 else [0,0]

    def setmode(self,mode,obs = None):
        self.mode = mode
        if mode == 'spin':
            self.spin_turns = 0
            self.max_r = -100
        elif mode == 'explore':
            desired_spin = self.update(-self.theta, self.best_dir)
            if desired_spin > 0:
                self.explore_turn = [0, 2]
                self.explore_turns = desired_spin // (6 / 180 * self.pi)
            else:
                self.explore_turn = [0, 1]
                self.explore_turns = desired_spin // (-6 / 180 * self.pi)
            self.sky_mean = -1
            self.convex_hull.add_points(np.array([self.attempt_point]))
        elif mode == 'fetch':
            self.fetch_steps = 0
        elif mode == 'check':
            assert obs is not None
            self.spin_turns = 0
            self.max_r = -100

            img = obs[0]
            blue = self.get_colours(img, self.blue)
            a, b = np.where(blue == True)
            if len(a) > 0:
                if np.mean(b) < 42:
                    self.check_turn = [0,2]
                else:
                    self.check_turn = [0,1]
                self.sky_appear = True
            else:
                if self.sky_mean < 42:
                    self.check_turn = [0, 2]
                else:
                    self.check_turn = [0, 1]
                self.sky_appear = False
        elif mode == 'forward':
            pass



    def update(self,theta,angle):
        theta = theta + angle
        if theta > self.pi:
            theta = theta - 2*self.pi
        if theta < -self.pi:
            theta = theta + 2*self.pi
        return theta

    def free_reward(self,obs):
        ch = ConvexHull(np.array(self.points), incremental=True)
        area = ch.area
        direction = np.array([math.cos(self.theta), math.sin(self.theta)])
        velocity = direction * 50
        newp = self.point + velocity
        ch.add_points(np.array([newp]))
        narea = ch.area
        return narea-area,newp
    def step(self, obs, reward, done, info):
        """
        A single step the agent should take based on the current state of the environment
        We will run the Gym environment (AnimalAIEnv) and pass the arguments returned by env.step() to
        the agent.

        Note that should if you prefer using the BrainInfo object that is usually returned by the Unity
        environment, it can be accessed from info['brain_info'].

        :param obs: agent's observation of the current environment
        :param reward: amount of reward returned after previous action
        :param done: whether the episode has ended.
        :param info: contains auxiliary diagnostic information, including BrainInfo.
        :return: the action to take, a list or size 2
        """
        a,b = obs
        obs = (a*255,b)
        if self.inspect(obs[0]) and self.mode == 'spin':
            self.setmode('fetch')

        self.semantic_mask.process(obs[0])

        self.steps = self.steps + 1
        self.update_convexhull(obs)
        if self.valid_openning(obs):
            free_r,newp = self.free_reward(obs)
            if free_r > self.max_r:
                self.max_r = free_r
                self.best_dir = self.theta
                self.attempt_point = newp
        print(self.mode)
        if self.mode == 'spin':
            action = self.spin(obs,reward,done,info)
        elif self.mode == 'explore':
            action = self.explore(obs,reward,done,info)
        elif self.mode == 'fetch':
            action = self.fetch(obs)
        elif self.mode == 'forward':
            action = self.forward(obs)
        else:
            action = self.check(obs)
        if action[1] == 1:
            self.theta = self.update(self.theta, -6 / 180 * self.pi)
        if action[1] == 2:
            self.theta = self.update(self.theta, 6 / 180 * self.pi)




        return action



agent = Agent()
agent.reset()

def create_env_fn():
    env = AnimalAIEnv(environment_filename=env_path,
                      worker_id=worker_id,
                      n_arenas=1,
                      arenas_configurations=arena_config_in,
                      docker_training=False,
                      retro=False,
                      inference=True,
                      resolution=84,
                      seed = random.randint(0,100))
    return env


env = create_env_fn()
while True:
    action = [1, 0]
    for i in range(500):
        obs,reward,done,info = env.step(action)
        action=agent.step(obs,reward,done,info)
    agent.reset(500)




