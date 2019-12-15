from animalai.envs.gym.environment import AnimalAIEnv
from animalai.envs.arena_config import ArenaConfig
from dopamine.agents.rainbow import rainbow_agent
from dopamine.discrete_domains import run_experiment
import random
import numpy as np
import math
import copy
from PIL import Image
from map import map
from scipy import ndimage as ndi
from scipy.spatial import ConvexHull
import collections
import heapq

from scipy import ndimage
env_path = '../../env/AnimalAI'
worker_id = random.randint(1, 100)
arena_config_in = ArenaConfig('../configs/mapbuild.yaml')


savefile = True
class semantic_mask(object):
    def __init__(self):
        self.green = np.array([129.0, 191.0, 65.0])
        self.yellow = np.array([100.0, 65.0, 5.0])
        self.red = np.array([185.0, 50.0, 50.0])
        self.blue = np.array([50.0, 77.0, 121.0])
        self.mean = np.zeros((15, 3))
        for i in range(7):
            self.mean[i,:] = 1
            self.mean[i+8,:] = -1
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
    def boundary_mask(self,img):  # criteria 1: edge criteria 2: color above it 3: color below it


        edge = np.mean(img, axis=-1)
        darkest = np.min(img[30:,:,0],axis = 0)
        intensity = np.mean(edge,axis = 0)
        #print(intensity.shape,intensity.mean()) #1 d array mean is 100
        edge = ndimage.convolve(edge, self.mean, mode='constant', cval=0.0)

        result = np.zeros((84,84))
        grey = self.grey_mask(img) * 1
        blue = self.get_colours(img, self.blue)
        #ab,bb = np.where(blue)
        #skymax = np.max(ab) - 2

        grey_col = np.sum(grey, axis=0)
        valid_col = (grey_col < 2)
        for i in range(84):
            if valid_col[i]:
                criteria = 21 * 50 * (intensity[i] / 100)
                #valid_rows = np.where((img[30:,i,0]-darkest[i])<30)[0] + 30
                valid_rows = np.where((img[30:, i, 0] - 90) < 30)[0] + 30
                if len(valid_rows) == 0:
                    continue
                new_rows = edge[valid_rows,i]
                pos = valid_rows[np.argmax(new_rows)]

                #pos = np.argmax(edge[30:,i]) + 30
                #print(pos)
                if edge[pos,i] > criteria:
                    result[pos,i] = 255

        return result



    def process(self,img):
        print(self.steps, img[::5, ::5, 0], img[::5, ::5, 1], img[::5, ::5, 2])
        self.grey_mask(img)
        self.steps = self.steps+1

    def checkmask(self, img, points):
        self.steps = self.steps + 1

        img = Image.fromarray(img.astype(np.uint8))
        img.save('crr/my' + str(self.steps) + '.png')
        img = Image.fromarray(points.astype(np.uint8))
        img.save('cr/my' + str(self.steps) + '.png')
        return
        mask = np.zeros((84,84))
        for i in points:
            mask[i[0],i[1]] = 250
        img = Image.fromarray(mask.astype(np.uint8))
        img.save('cr/my' + str(self.steps) + '.png')


class Planner(object):
    def __init__(self,map):
        self.dir = [[1,0],
                    [0,1],
                    [-1,0],
                    [0,-1]]
                    #[0.7,0.7],
                    #[0.7,-0.7],
                    #[-0.7,0.7],
                    #[-0.7,-0.7]]

        self.conv = np.ones((10,10))
        self.blur = np.ones((3,3))
        self.confidence_bar = 1 # when do we want to take a loot at the pixel
        self.curiosity_bar = 2 # how many of these points do we need
        self.close_enough = 10 # close enough to count as obstacle -> higher, easy to cross wall -> lower, harder
        self.map = map
        self.obstacle_bfs = 0.5 # the lower, harder to cross wall, harder to find a existing path
        self.redestination_bar = 10*10*100

    def scandir(self,f,c,pos,dir):   # pos is in matrix's coordinate system
        curiosity_credit = 0
        obstacle_credit = 0
        ob = 0
        #a = ndimage.convolve(f, self.blur, mode='constant', cval=0.0)
        for i in range(30):
            n,m = int(dir[0]*i+pos[0]+0.5),int(dir[1]*i+pos[1]+0.5)
            if self.map.outbound(n,m):
                return False
            #obstacle_credit = obstacle_credit + f[n,m]
            if f[n,m] > 0 and self.map.maxc[n,m] > 0.1:
                return False
            if f[n,m] > 0:
                obstacle_credit = max(obstacle_credit,self.map.maxc[n,m])
            ob = ob + f[n,m]
            #if c[n,m] > 1000:
            #    if f[n,m] > 0.5:
            #        obstacle_credit = obstacle_credit + 1
            if self.map.maxc[n,m]<self.confidence_bar:
                print(self.map.maxc[n,m],obstacle_credit,ob)
            curiosity_credit = curiosity_credit + (1 if self.map.maxc[n,m]<self.confidence_bar else 0)
            if curiosity_credit > self.curiosity_bar:
                return True

        return False

    def get_scan_dir(self,f,c,pos):
        ans = []
        for i in self.dir:
            if np.sum(f) == 0:
                ans.append(np.array(i))
            #if self.scandir(f,c,pos,i):
            #    ans.append(np.array(i))
        return ans

    def get_best_loc(self,f,c,pos):
        a = ndimage.convolve(self.map.maxc, self.conv, mode='constant', cval=0.0)
        return np.unravel_index(np.argmin(a, axis=None), a.shape)

    def update_destination(self,dest,f,c):
        if dest is None:
            return self.get_best_loc(f,c,None)
        a = ndimage.convolve(self.map.maxc, self.conv, mode='constant', cval=0.0)
        if a[dest[0],dest[1]] > self.redestination_bar:
            return np.unravel_index(np.argmin(a, axis=None), a.shape)
        else:
            return dest

    def dijkstra(self,f,c,start,ed):
        grid = ndimage.convolve(f, self.blur, mode='constant', cval=0.0)
        did = np.ones((self.map.precision, self.map.precision)) * (-1)
        did[start[0], start[1]] = 0
        pq = [(0, start)]
        pax = np.zeros(f.shape)
        pay = np.zeros(f.shape)
        minconf = -1
        minx = 0
        miny = 0
        while len(pq) > 0:

            score, tp = heapq.heappop(pq)
            x, y = tp
            if score > did[x, y] + 0.01:
                continue
            did[x, y] = 1

            if (x,y) == ed:
                if score > 400:
                    break
                ans = [(x,y)]
                while (x,y) != start:
                    x,y = pax[int(x),int(y)],pay[int(x),int(y)]
                    ans.append((int(x),int(y)))

                return ans,ed
            if self.map.maxc[int(x),int(y)] < minconf or minconf == -1:
                minconf = self.map.maxc[int(x),int(y)]
                minx = int(x)
                miny = int(y)
            for x2, y2 in (
                    (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):  # ,(x+1,y+1),(x-1,y-1),(x+1,y-1),(x-1,y+1)):
                if self.map.outbound(x2, y2) == False:
                    newscore = score + 1 + grid[x2, y2] * c[x2,y2]
                    if did[x2, y2] < 0 or newscore < did[x2, y2]:
                        did[x2, y2] = newscore
                        heapq.heappush(pq, (newscore, (x2, y2)))
                        pax[x2, y2] = x
                        pay[x2, y2] = y
        x, y = minx, miny
        ans = [(x, y)]
        while (int(x), int(y)) != start:
            x, y = pax[int(x), int(y)], pay[int(x), int(y)]
            # print(x,y)
            ans.append((int(x), int(y)))

        return ans, (minx, miny)

    def get_score(self,f,c,start):

        totscore = 0
        grid = f  # ndimage.convolve(f, self.blur, mode='constant', cval=0.0)
        did = np.ones((self.map.precision,self.map.precision))*(-1)
        did[start[0],start[1]] = 0
        pq = [(0,start)]

        while len(pq) > 0:

            score, tp = heapq.heappop(pq)
            x, y = tp
            if score > did[x,y]+0.01:
                continue
            did[x,y] = 1
            totscore = score + max(100-self.map.maxc[x,y],0)*max((84-score),0.1)

            for x2, y2 in (
            (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):  # ,(x+1,y+1),(x-1,y-1),(x+1,y-1),(x-1,y+1)):
                if self.map.outbound(x2, y2) == False:
                    newscore = score + 1 + f[x2,y2]*10
                    if did[x2,y2] < 0 or newscore < did[x2,y2]:
                        did[x2,y2] = newscore
                        heapq.heappush(pq,(newscore,(x2,y2)))
        return totscore





    def no_obstacle(self, f, c, st,
                ed):

        ori = st


        tp = ed
        vec = (tp[0] - ori[0], tp[1] - ori[1])
        uni_vec = self.map.vec_uni(vec)
        if uni_vec == (0, 0):
            return True
        multiples = (abs(vec[0]) + abs(vec[1])) / (abs(uni_vec[0]) + abs(uni_vec[1]))
        for j in range(int(multiples + 1)):
            newpos = (int(ori[0] + j * uni_vec[0] + 0.5), int(ori[1] + j * uni_vec[1] + 0.5))
            if f[newpos[0], newpos[1]] > self.obstacle_bfs and self.map.maxc[newpos[0], newpos[1]] > self.close_enough:
                return False
        return True



# algorithm:
# check the need to scan -> a function f,c -> set of directions -> a planner that plans the turning
# if none, choose the best location to go -> use c matrix to find, easy
# then choose the best direction to go -> use "sky" or obstacle map to plan it
# then turn and go

# mode:
# fetch: as it is
# spin: when initialized, list of directions -> plan of turning in terms of a list of actions to take
# move: when initialized, plan of moving in terms of a list of actions to take
# it uses bfs to find the shortest path to destination
# then go though the list of points in bfs, finding one that gives no obstacle
# turn to that direction, and plan the move

#automaton graph:
# every frame, fetch can be reached
# spincheck is called everytime too, and spin mode can be directly forced
# when spincheck finish, move mode is initiated
# when move mode initiates, it uses floodfill to find a path, and return the next points along with pixels associated with it
# everytime map is updated, we check the pixels, and if they are blocked, move mode is reinitiated
# everytime 10 steps, move mode is reinitiated
# when the destination point has too large of confidence, re-destination is called
# move module should independently control its redestination and replanning of path
# if switched out by scanning: redo the turning and dist, redo the path
# if dist < 0: redo the turning and dist, redo path
# id destination points has too large confidence, redestination is called

# if cant move, mark obstacle, backward mode, replan

#details on graph:
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
        self.backward_turn = 0

        self.map = map()
        self.planner = Planner(self.map)
        self.mode = 'move'
        self.spin_turns = 0
        self.steps = 0
        self.k = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
        self.mean = np.zeros((3,3))
        self.mean = self.mean + 1/9
        self.k2 = np.array([[1,0,-1],[0,0,0],[-1,0,1]])
        self.max_r = -100
        self.destination = None

        self.theta = 3.1415926 / 2
        self.pi = 3.1415926
        self.points = [np.array([0, 0]),np.array([0.1, 0]),np.array([0, 0.1])]
        self.convex_hull = ConvexHull(np.array(self.points), incremental=True)
        self.point = np.array([0, 0])
        self.attempt_point = None

        self.fetch_steps = 0

        self.semantic_mask = semantic_mask()
        self.dark = False

    def reset(self, t=250):
        """
        Reset is called before each episode begins
        Leave blank if nothing needs to happen there
        :param t the number of timesteps in the episode
        """

        self.steps = 0
        self.points = [np.array([0, 0]), np.array([0.1, 0]), np.array([0, 0.1])]
        self.convex_hull = ConvexHull(np.array(self.points), incremental=True)
        self.point = np.array([0, 0])
        self.map=map()
        self.planner = Planner(self.map)
        self.destination = None
        self.setmode('move')


    def get_colours(self,pixels, colour):
        if np.array_equal(colour,self.blue):
            return np.all(pixels > np.minimum(colour * .95, colour - 5), axis=-1) & np.all(
                pixels < np.maximum(colour * 1.05, colour + 5), axis=-1)
        return np.all(pixels > np.minimum(colour * .8, colour - 25), axis=-1) & np.all(
            pixels < np.maximum(colour * 1.2, colour + 25), axis=-1)

    def inspect(self,img):
        green = self.get_colours(img,self.green)
        yellow = self.get_colours(img,self.yellow)
        ag, bg = np.where(green == True)
        ay, by = np.where(yellow == True)
        return len(ag)>0 or len(ay) > 0

    def fetch(self,obs):
        img = obs[0]
        green = self.get_colours(img, self.green)
        yellow = self.get_colours(img, self.yellow)
        a, b = np.where(green == True)
        ay, by = np.where(yellow == True)
        self.fetch_steps = self.fetch_steps + 1
        if len(ay) != 0:
            # print('ye')
            self.spin_turns = 0
            if np.mean(by) > 42:
                action = [1, 1]
            else:
                action = [1, 2]

        elif len(a) != 0:
            # print('gr')
            self.spin_turns = 0
            if np.mean(b) > 60:
                action = [0, 1]
            elif np.mean(b) < 20:
                action = [0, 2]
            elif np.mean(b) > 42:
                action = [1, 1]
            else:
                action = [1, 2]

        else:
            action = [0, 1]
            self.setmode('move')

        if self.fetch_steps > 2 and obs[1][2] < 0.5:
            grey = self.get_colours(img, self.grey)
            a,b = np.where(grey == True)
            if np.mean(b) > 43:
                action = [2,2]
            else:
                action = [2,1]
            if np.mean(b) < 25 or np.mean(b) > 57:
                self.fetch_steps = 0
        return action

    def spin(self,obs,reward,done,info):
        img = obs[0]
        action = self.actionlist[self.spin_turns]
        self.spin_turns = self.spin_turns + 1
        if self.spin_turns == len(self.actionlist):
            self.setmode('move')
        return action


    def move(self,obs,reward,done,info):
        if self.backward_turn != 0:
            self.backward_turn = self.backward_turn - 1
            if self.backward_turn != 0:
                return [2,0]
            else:
                self.replan()

        if self.move_turns < len(self.actionlist):
            action = self.actionlist[self.move_turns]
            self.move_turns = self.move_turns + 1
        else: # you need to keep adjusting angles, and recalculate destination path . you may also want to recalculate destination.
            action = [1,0]
            self.move_turns = self.move_turns + 1
            self.distance = self.distance - obs[1][2]
            if self.distance < 0:
                self.replan()
                action = self.actionlist[self.move_turns]
                self.move_turns = self.move_turns + 1
            if action == [1,0] and obs[1][2]>3:
                action = [0,0]

            if self.valid_openning(obs) == False:
                action[1] = self.seek_sky(obs[0])

            if obs[1][0] > 0.1:
                action[1]=1
            if obs[1][0] < -0.1:
                action[1]=2

            if (obs[1][2] < 0.1 and self.move_turns - len(self.actionlist) > 3):
                self.update_obs()
                action = [2,0]
                self.backward_turn = 2
        return action

    def seek_sky(self,img):
        blue = self.get_colours(img[:, :, :], self.blue)
        a, b = np.where(blue == True)
        skymean = np.mean(b)
        if skymean <= 42:
            return 2
        else:
            return 1
        # print(len(a))


    def update_obs(self):
        self.map.update_obstacle_strong(self.point)

    def replan(self):
        # see if we need to changed destination
        self.destination = self.planner.update_destination(self.destination,self.map.f,self.map.c)

        points = self.getpath()  # an array of points
        subp = points[-3:]
        for i in range(0,len(subp),2):
            if self.planner.no_obstacle(self.map.f, self.map.c, self.map.world_coord_2_map_loc(self.point),
                                        subp[i]) or i == 0:
                desired_p = self.map.map_loc_2_world_coord(subp[i])
                x, y = desired_p[0] - self.point[0], desired_p[1] - self.point[1]
                theta = self.update(self.vec_2_theta((x, y)), - self.theta)  # implement this
                self.actionlist = self.theta_2_action(theta)
                self.distance = math.sqrt(x * x + y * y) / self.map.scaling
                break
        self.move_turns = 0

    def valid_openning(self,obs):
        img = obs[0]
        blue = self.get_colours(img[:,41:45,:], self.blue)
        a, b = np.where(blue == True)
        #print(len(a))
        return len(a)>self.blue_threshold

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

    def setmode(self,mode,args = None):
        self.mode = mode

        if mode == 'spin':
            dir = args['dir'] # a list of angle in radians  # need to calculate this #in world coor reference
            maxangle = -100
            minangle = 100
            for i in dir:
                maxangle = max(maxangle,self.update(i,-self.theta))
                minangle = min(minangle,self.update(i,-self.theta))
            if maxangle <= 0:
                actionlist = self.theta_2_action(minangle)

            elif minangle >= 0:
                actionlist = self.theta_2_action(maxangle)
            elif minangle < -self.pi / 3.0 and maxangle > self.pi/3.0:
                actionlist = self.theta_2_action(self.pi*2.0)
            elif -minangle > maxangle: #meaning turning maxangle then minangle
                actionlist = self.theta_2_action(maxangle) + self.theta_2_action(minangle-maxangle)
            else:
                actionlist = self.theta_2_action(minangle) + self.theta_2_action(maxangle-minangle)
            self.actionlist = actionlist
            self.spin_turns = 0
        elif mode == 'fetch':
            self.fetch_steps = 0
        elif mode == 'move':
            self.replan()
        return

    def vec_2_theta(self,vec):
        x,y = vec
        return math.atan2(y,x)

    def theta_2_action(self,angle):
        actionlist = []
        theta = 0
        while True:
            if angle < 0:
                theta = theta - 6 / 180 * self.pi
                actionlist.append([0, 1])
                if theta < angle:
                    break
            else:
                theta = theta + 6 / 180 * self.pi
                actionlist.append([0, 2])
                if theta > angle:
                    break
        return actionlist



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

    def savefig(self,img):
        img = Image.fromarray(img.astype(np.uint8))
        img.save('crr/my' + str(self.steps) + '.png')

    def test_planner(self):
        pos = self.planner.get_best_loc(self.map.f,self.map.c,self.map.world_coord_2_map_loc( self.point))
        #print('start=',self.map.world_coord_2_map_loc( self.point))
        #print('end=',pos)
        path,dest = self.planner.dijkstra(self.map.f,self.map.c,self.map.world_coord_2_map_loc( self.point),pos)
        #print('path=',path)
        img = np.zeros((self.map.precision, self.map.precision, 3))
        # img[:,:,0] = self.f*255
        img[:, :, 1] = self.map.f * 255
        #img[:, :, 2] = self.f * 255
        img[:, :, 0] = img[:, :, 0] + self.map.i * 255
        for i in path:
            x,y = i
            img[x,y,2] = 255
        img = Image.fromarray((img).astype(np.uint8))
        img.save('pla/my' + str(self.steps) + '.png')
        cmap = np.zeros((self.map.precision, self.map.precision))
        cmap[self.map.c>10] = 255
        img = Image.fromarray((cmap).astype(np.uint8))
        img.save('c/my' + str(self.steps) + '.png')
        #self.steps = self.steps + 1



    def getpath(self):
        mappos = self.map.world_coord_2_map_loc(self.point)
        path,self.destination = self.planner.dijkstra(self.map.f, self.map.c, mappos, self.destination)
        return path

    def getdir(self):
        mappos = self.map.world_coord_2_map_loc(self.point)
        if self.steps == 0:
            dir = copy.deepcopy(self.planner.dir)
        else:
            return []
        #dir = self.planner.get_scan_dir(self.map.f, self.map.c, mappos)
        worldpos = []
        for i in dir:
            worldpos.append(self.map.map_loc_2_world_coord((mappos[0] + i[0], mappos[1] + i[1])))
        thetas = []
        for i in worldpos:
            thetas.append(self.vec_2_theta((i[0] - self.point[0], i[1] - self.point[1])))
        return thetas

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


        #points = self.semantic_mask.boundary_mask(obs[0])

        #self.semantic_mask.checkmask(obs[0],self.semantic_mask.boundary_mask(obs[0]))

        self.map.update_obstacle(self.point,self.theta,self.semantic_mask.grey_mask(obs[0]))
        self.map.update_boundary(self.point,self.theta,self.semantic_mask.boundary_mask(obs[0]))
        self.map.update_self(self.point)
        if savefile:
            self.map.savemap()
        if savefile:
            self.test_planner()
        if self.inspect(obs[0]):
            self.setmode('fetch')
        elif self.mode != 'spin':
            thetas = self.getdir()
            print('thetas = ',thetas,self.planner.get_scan_dir(self.map.f, self.map.c, self.map.world_coord_2_map_loc(self.point)))
            if len(thetas) != 0:
                self.setmode('spin', {'dir': thetas})
        if savefile:
            self.savefig(obs[0])
        self.steps = self.steps + 1
        self.update_convexhull(obs)

        print(self.mode)
        if self.mode == 'spin':
            action = self.spin(obs,reward,done,info)
        elif self.mode == 'move':
            action = self.move(obs,reward,done,info)
        else:
            action = self.fetch(obs)

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
        if done:
            break
    agent.reset(500)




