from animalai.envs.gym.environment import AnimalAIEnv
from animalai.envs.arena_config import ArenaConfig
import random
import numpy as np
import math
import copy
from PIL import Image

from scipy.spatial import ConvexHull
import collections
import pickle
import heapq
from scipy import ndimage
env_path = '../../env/AnimalAI'
worker_id = random.randint(1, 100)


savefile = False

class map(object):
    def __init__(self,savepath = 'map/my'):
        self.pic2pos = pickle.load(open('/aaio/data/reconstruct.p','rb'))
        self.precision = 125
        self.scaling = self.precision/2000
        self.f = np.zeros((self.precision,self.precision))
        self.c = np.zeros((self.precision,self.precision))
        self.maxc = np.zeros((self.precision,self.precision))
        self.i = np.zeros((self.precision,self.precision))
        self.midpoint = 2000/2
        self.pi = 3.1415926
        self.ang = self.pi/2.5
        self.stepsize = 1
        self.steps = 0
        self.savepath = savepath

    def getabsolutepoints(self,mask,val = True):
        points = []
        for i in range(0,84):
            pix = np.where(mask[42:,i]==val)[0]
            pix = pix + 42
            if len(pix) == 0:
                continue
            #print(pix,len(pix))
            pix = np.max(pix)
            pos = self.pic2pos[pix]
            #print(self.steps,pix,pos)
            y = pos
            theta = abs(i-42)/84*self.ang

            x = math.tan(theta) * y
            if i < 43:
                x = -x
            c = abs(i-44)
            c = c if (c>30 and pix < 50) else 0
            if pix > 0:
                points.append((x,y,c))
        return points
    def getabsoluteboundary(self,mask):
        points = []
        for i in range(0,84):
            pix = np.where(mask[42:,i]==True)[0]
            pix = pix + 42
            if len(pix) == 0:
                continue
            pix = np.max(pix)
            pos = self.pic2pos[pix]
            y = pos
            theta = abs(i-42)/84*self.ang

            x = math.tan(theta) * y
            if i < 43:
                x = -x
            c = abs(i-44)
            c = 0 if c < 20 else c
            points.append((x,y,c))
        return points

    def corrected(self,points,theta):
        for i in range(len(points)):
            x,y,c= points[i]
            nx = math.cos(theta)*x-math.sin(theta)*y
            ny = math.sin(theta)*x+math.cos(theta)*y
            points[i]=(nx,ny,c)
        return points

    def outbound(self,x,y):
        return x<0 or x >= self.precision or y<0 or y>=self.precision

    def world_coord_2_map_loc(self,coord):
        x,y = coord
        x,y = -y,x
        x,y = self.midpoint - y,self.midpoint + x
        x,y = int(x*self.scaling+0.5),int(y*self.scaling+0.5)
        return x,y

    def map_loc_2_world_coord(self,loc):
        x,y = loc
        x,y = x/self.scaling,y/self.scaling
        x,y = y - self.midpoint,self.midpoint - x
        x,y = y,-x
        return x,y

    def self_loc_2_map_loc(self,loc):
        x, y = loc
        x, y = self.midpoint - y, self.midpoint + x
        x, y = int(x * self.scaling+0.5), int(y * self.scaling+0.5)
        return x, y

    def set(self,f,x,y,v,c=-1):

        nx = self.midpoint - y
        ny = self.midpoint + x
        nx = int(nx*self.scaling+0.5)
        ny = int(ny*self.scaling+0.5)
        if self.outbound(nx,ny):
            return False
        if c == -1:
            f[nx,ny] = v
        else:  # idea: if previosu max c is 20, and rn c is 25, then we want current c to have weight 25:20 in the composition
            # if previous is max c 20, total c 500, and rn c is 25, you want ratio of 20 * (sqrt(500/20)):25, not 100:25
            # total c is sum, max c is max
            if c < 1.11 and self.maxc[nx,ny] > 5:
                return
            f[nx, ny] = (f[nx, ny] * (self.maxc[nx,ny] * math.sqrt(self.c[nx,ny]/(self.maxc[nx,ny]+0.0001))) + v * c) / \
                        ((self.maxc[nx,ny] * math.sqrt(self.c[nx,ny]/(self.maxc[nx,ny]+0.0001))) + c + 0.000001)
            self.c[nx, ny] = self.c[nx, ny] + c
            self.maxc[nx, ny] = max(self.maxc[nx,ny],c)
        return True

    def raw_set(self,f,x,y,v,c=-1):


        if self.outbound(x,y):
            return False
        if c == -1:
            f[x,y] = v
        else:
            nx,ny = x,y
            if c < 1.11 and self.maxc[nx,ny] > 5:
                return

            f[nx, ny] = (f[nx, ny] * (self.maxc[nx, ny] * math.sqrt(self.c[nx, ny] / (self.maxc[nx, ny]+0.0001))) + v * c) / (
                        (self.maxc[nx, ny] * math.sqrt(self.c[nx, ny] / (self.maxc[nx, ny]+0.0001))) + c + 0.000001)
            self.c[nx, ny] = self.c[nx, ny] + c
            self.maxc[nx, ny] = max(self.maxc[nx,ny], c)

        return True

    def _set(self,f,x,y,v,c=-1):
        for i in range(-1,2):
            for j in range(-1,2):
                self.set(f,x+i,y+j,v,c)
        return


    def get(self,f,x,y):
        return f[int(self.scaling*(self.midpoint-y)+0.5),int(self.scaling*(self.midpoint+x)+0.5)]

    def vec_uni(self,vec):
        x,y = vec
        if x == 0 and y == 0:
            #print('zero')
            return (0,0)
        len = math.sqrt(x*x+y*y)
        return (x/len,y/len)
    def updatemap(self,points,pos):
        for i in points:
            self.set(self.f,i[0]+pos[0],i[1]+pos[1],1,self.dist_2_conf(math.sqrt(i[0]*i[0]+i[1]*i[1])+i[2]*10))

        ori = self.self_loc_2_map_loc(pos)

        for i in points:
            tp = self.self_loc_2_map_loc((i[0]+pos[0],i[1]+pos[1]))
            vec = (tp[0]-ori[0],tp[1]-ori[1])
            uni_vec = self.vec_uni(vec)
            if uni_vec == (0,0):
                continue
            multiples = (abs(vec[0])+abs(vec[1])) / (abs(uni_vec[0])+abs(uni_vec[1]))
            tpdist = math.sqrt(i[0]*i[0]+i[1]*i[1])
            for j in range(int(multiples+1)):
                newpos = (int(ori[0]+j*uni_vec[0]+0.5),int(ori[1]+j*uni_vec[1]+0.5))
                if newpos != tp:
                    self.raw_set(self.f,newpos[0],newpos[1],0,self.dist_2_conf(tpdist * j/ multiples+i[2]*10))




    def dist_2_conf(self,dist):
        dist = max(dist,1)
        #print('conf',100000/(dist*dist))
        return 100000/(dist*dist)
    def updateboundary(self,points,pos):
        #for i in points:
        #    self.set(self.f,i[0]+pos[0],i[1]+pos[1],1,i[2])

        ori = self.self_loc_2_map_loc(pos)

        for i in points:
            tp = self.self_loc_2_map_loc((i[0] + pos[0], i[1] + pos[1]))
            vec = (tp[0] - ori[0], tp[1] - ori[1])
            uni_vec = self.vec_uni(vec)
            if uni_vec == (0,0):
                continue
            multiples = (abs(vec[0])+abs(vec[1])) / (abs(uni_vec[0])+abs(uni_vec[1]))
            tpdist = math.sqrt(i[0] * i[0] + i[1] * i[1])
            for j in range(1000):
                newpos = (int(ori[0] + j * uni_vec[0]+0.5), int(ori[1] + j * uni_vec[1]+0.5))
                if j < multiples:
                    if newpos != tp:
                        self.raw_set(self.f, newpos[0], newpos[1], 0,
                                     self.dist_2_conf(tpdist * j / multiples + i[2] * 10)) # original 1000 multiplier for conf
                else:
                    kk = self.raw_set(self.f, newpos[0], newpos[1], 1,
                                      self.dist_2_conf(tpdist * (multiples - (j-multiples)) / multiples + i[2] * 10))
                    if kk == False:
                        break





    def update_obstacle(self,pos_,dir,mask):
        pos = (-pos_[1],pos_[0])
        points = self.getabsolutepoints(mask)
        if len(points) == 0:
            return
        points = self.corrected(points,dir)
        self.updatemap(points,pos)

    def update_self(self,pos_):
        pos = (-pos_[1], pos_[0])
        self.set(self.i,pos[0],pos[1],1)

    def update_obstacle_strong(self,pos_):
        pos = (-pos_[1], pos_[0])
        self.set(self.f,pos[0],pos[1],1,1000000)
    def update_boundary(self,pos_,dir,mask):
        pos = (-pos_[1], pos_[0])
        points = self.getabsolutepoints(mask,val = 255)
        # self.i[int(pos[1]),int(pos[0])] = 1
        if len(points) == 0:
            return
        points = self.corrected(points, dir)
        self.updateboundary(points, pos)


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
    def bfs_(self, f,c, start,end,obstacle_bfs,conf):
        #grid = ndimage.convolve(f, self.blur, mode='constant', cval=0.0)
        queue = collections.deque([start])
        seen = set([start])
        pax = np.zeros(f.shape)
        pay = np.zeros(f.shape)
        a = ndimage.convolve(c, self.conv, mode='constant', cval=0.0)
        grid = f#ndimage.convolve(f, self.blur, mode='constant', cval=0.0)

        minconf = -1
        minx = 0
        miny = 0
        while queue:
            path = queue.popleft()
            x, y = path
            if self.map.maxc[int(x),int(y)] < minconf or minconf == -1:
                minconf = self.map.maxc[int(x),int(y)]
                minx = int(x)
                miny = int(y)
            if (x,y) == end:
                ans = [(x,y)]
                while (x,y) != start:
                    x,y = pax[int(x),int(y)],pay[int(x),int(y)]
                    ans.append((int(x),int(y)))

                return ans, end
            for x2, y2 in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):#,(x+1,y+1),(x-1,y-1),(x+1,y-1),(x-1,y+1)):
                if self.map.outbound(x2,y2) == False and (grid[x2][y2] <= obstacle_bfs or self.map.maxc[x2,y2] < conf) and (x2, y2) not in seen:
                    queue.append((x2, y2))
                    seen.add((x2, y2))
                    pax[x2,y2] = x
                    pay[x2,y2] = y
        x,y = minx,miny
        ans = [(x, y)]
        while (int(x), int(y)) != start:
            x, y = pax[int(x), int(y)], pay[int(x), int(y)]
            #print(x,y)
            ans.append((int(x), int(y)))


        return ans, (minx,miny)

    def bfs(self,f,c,start,end):
        minobs = 0
        highobs = 1
        minconf = 0.5
        maxconf = 1.4
        myans = None
        myed = (-1,-1)
        for i in range(8):
            midobs = (minobs + highobs)/2.0
            midconf = (maxconf+minconf)/2.0
            conf = 10**midconf
            ans, ed = self.bfs_(f,c,start,end,midobs,conf)
            if ed == end:
                myans,myed = ans,ed
                self.obstacle_bfs = midobs
                self.close_enough=conf

                highobs = midobs
                maxconf = midconf
            else:
                if myed != end:
                    myans,myed = ans,ed

                    self.obstacle_bfs = midobs
                    self.close_enough=conf

                minobs = midobs
                minconf = midconf
        #self.obstacle_bfs = midobs
        return myans,myed

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
            totscore = score + max(10-self.map.maxc[x,y],0)*max((104-score),0.1)

            for x2, y2 in (
            (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):  # ,(x+1,y+1),(x-1,y-1),(x+1,y-1),(x-1,y+1)):
                if self.map.outbound(x2, y2) == False:
                    newscore = score + 1 + f[x2,y2]*50
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
        self.obstacle_threshold = 1764
        self.map = map()
        self.planner = Planner(self.map)

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
        self.init = True
        self.force_turn = False

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
        self.map=map()
        self.planner=Planner(self.map)
        self.init = True
        self.force_turn = False
        self.t = t


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
            if np.mean(by) > 40:
                action = [1, 1]
            elif np.mean(by) < 40:
                action = [1, 2]
            else:
                action = [1, 0]
        elif len(a) != 0:
            # print('gr')
            self.spin_turns = 0
            if np.mean(b) > 60:
                action = [0, 1]
            elif np.mean(b) < 20:
                action = [0, 2]
            elif np.mean(b) > 40:
                action = [1, 1]
            elif np.mean(b) < 40:
                action = [1, 2]
            else:
                action = [1, 0]
        else:
            action = [0, 1]
            self.setmode('spin')

        if self.fetch_steps > 2 and obs[1][2] < 0.5:
            grey = self.get_colours(img, self.grey)
            a,b = np.where(grey == True)
            if np.mean(b) > 40:
                action = [0,2]
            else:
                action = [0,1]
            if np.mean(b) < 25 or np.mean(b) > 57:
                self.fetch_steps = 0
        return action

    def spin(self,obs,reward,done,info):
        img = obs[0]


        action = [0, 1]
        self.spin_turns = self.spin_turns + 1
        if self.spin_turns > 360/6:
            self.setmode('explore')
            if self.init:
                self.init = False
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
        if self.explore_turns == 0:
            self.force_turn = False
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
            return [2,0]
        return [1,0]

    def valid_openning(self,obs):
        img = obs[0]
        blue = self.get_colours(img[:,41:45,:], self.blue)
        a, b = np.where(blue == True)
        grey = self.semantic_mask.grey_mask(img)
        ga,gb = np.where(grey[42:,:] == True)

        #print(len(a))
        return len(a)>self.blue_threshold and len(ga) < self.obstacle_threshold and self.danger(img) == False

    def score(self,img):
        green = self.get_colours(img, self.green)
        yellow = self.get_colours(img, self.yellow)
        ag, bg = np.where(green == True)
        ay, by = np.where(yellow == True)
        return 10000*max(len(ag),len(ay)*2)


    def getedge(self,img):
        edge = img
        edge = np.mean(edge, axis=-1)
        img = Image.fromarray(edge.astype(np.uint8))
        img.save('crr/my' + str(self.steps) + '.png')
        edge = ndimage.convolve(edge, self.mean, mode='constant', cval=0.0)
        edge = ndimage.convolve(edge, self.k, mode='constant', cval=0.0)
        # edge = feature.canny(edge, sigma=3)*255
        print(np.mean(edge))
        ff = edge.astype(np.uint8)
        img = Image.fromarray(ff)
        img.save('cr/my' + str(self.steps) + '.png')
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
            self.mode = 'forward'

    def forward(self,obs):
        if self.nodanger(obs[0]):
            self.setmode('spin')
        return [1,0]
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
        path,dest = self.planner.bfs(self.map.f,self.map.c,self.map.world_coord_2_map_loc( self.point),pos)
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

    def danger(self,img):
        red = self.get_colours(img, self.red)
        kk = red[50:,35:50]
        aa,bb = np.where(kk)
        return len(aa) > 0
    def nodanger(self,img):
        red = self.get_colours(img, self.red)
        kk = red[50:,:]
        aa,bb = np.where(kk)
        return len(aa) == 0

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
        if self.inspect(obs[0]) and self.force_turn == False and self.init == False and self.danger(obs[0]) == False:
            self.setmode('fetch')

        #points = self.semantic_mask.boundary_mask(obs[0])

        #self.semantic_mask.checkmask(obs[0],self.semantic_mask.boundary_mask(obs[0]))

        #self.map.update_obstacle(self.point, self.theta, self.semantic_mask.grey_mask(obs[0]))
        #self.map.update_boundary(self.point, self.theta, self.semantic_mask.boundary_mask(obs[0]))
        #self.map.update_self(self.point)
        if savefile:
            self.map.savemap()
        if savefile:
            self.test_planner()
        if savefile:
            self.savefig(obs[0])
        self.steps = self.steps + 1

        self.update_convexhull(obs)
        if (self.valid_openning(obs) or self.inspect(obs[0])) and (self.mode == 'spin' or self.mode == 'check'):
            free_r,newp = self.free_reward(obs)
            if self.inspect(obs[0]):
                free_r = self.score(obs[0])
                self.force_turn = True
            if free_r > self.max_r:
                self.max_r = free_r
                self.best_dir = self.theta
                self.attempt_point = newp
        #print(self.mode)
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

        if self.danger(obs[0]) and (obs[1][2]>0.001 or action[1] == 1):
            action = [0,1]
            if obs[1][2] > 5:
                action = [2,1]
            self.setmode('forward')
        if action[1] == 1:
            self.theta = self.update(self.theta, -6 / 180 * self.pi)
        if action[1] == 2:
            self.theta = self.update(self.theta, 6 / 180 * self.pi)




        return action

