import pickle
import numpy as np
import math
from PIL import Image

class map(object):
    def __init__(self,savepath = 'map/my'):
        self.pic2pos = pickle.load(open('reconstruct.p','rb'))
        self.precision = 100
        self.scaling = self.precision/1600
        self.f = np.zeros((self.precision,self.precision))
        self.c = np.zeros((self.precision,self.precision))
        self.maxc = np.zeros((self.precision,self.precision))
        self.i = np.zeros((self.precision,self.precision))
        self.midpoint = 1600/2
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
            if c < 1.11 and self.maxc[nx, ny] > 5:
                return

            f[nx, ny] = (f[nx, ny] * (
                        self.maxc[nx, ny] * math.sqrt(self.c[nx, ny] / (self.maxc[nx, ny] + 0.0001))) + v * c) / (
                                (self.maxc[nx, ny] * math.sqrt(
                                    self.c[nx, ny] / (self.maxc[nx, ny] + 0.0001))) + c + 0.000001)
            self.c[nx, ny] = self.c[nx, ny] + c
            self.maxc[nx, ny] = max(self.maxc[nx, ny], c)
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
        #for i in points:
        #    self.set(self.f,i[0]+pos[0],i[1]+pos[1],1,self.dist_2_conf(math.sqrt(i[0]*i[0]+i[1]*i[1])+i[2]*10))

        ori = self.self_loc_2_map_loc(pos)

        for i in points:
            tp = self.self_loc_2_map_loc((i[0]+pos[0],i[1]+pos[1]))
            vec = (tp[0]-ori[0],tp[1]-ori[1])
            if vec[0]*vec[0]+vec[1]*vec[1] < 1:
                continue
            uni_vec = self.vec_uni(vec)
            if uni_vec == (0,0):
                continue
            multiples = (abs(vec[0])+abs(vec[1])) / (abs(uni_vec[0])+abs(uni_vec[1]))
            tpdist = math.sqrt(i[0]*i[0]+i[1]*i[1])
            for j in range(int(multiples+2)):
                newpos = (int(ori[0]+j*uni_vec[0]+0.5),int(ori[1]+j*uni_vec[1]+0.5))
                if newpos != tp and j < multiples:
                    self.raw_set(self.f,newpos[0],newpos[1],0,self.dist_2_conf(tpdist * j/ multiples+i[2]*10))
                else:
                    self.raw_set(self.f, newpos[0], newpos[1], 1, self.dist_2_conf(tpdist * j / multiples + i[2] * 10))




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
                                     20*self.dist_2_conf(tpdist * j / multiples + i[2] * 10)) # original 1000 multiplier for conf
                else:
                    kk = self.raw_set(self.f, newpos[0], newpos[1], 1,
                                      20*self.dist_2_conf(tpdist * (multiples - (j-multiples)) / multiples + i[2] * 10))
                    if kk == False:
                        break



    def savemap(self):
        img = np.zeros((self.precision,self.precision,3))
        #img[:,:,0] = self.f*255
        img[:,:,1] = self.f*255
        img[:,:,2] = self.f*255
        img[:,:,0] = img[:,:,0] + self.i*255
        img = Image.fromarray((img).astype(np.uint8))
        img.save(self.savepath + str(self.steps) + '.png')
        self.steps = self.steps+1


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


