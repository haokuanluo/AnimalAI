import matplotlib
matplotlib.use("TKAgg")
from matplotlib import pyplot as plt
import numpy as np
import pickle

def moving_average(a, n=50) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
def get(path):
    x=[]
    y=[]
    z=[]
    l = []
    l1 = []
    l2 = []
    l3 = []
    l4 = []
    s = 0
    with open(path,'rb') as f:
        a = pickle.load(f)
    for i in a:
        i1,i2,i3,i4,i5,i6=i
        x.append(i1)
        y.append(i2)
        z.append(i3)
        l.append(s)
        l1.append(i4)
        l2.append(i5)
        l3.append(i6)
        l4.append(0)
        s=s+1
    return x,y,z,l,l1,l2,l3,l4


x,y,z,l,l1,l2,l3,l4 = get('navigation_loss.p')


fig = plt.figure()
wind = 20

ax1 = fig.add_subplot(421)
ax1.plot(moving_average(x,n=wind)) # lower the better
print(y)

ax2 = fig.add_subplot(422)
ax2.plot(moving_average(y,n=wind)) # higher the better

ax3 = fig.add_subplot(423)
ax3.plot(moving_average(z,n=wind))  # higher the better

ax4 = fig.add_subplot(424)
ax4.plot(moving_average(l1,n=wind)) # lower

ax5 = fig.add_subplot(425)
ax5.plot(moving_average(l2,n=wind)) #lower

ax6 = fig.add_subplot(426)
ax6.plot(moving_average(l3,n=wind)) #lower

ax7 = fig.add_subplot(427)
ax7.plot(moving_average(l4,n=wind)) #lower


plt.show()