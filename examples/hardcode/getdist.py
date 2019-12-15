import pickle
import numpy as np

a,b=pickle.load(open('dist.p','rb'))

print(a,b)

ct = np.zeros(100)
dist = np.zeros(100)
fitted = np.zeros(100)
for i in range(len(a)):
    dist[a[i]] = dist[a[i]] + b[i]
    ct[a[i]]=ct[a[i]]+1

for i in range(100):
    dist[i] = dist[i]/ct[i]

print(dist[42:84])

for i in range(42,84):
    if ct[i] == 0:
        up = 0
        dw = 0
        while ct[i+up] == 0:
            up = up + 1
        while ct[i-dw] == 0:
            dw = dw + 1
        val = dist[i+up] + (dist[i-dw]-dist[i+up])*(up/(up+dw))
        dist[i] = val
print(dist[42:84])
dist[41] = 600


pickle.dump(dist,open('reconstruct.p','wb'))