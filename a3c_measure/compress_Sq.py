import numpy as np 
import os,sys
import pickle as pkl

Sq = []
for s in np.arange(7,12,1):
    if not os.path.exists('log_obs/Sq.npd.s%d.npy'%(s)):
        continue
    Sq.append(np.load('log_obs/Sq.npd.s%d.npy'%(s)))

Sq = np.array(Sq)
Sq = np.mean(Sq,axis=0)

print(len(Sq))

f = open("log_obs/Sq.pkl","wb")
pkl.dump(Sq,f)
f.close()

