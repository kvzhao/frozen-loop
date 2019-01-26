import numpy as np 
import os,sys
import pickle as pkl

Crx = []
Cry = []
for s in np.arange(7,12,1):
    if not os.path.exists('log_obs/Cx.npd.s%d.npy'%(s)):
        continue
    if not os.path.exists('log_obs/Cy.npd.s%d.npy'%(s)):
        continue
    Crx.append(np.load('log_obs/Cx.npd.s%d.npy'%(s)))
    Cry.append(np.load('log_obs/Cy.npd.s%d.npy'%(s)))

Crx = np.array(Crx)
Cry = np.array(Cry)
Crx = np.mean(Crx,axis=0)
Cry = np.mean(Cry,axis=0)

print(Crx)
print(Cry)

f = open("log_obs/Cx.pkl","wb")
pkl.dump(Crx,f)
f.close()
f = open("log_obs/Cy.pkl","wb")
pkl.dump(Cry,f)
f.close()

