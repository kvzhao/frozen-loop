import numpy as np 
import os,sys
import matplotlib.pyplot as plt
from matplotlib import rc

Log_accuFwd = []
Log_accuRev = []
for i in range(1024):   
    tmp = []
    tmp2= []
    for s in range(3):
        tmp.append(np.load('log_accu/Fwd.%d.npd.s%d.npy'%(i,s)))
        tmp2.append(np.load('log_accu/Rev.%d.npd.s%d.npy'%(i,s)))
    Log_accuFwd.append(np.concatenate(tmp))
    Log_accuRev.append(np.concatenate(tmp2))


FwdCount = [len(Log_accuFwd[i]) for i in range(1024)]
RevCount = [len(Log_accuRev[i]) for i in range(1024)]

ihas = np.argwhere(np.array(FwdCount)>0).flatten()

plt.figure(1)
for i in ihas:
    #print(i)
    #print("loopsize = %d"%(i))
    #print(Log_accuFwd[i])
    plt.plot(np.ones(len(Log_accuFwd[i]))*i,Log_accuFwd[i],'.r')
    plt.plot(np.ones(len(Log_accuRev[i]))*i,Log_accuRev[i],'.b')
    plt.plot(np.ones(len(Log_accuRev[i]))*i,Log_accuFwd[i]-Log_accuRev[i],'.k')
plt.grid(1)
plt.xlabel('loop-size')
plt.ylabel(r'$log P$')
plt.show()

