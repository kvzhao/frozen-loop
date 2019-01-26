import numpy as np 
import os,sys
import pickle as pkl

Log_accuFwd = []
Log_accuRev = []
for i in range(1024):  
    tmp = []
    tmp2= []
    for s in range(8):
        if not os.path.exists('log_accu/Fwd.%d.npd.s%d.npy'%(i,s)):
            continue
        tmp.append(np.load('log_accu/Fwd.%d.npd.s%d.npy'%(i,s)))
        tmp2.append(np.load('log_accu/Rev.%d.npd.s%d.npy'%(i,s)))
    Log_accuFwd.append(np.concatenate(tmp))
    Log_accuRev.append(np.concatenate(tmp2))


f = open("log_accu/logFwdProb.pkl","wb")
pkl.dump(Log_accuFwd,f)
f.close()
f = open("log_accu/logRevProb.pkl","wb")
pkl.dump(Log_accuRev,f)
f.close()

#FwdCount = [len(Log_accuFwd[i]) for i in range(1024)]
#RevCount = [len(Log_accuRev[i]) for i in range(1024)]


#np.save("log_accu/distrib",np.array(FwdCount))
#print("[Saved] loop-size distribution")





#ihas = np.argwhere(np.array(FwdCount)>0).flatten()

#for i in ihas:
#    print(i)
#    print("loopsize = %d"%(i))
#    #print(Log_accuFwd[i])
#    print(FwdCount[i])
