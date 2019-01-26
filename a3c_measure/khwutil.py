import numpy as np 




def BuildMaper(L):
    conf = np.zeros(L*L*4)
    for j in range(L):
        for i in range(L):
            conf[((2*j)*(L*2)+(2*i))] = (j*L+i)*4 + 0
            conf[((2*j)*(L*2)+(2*i+1))] = (j*L+i)*4 + 1
            conf[((2*j+1)*(L*2)+(2*i+1))] = (j*L+i)*4 + 2
            conf[((2*j+1)*(L*2)+(2*i))] = (j*L+i)*4 + 3
    return conf.astype(np.int)


def calc_corr(conf,L,Maper):
    tmp = np.array(conf,dtype=np.float)
    tmp = tmp[Maper]
    tmp = tmp.reshape(L*2,L*2)
    print(tmp.dtype)
    ## indexing rule:: A[ry,rx]
    
    ## Cx:
    Crx = []
    for r in range(L):
        Crx.append(np.sum(np.roll(tmp,r,axis=1)*tmp))
    Cry = []
    for r in range(L):
        Cry.append(np.sum(np.roll(tmp,r,axis=0)*tmp))


    return np.array(Crx), np.array(Cry)


def calc_sq(conf,L):
    tmp = np.array(conf,dtype=np.float).reshape(L,L,4)

    Sq = []
    rx,ry = np.meshgrid(np.arange(0,L*2,2).astype(np.float),np.arange(0,L*2,2).astype(np.float))
    for ky in range(L*2):
        for kx in range(L*2):
            Cr = np.mean(np.cos(2.*np.pi/L/2*(kx*rx+ky*ry))*tmp[:,:,0])
            Cr += np.mean(np.cos(2.*np.pi/L/2*(kx*(rx+1)+ky*ry))*tmp[:,:,1])
            Cr += np.mean(np.cos(2.*np.pi/L/2*(kx*(rx+1)+ky*(ry+1)))*tmp[:,:,2])
            Cr += np.mean(np.cos(2.*np.pi/L/2*(kx*rx+ky*(ry+1)))*tmp[:,:,3])
            Cr /= 4
            
            Ci = np.mean(np.sin(2.*np.pi/L/2*(kx*rx+ky*ry))*tmp[:,:,0])
            Ci += np.mean(np.sin(2.*np.pi/L/2*(kx*(rx+1)+ky*ry))*tmp[:,:,1])
            Ci += np.mean(np.sin(2.*np.pi/L/2*(kx*(rx+1)+ky*(ry+1)))*tmp[:,:,2])
            Ci += np.mean(np.sin(2.*np.pi/L/2*(kx*rx+ky*(ry+1)))*tmp[:,:,3])
            Ci /= 4
            Sq.append(Cr**2 + Ci**2)

    #print(Sq)
    return np.array(Sq)

#def BuildSSTL():
#    for j in range(16):
#        for i in range(16):
#            x = i*2+0
#            y = j*2+0
#            s = 2*(y * 2* L + x)
#
#for (int i=0; i < N; ++i) {
#       double x = coor[i].x;
#       double y = coor[i].y;
#       int s = int(2*(y * 2* L + x));
#       indices[s] = i;
#}



#def state_site_to_latt_site(x):



#if __name__=="__main__":
#    maper = BuildMaper()


