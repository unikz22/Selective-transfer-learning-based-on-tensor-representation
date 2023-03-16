# -*- coding: utf-8 -*-
import numpy as np
import scipy.io as sio
import torch

device='cpu'

def transdatax(x):
    x=torch.Tensor(x).float().to(device)
    x=x.unsqueeze(2).unsqueeze(3)
    X=x.permute(0,3,2,1)
    return X


def creat_datadast(X,time_steps):
    dataX=[]
    for i in range(X.shape[0]-time_steps):
        a = X[i:(i+time_steps)]
        dataX.append(a)
    return dataX

def MDT_inverst(tucker_feature):
    sampe_num,time_step,fea_dim = tucker_feature.shape
    mdt_inv = [tucker_feature[0,:,:]]
    mdt_inv = mdt_inv + [tucker_feature[i:i+1,-1,:] for i in range(sampe_num)] 
    return torch.cat(mdt_inv, dim=0)


def get_recreate(hO,hI,hC,hM,time_steps):
    #对特征数据进行相空间重构
    chO=[]
    for i in range(len(hO)):
        chh = creat_datadast(hO[i],time_steps)
        chO.append(chh)
   
    chI=[]
    for i in range(len(hI)):
        chh = creat_datadast(hI[i],time_steps)
        chI.append(chh)
    
    chC=[]
    for i in range(len(hC)):
        chh = creat_datadast(hC[i],time_steps)
        chC.append(chh)
    
    chM=[]
    for i in range(len(hM)):
        chh = creat_datadast(hM[i],time_steps)
        chM.append(chh)
    
    return chO,chI,chC,chM