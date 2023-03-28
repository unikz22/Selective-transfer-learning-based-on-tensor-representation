# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as sio
import numpy as np
from minepy import MINE
import seaborn as sns
import pandas as pd
from dtaidistance import dtw
from sklearn.decomposition import PCA
from model.CNNmodel import CNNmodel

def get_FM(xjh,target,sfm):
    xjtuallh=read_fea_XJTU(xjh)
    DM = torch.Tensor(get_DM(xjtuallh,target))
    fm=[]
    for i in range(len(sfm)):
    # for i in range(4):
        fm.append(sfm[i]/DM[0][i])
        # fm.append(sfm/DM[0][i])
    fw=[]
    for i in range(len(fm)):
        dfm=fm[i]/sum(fm)
        fw.append(dfm.detach().numpy())
        print(fw[i])
    sio.savemat('./tradeoff/beta1/DFMvalue162.mat',{'fw2':fw})
    # sio.savemat('softmax261.mat',{'sfm1':sfm.detach().numpy()})
    return fw

def get_DM(xjtuallh,target):
    pcao,pcai,pcac,pcam,pcat = get_pca(xjtuallh,target)
    xmic = get_MIC(pcao,pcai,pcac,pcam,pcat)
    xdtw = get_DTW(pcao,pcai,pcac,pcam,pcat)
    MICX=[xmic[4]]
    MICY=xmic[:4]
    MIC_value=np.ones((len(MICX),len(MICY)))
    for i in range(len(MICX)):
        for j in range(len(MICY)):
            MIC_value[i,j]=abs(MICX[i]-MICY[j])*xdtw[i,j]/(MICX[i]*MICY[j])
            # print(MIC_value[i,j])
    # sio.savemat('MICvalue261.mat',{'MIC_value':MIC_value})
    return MIC_value

def GetMICY(data):
    x=[]
    for i in range(1,len(data)+1):
        x.append(i)
    return x

def get_pca(xjtuallh,target):
    pca=PCA(n_components=1)
    xjpca=[]
    for i in range(len(xjtuallh)):
        pca.fit(xjtuallh[i][-200:,:])
        ph=pca.transform(xjtuallh[i][-200:,:])
        xjpca.append(ph)
    phmpca=[]
    for i in range(len(target)):
        pca.fit(target[i][-200:,:])
        ph=pca.transform(target[i][-200:,:])
        phmpca.append(ph)
    
    indexo=[0,1,2,6,8,9,10,14]
    indexi=[5,12,13]
    indexc=[3,7]
    indexm=[4,11,]
    hh_outter=[xjpca[i] for i in indexo]
    hh_inner=[xjpca[i] for i in indexi]
    hh_cage=[xjpca[i] for i in indexc]
    hh_mixed=[xjpca[i] for i in indexm]
    pca.fit(np.concatenate(hh_outter,axis=1))
    hh_outter=pca.transform(np.concatenate(hh_outter,axis=1))
    pca.fit(np.concatenate(hh_inner,axis=1))
    hh_inner=pca.transform(np.concatenate(hh_inner,axis=1))
    pca.fit(np.concatenate(hh_cage,axis=1))
    hh_cage=pca.transform(np.concatenate(hh_cage,axis=1))
    pca.fit(np.concatenate(hh_mixed,axis=1))
    hh_mixed=pca.transform(np.concatenate(hh_mixed,axis=1))
    pca.fit(np.concatenate(phmpca,axis=1))
    hh_phm=pca.transform(np.concatenate(phmpca,axis=1))
    return hh_outter,hh_inner,hh_cage,hh_mixed,hh_phm

def get_MIC(pcao,pcai,pcac,pcam,pcat):
    pca1 = np.array(GetMICY(pcao))
    pca2 = np.array(GetMICY(pcai))
    pca3 = np.array(GetMICY(pcac))
    pca4 = np.array(GetMICY(pcam))
    pca5 = np.array(GetMICY(pcat))
    RMS_X = [pcao,pcai,pcac,pcam,pcat]
    RMS_Y = [pca1,pca2,pca3,pca4,pca5]
    
    mine = MINE(alpha=0.6, c=15)
    xmic=[]
    for Num in range(0,5):
        x = RMS_X[Num].reshape(len(RMS_X[Num]),)
        y = RMS_Y[Num].reshape(len(RMS_Y[Num]),)
        mine.compute_score(x,y)
        xmic.append(mine.mic())
    # sio.savemat("MIC_data162.mat",{'PHM_MIC':xmic})
    return xmic

def get_DTW(pcao,pcai,pcac,pcam,pcat):
    Y_ALL = [pcao,pcai,pcac,pcam]
    X_ALL = [pcat]
    DTW=np.ones((len(X_ALL),len(Y_ALL)))
    for i in range(len(X_ALL)):
        for j in range(len(Y_ALL)):
            a = np.double(X_ALL[i])
            b = np.double(Y_ALL[j])
            DTW[i,j]=dtw.distance(a, b)
    # sio.savemat('dtw_data162.mat',{'dtw':DTW})
    return DTW
    
def read_fea_XJTU(xjh):
    hx11=xjh['xo'][0]
    hx12=xjh['xo'][1]
    hx13=xjh['xo'][2]
    hx14=xjh['xc'][0]
    hx15=xjh['xm'][0]
    
    hx21=xjh['xi'][0]
    hx22=xjh['xo'][3]
    hx23=xjh['xc'][1]
    hx24=xjh['xo'][4]
    hx25=xjh['xo'][5]
    
    hx31=xjh['xo'][6]
    hx32=xjh['xm'][1]
    hx33=xjh['xi'][1]
    hx34=xjh['xi'][2]
    hx35=xjh['xo'][7]
    
    xjtuallh=[hx11,hx12,hx13,hx14,hx15,hx21,hx22,hx23,hx24,hx25,hx31,hx32,hx33,hx34,hx35]
    return xjtuallh