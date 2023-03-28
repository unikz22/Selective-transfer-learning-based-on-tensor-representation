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

pathh='F:/zw_work/work1/code7/cnntre/'

def read_sourceh(path):
    cho11=sio.loadmat(path+'CNNoutter_20.mat')['cho11']
    cho12=sio.loadmat(path+'CNNoutter_20.mat')['cho12']
    cho13=sio.loadmat(path+'CNNoutter_20.mat')['cho13']
    cho24=sio.loadmat(path+'CNNoutter_20.mat')['cho24']
    cho31=sio.loadmat(path+'CNNoutter_20.mat')['cho31']
    
    chi21=sio.loadmat(path+'CNNinner_20.mat')['chi21']
    chi33=sio.loadmat(path+'CNNinner_20.mat')['chi33']
    chi34=sio.loadmat(path+'CNNinner_20.mat')['chi34']
    
    chc14=sio.loadmat(path+'CNNcage_20.mat')['chc14']
    chc23=sio.loadmat(path+'CNNcage_20.mat')['chc23']
    
    chm15=sio.loadmat(path+'CNNmixed_20.mat')['chm15']
    chm32=sio.loadmat(path+'CNNmixed_20.mat')['chm32']
    
    hho=[cho11,cho12,cho13,cho24,cho31]
    hhi=[chi21,chi33,chi34]
    hhc=[chc14,chc23]
    hhm=[chm15,chm32]
    return hho,hhi,hhc,hhm

def GetMICY(data):
    x=[]
    for i in range(1,len(data)+1):
        x.append(i)
    return x

def get_pca(h_outter,h_inner,h_cage,h_mixed,thPFM):
    hh_outter=[]
    pca=PCA(n_components=1)
    for i in range(h_outter):
        pca.fit(h_outter[i])
        ph=pca.transform(h_outter[i])
        hh_outter.append(ph)
    hh_inner=[]
    for i in range(h_inner):
        pca.fit(h_inner[i])
        ph=pca.transform(h_inner[i])
        hh_inner.append(ph)
    hh_cage=[]
    for i in range(h_cage):
        pca.fit(h_cage[i])
        ph=pca.transform(h_cage[i])
        hh_cage.append(ph)
    hh_mixed=[]
    for i in range(h_mixed):
        pca.fit(h_mixed[i])
        ph=pca.transform(h_mixed[i])
        hh_mixed.append(ph)
    hh_phm=[]
    for i in range(thPFM):
        pca.fit(thPFM[i])
        ph=pca.transform(thPFM[i])
        hh_phm.append(ph)
    pca.fit(hh_outter)
    hh_outter=pca.transform(hh_outter)
    pca.fit(hh_inner)
    hh_inner=pca.transform(hh_inner)
    pca.fit(hh_cage)
    hh_cage=pca.transform(hh_cage)
    pca.fit(hh_mixed)
    hh_mixed=pca.transform(hh_mixed)
    pca.fit(hh_phm)
    hh_phm=pca.transform(hh_phm)
    return hh_outter,hh_inner,hh_cage,hh_mixed,hh_phm

def get_MIC(hh_outter,hh_inner,hh_cage,hh_mixed,hh_phm2):
    pca1 = np.array(GetMICY(hh_outter))
    pca2 = np.array(GetMICY(hh_inner))
    pca3 = np.array(GetMICY(hh_cage))
    pca4 = np.array(GetMICY(hh_mixed))
    pca5 = np.array(GetMICY(hh_phm2))
    RMS_X = [hh_outter,hh_inner,hh_cage,hh_mixed,hh_phm2]
    RMS_Y = [pca1,pca2,pca3,pca4,pca5]
    
    mine = MINE(alpha=0.6, c=15)
    xmic=[]
    for Num in range(0,5):
        x = RMS_X[Num].reshape(len(RMS_X[Num]),)
        y = RMS_Y[Num].reshape(len(RMS_Y[Num]),)
        mine.compute_score(x,y)
        xmic.append(mine.mic())
    sio.savemat("MIC_data.mat",{'PHM_MIC':xmic})
    return xmic

def get_DTW(hh_outter,hh_inner,hh_cage,hh_mixed,hh_phm2):
    Y_ALL = [hh_outter,hh_inner,hh_cage,hh_mixed]
    X_ALL = [hh_phm2]
    DTW=np.ones((len(X_ALL),len(Y_ALL)))
    for i in range(len(X_ALL)):
        for j in range(len(Y_ALL)):
            a = np.double(X_ALL[i])
            b = np.double(Y_ALL[j])
            DTW[i,j]=dtw.distance(a, b)
    sio.savemat('dtw_data.mat',{'dtw':DTW})
    return DTW

def get_FM(MIC,DTW,sfm):
    MICX=MIC[0][4:]
    MICY=MIC[0][:4]
    MIC_value=np.ones((len(MICX),len(MICY)))
    for i in range(len(MICX)):
        for j in range(len(MICY)):
            MIC_value[i,j]=abs(MICX[i]-MICY[j])*DTW[i,j]/(MICX[i]*MICY[j])
            print(MIC_value)
    sio.savemat('MICvalue.mat',{'MIC_value':MIC_value})
    return MIC_value