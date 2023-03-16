# -*- coding: utf-8 -*-
import torch

device='cpu'

def L2distance(X,Y):
    #类间距离
    line,row=X.shape
    st=0
    for i in range(line):
        for j in range(row):
            ss=torch.pow((X[i][j]-Y[i][j]),2)
            st=st+ss
    st=torch.sqrt(st)
    return st.float()

def lndistance(hx):
    #类内距离
    lno=torch.zeros(1).to(device)
    for i in range(hx.shape[0]-1):
        olm=(torch.mean(hx[:i+1,:])+torch.mean(hx[i+1:,:]))/2
        xx=hx[i]-olm
        preln=torch.dot(xx,xx)
        lno=lno+preln
    lno=lno/(hx.shape[0]-1)
    return lno

def Monotonicity(hx1,smt=10):
    #单调性
    dhdo=torch.zeros(1).to(device)
    dhxo=torch.zeros(1).to(device)
    mono=torch.zeros(1).to(device)
    hx_len=int(hx1.shape[0]/smt)
    hx_block=[hx1[i*smt:(i+1)*smt] for i in range(hx_len)]
    hx_smooth=[torch.mean(i) for i in hx_block]
    for i in range(len(hx_smooth)):
        if hx_smooth[i]<=hx_smooth[i+1]:
            dhdo=dhdo+1
        else:
            dhxo=dhxo+1
    Mon=torch.abs(dhdo-dhxo)/(len(hx_smooth))
    mono=mono+Mon
    return mono

def mmd(x,y):
     return torch.sum(torch.pow((torch.mean(x,dim = 0) - torch.mean(y,dim = 0)),2))

def trend(trehX):
    #趋势性
    treo=torch.zeros(1).to(device)
    for i in range(len(trehX)):
        if torch.any(torch.isnan(trehX)):
            print('22222')
        Kc=trehX.shape[0]
        hh=torch.squeeze(trehX,1)
        tt=torch.linspace(1,2,Kc)
        part1=Kc*torch.dot(hh,tt)
        part2=torch.sum(hh)*torch.sum(tt)
        part3=torch.abs(Kc*torch.sum(hh**2)-torch.pow(torch.sum(hh),2))
        part4=Kc*torch.sum(tt**2)-torch.pow(torch.sum(tt),2)
        tre=torch.sqrt(part3*part4)/torch.abs(part1-part2)
        if torch.any(torch.isnan(tre)):
            print('33333')
        treo=treo+tre    

    return treo