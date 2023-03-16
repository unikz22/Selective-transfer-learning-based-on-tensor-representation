
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import scipy.io as sio
import sys
from model.CNNmodel import CNNmodel
from model.AE import AutoEncoder
from model.LSTMModel import LSTMmodel
from dataset.read_data import read_data_XJTU,read_data_PHM,get_value,get_needtarget
from model.deal_fea import transdatax,creat_datadast,MDT_inverst
from dataset.dataset import MyDataset
from model.deal_FM import get_FM
from model.parameter import get_vseq
from model.fun_loss import mmd

from tqdm import tqdm
import tensorly as tl
tl.set_backend('pytorch')
from bhttucker3 import BHTTUCKER3
from BHTTUCKER3_update import BHTTUCKER3_update
device='cpu'

#%%数据加载
path = 'F:/zw_work/work1/dataset/XJTU_HHT/'
patht='F:/zw_work/work1/dataset/PHM_HHT/'
#参数设置
time_steps = 8
Rs = [25,8]
kk = 10
tol = 0.001
Us_mode = 4

precnn = CNNmodel().to(device)
precnn.load_state_dict(torch.load('./CNNresult/pth/precnn_90.pth'))
precnn.eval()

autoencoder = AutoEncoder(2558,[1024,512,128,50]).to(device)
preoutterLSTM = LSTMmodel(input_size=25,hidden_size=100,output_size=1,num_layers=3,time_steps=time_steps).to(device)
preinnerLSTM = LSTMmodel(input_size=25,hidden_size=100,output_size=1,num_layers=3,time_steps=time_steps).to(device)
precageLSTM = LSTMmodel(input_size=25,hidden_size=100,output_size=1,num_layers=3,time_steps=time_steps).to(device)
premixedLSTM = LSTMmodel(input_size=25,hidden_size=100,output_size=1,num_layers=3,time_steps=time_steps).to(device)
targetLSTM = LSTMmodel(input_size=25,hidden_size=100,output_size=1,num_layers=3,time_steps=time_steps).to(device)
 
optimizer = torch.optim.Adam([{'params':autoencoder.parameters(), 'lr':0.001},
                              {'params':preoutterLSTM.parameters(),'lr':0.001},
                              {'params':preinnerLSTM.parameters(),'lr':0.001},
                              {'params':precageLSTM.parameters(),'lr':0.001},
                              {'params':premixedLSTM.parameters(),'lr':0.001},
                              {'params':targetLSTM.parameters(),'lr':0.001}])
lstm={'xo':preoutterLSTM,'xi':preinnerLSTM,'xc':precageLSTM,'xm':premixedLSTM}
indexs={'xo':0,'xi':1,'xc':2,'xm':3}
criterion = nn.MSELoss()

print('start read source data')
xo,xi,xc,xm=read_data_XJTU(path,time_steps) #XO外圈故障   XI内圈故障   XC支架故障  XM混合故障
dataset=MyDataset(xo,xi,xc,xm,block_size=-1,samples=1)
all_data=dataset.all_data
print('start read target data')
phmx,phmxl = read_data_PHM(patht,time_steps)
phmy = get_value(phmxl)
trainyy=phmy[:2]
testyy=phmy[2]
print('start train')
#%%模型训练
ploss=[]
gz=[[],[],[],[]]#保存不同的故障类型的AE特征
LSTMmiddle={}
uso=[[],[],[],[],[],[],[],[]]
usi=[[],[],[]]
usc=[[],[]]
usm=[[],[]]
Us_list={'xo':uso,'xi':usi,'xc':usc,'xm':usm}
tuckerenc={}
pptenc=[[],[],[]]
for epoch in range(10):
    print("epoch:",epoch)
    optimizer.zero_grad()
    data = dataset[0]
    data_dict, _, regress_label_dict=data
    lossae=torch.zeros(1).to(device)
    lstmloss=torch.zeros(1).to(device)
    losstae=torch.zeros(1).to(device)
    tlstmloss=torch.zeros(1).to(device)
    for inepoch in range(91):
        for key in data_dict.keys():
            datas=data_dict[key]  #一个故障类型的所有轴承数据
            regress_labels=regress_label_dict[key]
            prelstm=lstm[key]
            temp=[]
            tuck_enc=[]
            for i in range(len(datas)):
                ti=i%3
                xx=torch.Tensor(datas[i]).float().to(device)
                enc, dec = autoencoder(xx)
                loss_ae = criterion(dec, xx)
                txx=torch.Tensor(phmx[ti]).float().to(device)
                enct, dect = autoencoder(txx)
                loss_tae = criterion(dect, txx)
                losstae = losstae+loss_tae
                lossae=lossae+loss_ae
                tenc = creat_datadast(enc,time_steps)
                tenc=torch.stack(tenc, dim=0)
                ptenc = creat_datadast(enct,time_steps)
                ptenc=torch.stack(ptenc, dim=0)
                if inepoch==0:
                    model_tucker = BHTTUCKER3(tenc.T,ptenc.T,Rs,kk,tol,verbose=0,Us_mode=Us_mode)
                    Us, cores,tcores, _ = model_tucker.run()
                    Us1 = [i.cpu().detach().numpy() for i in Us]
                    Us2 = [torch.tensor(i).to(device) for i in Us1]
                    Us_list[key][i] = Us2
                else:
                    model_tucker2 = BHTTUCKER3_update(tenc.T,ptenc.T,Us_list[key][i])
                    cores,tcores = model_tucker2.run()
                    
                coress = torch.cat([torch.unsqueeze(i,0) for i in cores], dim=0)
                coress = coress.permute(0,2,1).to(device)
                tcoress = torch.cat([torch.unsqueeze(i,0) for i in tcores], dim=0)
                tcoress = tcoress.permute(0,2,1).to(device)
                
                tuck_enc.append(enc.detach().numpy())
                pptenc[ti]=enct
                # aa=torch.unsqueeze(enc,1)
                if inepoch<80:
                    cc=torch.Tensor(regress_labels[i])
                    outl,hhl,middle_out = prelstm(coress)
                    loss_ls=criterion(outl,cc.view(-1,1))
                    lstmloss=lstmloss+loss_ls
                    temp.append(middle_out.detach().numpy())
                if inepoch==60:
                    with torch.no_grad():
                        softmax=nn.Softmax(dim=1)
                        sfm=[torch.mean(softmax(precnn(transdatax(i))[1]),0) for i in phmx]
                        sfm=sum(sfm)/len(sfm)
                        FM=get_FM(tuckerenc,pptenc,sfm)
                        targetLSTM = get_vseq(FM,preoutterLSTM,preinnerLSTM,precageLSTM,premixedLSTM,targetLSTM)
                        # seq = get_vseq(sfm.detach().numpy(),seq1,seq2,seq3,seq4,seq)
                        phmallh=pptenc
                        pptenc=get_needtarget(pptenc,time_steps=8)
                        trainxx=pptenc[:2] 
                if inepoch>=60:
                    for i in range(len(trainxx)):
                        taa=creat_datadast(trainxx[i],time_steps)
                        taa=torch.stack(taa, dim=0)
                        tcc=torch.Tensor(trainyy[i])
                        toutl,thhl,textra_out = targetLSTM(taa)
                        tloss_ls=criterion(toutl,tcc.view(-1,1))
                        tlstmloss=tlstmloss+tloss_ls
                        
                if inepoch%90==0 and inepoch!=0:
                    model_tucker = BHTTUCKER3(tenc.T,ptenc.T,Rs,kk,tol,verbose=0,Us_mode=Us_mode)
                    Us, cores,tcores, _ = model_tucker.run()
                    Us1 = [i.cpu().detach().numpy() for i in Us]
                    Us2 = [torch.tensor(i).to(device) for i in Us1]
                    Us_list[key][i] = Us2
        LSTMmiddle[key]=temp
        tuckerenc[key]=tuck_enc
    print(f'lossae {lossae} losstae {losstae} lstmloss {lstmloss} tlstmloss {tlstmloss}')
    loss=lossae+losstae+0.01*lstmloss+0.01*tlstmloss
    ploss.append(loss.detach().numpy())
    loss.backward()
    optimizer.step()
    print("train loss:",loss.item())
    if epoch%1==0:
        with torch.no_grad():
            sio.savemat('./AEresult/pretrain_{}.mat'.format(epoch),{'lossm':ploss})
            # torch.save(Us_list,'./Uslist_{}.pth'.format(epoch))
            torch.save(autoencoder.state_dict(),'./AEresult/pth/preae_{}.pth'.format(epoch))
            torch.save(preoutterLSTM.state_dict(),'./AEresult/pth/preoutter_{}.pth'.format(epoch))
            torch.save(preinnerLSTM.state_dict(),'./AEresult/pth/preinner_{}.pth'.format(epoch))
            torch.save(precageLSTM.state_dict(),'./AEresult/pth/precage_{}.pth'.format(epoch))
            torch.save(premixedLSTM.state_dict(),'./AEresult/pth/premixed_{}.pth'.format(epoch))# -*- coding: utf-8 -*-
            sio.savemat('./AEresult/middleout_{}.mat'.format(epoch),LSTMmiddle)
            sio.savemat('./AEresult/tuckerae_{}.mat'.format(epoch),tuckerenc)
