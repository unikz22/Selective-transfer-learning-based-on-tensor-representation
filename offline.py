import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import scipy.io as sio
import sys
import matplotlib.pyplot as plt
from model.AE import AutoEncoder
from model.LSTMModel import LSTMmodel
from model.CNNmodel import CNNmodel
from dataset.read_data import read_data_PHM, read_fea_XJTU,get_value,get_needtarget
from model.deal_FM import get_FM
from model.parameter import get_vseq
from model.fun_loss import mmd
from model.deal_fea import transdatax,creat_datadast,MDT_inverst
import tensorly as tl
tl.set_backend('pytorch')
from bhttucker3 import BHTTUCKER3
from BHTTUCKER3_update import BHTTUCKER3_update
device='cpu'

#%%数据加载
pathh='F:/zw_work/work1/code12/AEresult/'
path='F:/zw_work/work1/dataset/PHM_HHT/'

time_steps = 8
Rs = [25,8]
kk = 10
tol = 0.001
Us_mode = 4
phmx,phmxl = read_data_PHM(path,time_steps)
xjtuallh = read_fea_XJTU(pathh)
phmy = get_value(phmxl)
#加载模型
# torch.manual_seed(1000)
preae = AutoEncoder(2558,[1024,512,128,50]).to(device)
preae.eval()

precnn = CNNmodel().to(device)
precnn.load_state_dict(torch.load('./CNNresult/pth/precnn_90.pth'))
precnn.eval()

seq = LSTMmodel(input_size=25,hidden_size=100,output_size=1,num_layers=3,time_steps=time_steps).to(device)
seq1 = torch.load('./AEresult/pth/preoutter_270.pth')
seq2 = torch.load('./AEresult/pth/preinner_270.pth')
seq3 = torch.load('./AEresult/pth/precage_270.pth')
seq4 = torch.load('./AEresult/pth/premixed_270.pth')

# Us_list=torch.load('./USlist_150.pth')
optimizer = torch.optim.Adam([{'params':preae.parameters(), 'lr':0.001},
                              {'params':seq.parameters(),'lr':0.001}])
criterion = nn.MSELoss()

trainyy=phmy[:2]
testyy=phmy[2]
Us_list=[[],[],[]]
ploss=[]
lossls=[]
for epoch in range(300):
    optimizer.zero_grad()
    temp=[]
    lossls2=[0,0]
    lossae=torch.zeros(1).to(device)
    lstmloss=torch.zeros(1).to(device)
    lossmmd=torch.zeros(1).to(device)
    if epoch<=50:
        targetcores=[]
        targetenc=[]
        for i in range(len(phmx)):
            xx=torch.Tensor(phmx[i]).to(device)
            enc, dec = preae(xx)
           
            loss_ae = criterion(dec, xx)
            lossae=lossae+loss_ae
            targetenc.append(enc.detach().numpy())
            tenc = creat_datadast(enc,time_steps)
            tenc=torch.stack(tenc, dim=0)
            if epoch==0:
                model_tucker = BHTTUCKER3(tenc.T,Rs,kk,tol,verbose=0,Us_mode=Us_mode)
                Us, cores, _ = model_tucker.run()
                Us1 = [i.cpu().detach().numpy() for i in Us]
                Us2 = [torch.tensor(i).to(device) for i in Us1]
                Us_list[i] = Us2
            else:
                model_tucker2 = BHTTUCKER3_update(tenc.T,Us_list[i])
                cores = model_tucker2.run()
                
            coress = torch.cat([torch.unsqueeze(i,0) for i in cores], dim=0)
            coress = coress.permute(0,2,1).to(device)
            
            # for p in range(len(xjtuallh)):
            #     loss_mmd = mmd(enc,torch.Tensor(xjtuallh[p]))
            #     lossmmd = lossmmd+loss_mmd
            
            targetcores.append(MDT_inverst(coress).detach().numpy())
        if epoch==50:
            with torch.no_grad():
                softmax=nn.Softmax(dim=1)
                sfm=[torch.mean(softmax(precnn(transdatax(i))[1]),0) for i in phmx]
                sfm=sum(sfm)/len(sfm)
                FM=get_FM(xjtuallh,targetcores,sfm)
                seq = get_vseq(FM,seq1,seq2,seq3,seq4,seq)
                # seq = get_vseq(sfm.detach().numpy(),seq1,seq2,seq3,seq4,seq)
                phmallh=targetcores
                targetcores=get_needtarget(targetcores,time_steps=8)
                trainxx=targetcores[:2]  
    if epoch>50:
        for i in range(len(trainxx)):
            aa=creat_datadast(trainxx[i],time_steps)
            aa=torch.stack(aa, dim=0)
            cc=torch.Tensor(trainyy[i])
            outl,hhl,extra_out = seq(aa)
            loss_ls=criterion(outl,cc.view(-1,1))
            lstmloss=lstmloss+loss_ls
            lossls2[i]=lstmloss.detach().numpy()
            temp.append(extra_out.detach().numpy())
        lossls.append(lossls2)
    # loss=lossae+lstmloss+lossmmd
    
    loss=lossae+lstmloss
    ploss.append(loss.detach().numpy())
    loss.backward()
    optimizer.step()
    print(f'epoch {epoch} train loss {loss.item()}')
    if loss<=1e-05:
        break
sio.savemat('random_10/lossls261_r10.mat',{'lossls':lossls})
sio.savemat('random_10/targetmiddle261_r10.mat',{'tgm':temp})
sio.savemat('random_10/targetenc261_r10.mat',{'enc':targetenc})
sio.savemat('random_10/phmmmd261_r10.mat',{'p22':phmallh[0],'p26':phmallh[1],'p21':phmallh[2]})
# sio.savemat('random_10/phmmmd162_r10.mat',{'p22':phmallh[2],'p26':phmallh[1],'p21':phmallh[0]})
with torch.no_grad():
    aa=creat_datadast(targetcores[2],time_steps)
    aa=torch.stack(aa, dim=0)
    cc=torch.Tensor(testyy)
    pred,hhl,test_extra_out = seq(aa)
    testloss_lstm=criterion(pred,cc.view(-1,1))
    pred_y = pred.cpu().detach().numpy()
    print('testloss:',testloss_lstm)
    sio.savemat('random_10/npred261_r10.mat',{'pred110':pred_y})
plt.plot(pred_y)
plt.plot(testyy)
plt.show()