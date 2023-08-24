#tucker-CNN-pre_model放在一个整体框架中
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import scipy.io as sio
from tqdm import tqdm
from model.CNNmodel import CNNmodel
from dataset.read_data import read_data_XJTU
from model.fun_loss import L2distance,lndistance
from dataset.dataset import MyDataset
from model.deal_fea import transdatax

device='cpu'

#%%数据加载
path = './dataset/XJTU_HHT/'
#参数设置
time_steps = 8

MyConvent = CNNmodel().to(device)

optimizer = torch.optim.Adam([{'params':MyConvent.parameters(), 'lr':0.001}])
loss_func = nn.CrossEntropyLoss()
indexs={'xo':0,'xi':1,'xc':2,'xm':3}

print('start read data')
xo,xi,xc,xm=read_data_XJTU(path,time_steps) #XO外圈故障   XI内圈故障   XC支架故障  XM混合故障
_,xtg=read_data_UNSW(patht,time_steps)
xt=xtg[:1]
targetdata=UNSW_split(xt,block_size=80)
dataset=MyDataset(xo,xi,xc,xm,block_size=80,samples=50)
all_data=dataset.all_data
print('start train')
#%%模型训练
ploss=[]

def mmd(x,y):
     return torch.sum(torch.pow((torch.mean(x,dim = 0) - torch.mean(y,dim = 0)),2))

for epoch in range(30):
    print("epoch:",epoch)
    # CNN分类
    iters=len(all_data)
    for _iter in tqdm(range(iters)):
        optimizer.zero_grad()
    
        cnnloss=torch.zeros(1).to(device)
        lstmloss=torch.zeros(1).to(device)
        
        data=dataset[_iter]  #一块数据，包括数据，分类标签，回归标签
        data_dict,class_label_dict,_=data
        
        gz=[[],[],[],[]]#保存不同的故障类型
        gztre=[[],[],[],[]]
        #不同故障类型训练不同的LSTM模型
        for key in data_dict.keys():
            datas=data_dict[key]  #一个故障类型的所有轴承数据
            class_labels=class_label_dict[key]
            
            for i in range(len(datas)):
                xx=transdatax(datas[i])
                hhx, outc, trehh = MyConvent(xx)
                hhxt,_,_ = MyConvent(xxt)
                adp_mmd = mmd(hhx,hhxt)
                gz[indexs[key]].append(hhx)
                gztre[indexs[key]].append(trehh)
                cnn_loss = loss_func(outc, torch.Tensor(class_labels[i][:,0]).long())
                cnnloss=cnnloss+cnn_loss
                mmdloss=mmdloss+adp_mmd
                
        closs=torch.zeros(1).to(device)#类间损失
        ljhhx=[torch.cat(gz[0][:2],axis=0),torch.cat(gz[1][:2],axis=0),torch.cat(gz[2][:2],axis=0),torch.cat(gz[3][:2],axis=0)]
        for i in range(4):
            for j in range(i+1,4):
                cl = L2distance(ljhhx[i],ljhhx[j])
                closs = closs+cl
        closs=1/closs
        
        
        lossln=torch.zeros(1).to(device)#类内损失
        lnhhx=[torch.cat(gz[0],axis=0),torch.cat(gz[1],axis=0),torch.cat(gz[2],axis=0),torch.cat(gz[3],axis=0)]
        for i in range(4):
            loss_ln = lndistance(lnhhx[i])
            lossln =lossln+loss_ln

        
        print(f'epoch {epoch} cnnloss {cnnloss} closs {closs} lossln {lossln}')
        loss=cnnloss+closs+lossln+0.1*mmdloss         
        ploss.append(loss.detach().numpy())
        loss.backward()
        optimizer.step()
        print("train loss:",loss.item())
    torch.save(MyConvent.state_dict(),'./cnnresult/pth/precnn_{}.pth'.format(epoch))
    if epoch%2==0 and epoch!=0:
        with torch.no_grad():
            sio.savemat('./cnnresult/pretrain_{}.mat'.format(epoch),{'lossm':ploss})
            transxo = [transdatax(i) for i in xo]
            hhxo = [MyConvent(i)[0] for i in transxo]
            
            sio.savemat('./cnnresult/CNNoutter_{}.mat'.format(epoch),{'cho11':hhxo[0].cpu().detach().numpy(),'cho12':hhxo[1].cpu().detach().numpy(),\
                                                                    'cho13':hhxo[2].cpu().detach().numpy(),'cho24':hhxo[3].cpu().detach().numpy(),
                                                                    'cho31':hhxo[4].cpu().detach().numpy()})
            transxi = [transdatax(i) for i in xi]
            hhxi = [MyConvent(i)[0] for i in transxi]
            sio.savemat('./cnnresult/CNNinner_{}.mat'.format(epoch),{'chi21':hhxi[0].cpu().detach().numpy(),'chi33':hhxi[1].cpu().detach().numpy(),\
                                                                    'chi34':hhxi[2].cpu().detach().numpy()})
            transxc = [transdatax(i) for i in xc]
            hhxc = [MyConvent(i)[0] for i in transxc]
            sio.savemat('./cnnresult/CNNcage_{}.mat'.format(epoch),{'chc14':hhxc[0].cpu().detach().numpy(),'chc23':hhxc[1].cpu().detach().numpy()})
            
            transxm = [transdatax(i) for i in xm]
            hhxm = [MyConvent(i)[0] for i in transxm]
            sio.savemat('./cnnresult/CNNmixed_{}.mat'.format(epoch),{'chm15':hhxm[0].cpu().detach().numpy(),'chm32':hhxm[1].cpu().detach().numpy()})
    

        
