# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np

device='cpu'

loss_fune = nn.MSELoss()
def preoutter(hO2,Yall_outter,preoutterLSTM):
    loss_lso = torch.zeros(1).to(device)
    for i in range(len(hO2)):
        cc = np.double(Yall_outter[i])
        cc = torch.from_numpy(cc).to(device)
        aa = hO2[i]
        out,hh = preoutterLSTM(aa)

    loss_lso = loss_lso+loss_fune(out,cc.T.float())
    loss_lso = loss_lso/len(hO2)
    return out,hh,loss_lso

def preinner(hI2,Yall_inner,preinnerLSTM):
    loss_lsi = torch.zeros(1).to(device)
    for i in range(len(hI2)):
        cc = np.double(Yall_inner[i])
        cc = torch.from_numpy(cc).to(device)
        aa = torch.Tensor(hI2[i])
        out,hh = preinnerLSTM(aa)
        loss_lsi = loss_lsi+loss_fune(out,cc.T.float())
    loss_lsi = loss_lsi/len(hI2)
    return out,hh,loss_lsi

def precage(hC2,Yall_cage,precageLSTM):
    loss_lsc = torch.zeros(1).to(device)
    for i in range(len(hC2)):
        cc = np.double(Yall_cage[i])
        cc = torch.from_numpy(cc).to(device)
        aa = torch.Tensor(hC2[i])
        out,hh = precageLSTM(aa)
        loss_lsc = loss_lsc+loss_fune(out,cc.T.float())
    loss_lsc = loss_lsc/len(hC2)
    return out,hh,loss_lsc

def premixed(hM2,Yall_mixed,premixedLSTM):
    loss_lsm = torch.zeros(1).to(device)
    for i in range(len(hM2)):
        cc = np.double(Yall_mixed[i])
        cc = torch.from_numpy(cc).to(device)
        aa = torch.Tensor(hM2[i])
        out,hh = premixedLSTM(aa)
        loss_lsm = loss_lsm+loss_fune(out,cc.T.float())
    loss_lsm = loss_lsm/len(hM2)
    return out,hh,loss_lsm
