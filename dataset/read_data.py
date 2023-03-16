# -*- coding: utf-8 -*-
import numpy as np
import scipy.io as sio
import torch


def read_data_XJTU(path,time_steps):
    
    X11 = sio.loadmat(path+'BJP_XJTUB1_1.mat')['BJP'][1017-time_steps:1476,:].astype(np.float32)
    X12 = sio.loadmat(path+'BJP_XJTUB1_2.mat')['BJP'][796-time_steps:1932,:].astype(np.float32)
    X13 = sio.loadmat(path+'BJP_XJTUB1_3.mat')['BJP'][1896-200-time_steps:1896,:].astype(np.float32)
    X14 = sio.loadmat(path+'BJP_XJTUB1_4.mat')['BJP'][1464-200-time_steps:1464,:].astype(np.float32)
    X15 = sio.loadmat(path+'BJP_XJTUB1_5.mat')['BJP'][624-200-time_steps:624,:].astype(np.float32)
    
    X21 = sio.loadmat(path+'BJP_XJTUB2_1.mat')['BJP'][5892-200-time_steps:5892,:].astype(np.float32)
    X22 = sio.loadmat(path+'BJP_XJTUB2_2.mat')['BJP'][1024-time_steps:1932,:].astype(np.float32)
    X23 = sio.loadmat(path+'BJP_XJTUB2_3.mat')['BJP'][4289-time_steps:6396,:].astype(np.float32)
    X24 = sio.loadmat(path+'BJP_XJTUB2_4.mat')['BJP'][504-200-time_steps:504,:].astype(np.float32)
    X25 = sio.loadmat(path+'BJP_XJTUB2_5.mat')['BJP'][2428-time_steps:4068,:].astype(np.float32)
    
    X31 = sio.loadmat(path+'BJP_XJTUB3_1.mat')['BJP'][29557-time_steps:,:].astype(np.float32)
    X32 = sio.loadmat(path+'BJP_XJTUB3_2.mat')['BJP'][26623-time_steps:,:].astype(np.float32)
    X33 = sio.loadmat(path+'BJP_XJTUB3_3.mat')['BJP'][4117-time_steps:4452,:].astype(np.float32)
    X34 = sio.loadmat(path+'BJP_XJTUB3_4.mat')['BJP'][17546-time_steps:,:].astype(np.float32)
    X35 = sio.loadmat(path+'BJP_XJTUB3_5.mat')['BJP'][285-time_steps:1368,:].astype(np.float32)
    
    # XN11 = sio.loadmat(path+'XJB1_1.mat')['XB1_1'][:400,:].astype(np.float32)
    # XN23 = sio.loadmat(path+'XJB2_3.mat')['XB2_3'][:400,:].astype(np.float32)
    # XN33 = sio.loadmat(path+'XJB3_3.mat')['XB3_3'][:400,:].astype(np.float32)
    # XN32 = sio.loadmat(path+'XJB3_2.mat')['XB3_2'][:400,:].astype(np.float32)
    
    # X11 = sio.loadmat(path+'XJB1_1.mat')['XB1_1'][1017-time_steps:1476,:].astype(np.float32)
    # X12 = sio.loadmat(path+'XJB1_2.mat')['XB1_2'][796-time_steps:1932,:].astype(np.float32)
    # X13 = sio.loadmat(path+'XJB1_3.mat')['XB1_3'][1896-200-time_steps:1896,:].astype(np.float32)
    # X14 = sio.loadmat(path+'XJB1_4.mat')['XB1_4'][1464-200-time_steps:1464,:].astype(np.float32)
    # X15 = sio.loadmat(path+'XJB1_5.mat')['XB1_5'][624-200-time_steps:624,:].astype(np.float32)
    
    # X21 = sio.loadmat(path+'XJB2_1.mat')['XB2_1'][5892-200-time_steps:5892,:].astype(np.float32)
    # X22 = sio.loadmat(path+'XJB2_2.mat')['XB2_2'][1024-time_steps:1932,:].astype(np.float32)
    # X23 = sio.loadmat(path+'XJB2_3.mat')['XB2_3'][4289-time_steps:6396,:].astype(np.float32)
    # X24 = sio.loadmat(path+'XJB2_4.mat')['XB2_4'][504-200-time_steps:504,:].astype(np.float32)
    # X25 = sio.loadmat(path+'XJB2_5.mat')['XB2_5'][2428-time_steps:4068,:].astype(np.float32)
    
    # X31 = sio.loadmat(path+'XJB3_1.mat')['XB3_1'][4101-time_steps:,:].astype(np.float32)
    # X32 = sio.loadmat(path+'XJB3_2.mat')['XB3_2'][1671-time_steps:,:].astype(np.float32)
    # X33 = sio.loadmat(path+'XJB3_3.mat')['XB3_3'][4117-time_steps:4452,:].astype(np.float32)
    # X34 = sio.loadmat(path+'XJB3_4.mat')['XB3_4'][4366-time_steps:,:].astype(np.float32)
    # X35 = sio.loadmat(path+'XJB3_5.mat')['XB3_5'][285-time_steps:1368,:].astype(np.float32)
    
    # XN=[XN11,XN23,XN33,XN32]
    XO=[X11,X12,X13,X22,X24,X25,X31,X35]
    XI=[X21,X33,X34]
    XC=[X14,X23]
    XM=[X15,X32]
    return XO,XI,XC,XM

def read_data_PHM(path,time_steps):
    X11=sio.loadmat(path+'H1_1.mat')['BJP'][2739-time_steps:2790].astype(np.float32)
    X12=sio.loadmat(path+'H1_2.mat')['BJP'][820-time_steps:871].astype(np.float32)
    X13=sio.loadmat(path+'H1_3.mat')['BJP'][2301-time_steps:2348].astype(np.float32)
    X14=sio.loadmat(path+'H1_4.mat')['BJP'][1082-time_steps:1170].astype(np.float32)
    X15=sio.loadmat(path+'H1_5.mat')['BJP'][2429-time_steps:2463].astype(np.float32)
    X16=sio.loadmat(path+'H1_6.mat')['BJP'][2410-time_steps:2448].astype(np.float32)
    X17=sio.loadmat(path+'H1_7.mat')['BJP'][2205-time_steps:2259].astype(np.float32)
    
    X21=sio.loadmat(path+'H2_1.mat')['BJP'][870-time_steps:911].astype(np.float32)
    X22=sio.loadmat(path+'H2_2.mat')['BJP'][742-time_steps:797].astype(np.float32)
    Xa21=sio.loadmat(path+'H2_1.mat')['BJP'].astype(np.float32)
    Xa22=sio.loadmat(path+'H2_2.mat')['BJP'].astype(np.float32)
    X23=sio.loadmat(path+'H2_3.mat')['BJP'][1942-time_steps:1955].astype(np.float32)
    X24=sio.loadmat(path+'H2_4.mat')['BJP'][735-time_steps:751].astype(np.float32)
    X25=sio.loadmat(path+'H2_5.mat')['BJP'][2296-time_steps:2311].astype(np.float32)
    X26=sio.loadmat(path+'H2_6.mat')['BJP'][683-time_steps:701].astype(np.float32)
    Xa26=sio.loadmat(path+'H2_6.mat')['BJP'].astype(np.float32)
    X27=sio.loadmat(path+'H2_7.mat')['BJP'][220-time_steps:229].astype(np.float32)
    
    xneed1=[Xa22,Xa26,Xa21]
    xneed2=[X22,X26,X21]
    return xneed1,xneed2
    
def read_fea_XJTU(path):
    hx11=sio.loadmat(path+'tuckerae_270.mat')['xo'][0][0]
    hx12=sio.loadmat(path+'tuckerae_270.mat')['xo'][0][1]
    hx13=sio.loadmat(path+'tuckerae_270.mat')['xo'][0][2]
    hx14=sio.loadmat(path+'tuckerae_270.mat')['xc'][0][0]
    hx15=sio.loadmat(path+'tuckerae_270.mat')['xm'][0][0]
    
    hx21=sio.loadmat(path+'tuckerae_270.mat')['xi'][0][0]
    hx22=sio.loadmat(path+'tuckerae_270.mat')['xo'][0][3]
    hx23=sio.loadmat(path+'tuckerae_270.mat')['xc'][0][1]
    hx24=sio.loadmat(path+'tuckerae_270.mat')['xo'][0][4]
    hx25=sio.loadmat(path+'tuckerae_270.mat')['xo'][0][5]
    
    hx31=sio.loadmat(path+'tuckerae_270.mat')['xo'][0][6]
    hx32=sio.loadmat(path+'tuckerae_270.mat')['xm'][0][1]
    hx33=sio.loadmat(path+'tuckerae_270.mat')['xi'][0][1]
    hx34=sio.loadmat(path+'tuckerae_270.mat')['xi'][0][2]
    hx35=sio.loadmat(path+'tuckerae_270.mat')['xo'][0][7]
    
    xjtuallh=[hx11,hx12,hx13,hx14,hx15,hx21,hx22,hx23,hx24,hx25,hx31,hx32,hx33,hx34,hx35]
    return xjtuallh
    
def get_value(xx):
    regress_labels=[]
    for i in range(len(xx)):
        regress_label=np.linspace(1,0,(len(xx[i])-8))
        regress_labels.append(regress_label)
    return regress_labels

def get_needtarget(target,time_steps):
    tg21=target[2][870-time_steps:911,:]
    tg22=target[0][742-time_steps:797,:]
    tg26=target[1][683-time_steps:701]
    
    xx=[torch.Tensor(tg22),torch.Tensor(tg26),torch.Tensor(tg21)]
    return xx