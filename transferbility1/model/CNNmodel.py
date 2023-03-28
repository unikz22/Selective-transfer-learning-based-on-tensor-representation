#tucker-CNN-pre_model放在一个整体框架中
import os
# os.chdir('/home/htu/workspace/zw/work1/code5/dong')
import torch
import torch.nn as nn

#%%CNN提特征模型框架定义
class CNNmodel(nn.Module):
    def __init__(self):  
        super(CNNmodel, self).__init__()
        self.conv1 = nn.Sequential(
            nn.ReplicationPad2d(padding=(2, 2, 0, 0)),
            nn.Conv2d(1,20,(1,5),1,),
            nn.ReLU(),
            nn.BatchNorm2d(20),
            nn.MaxPool2d((1,4),4),
            )
        self.conv2 = nn.Sequential(
            nn.ReplicationPad2d(padding=(2, 2, 0, 0)),
            nn.Conv2d(20,50,(1,5),1,),
            nn.ReLU(),
            nn.BatchNorm2d(50),
            nn.MaxPool2d((1,4),4),
            )
        self.conv3 = nn.Sequential(
            nn.ReplicationPad2d(padding=(2, 2, 0, 0)),
            nn.Conv2d(50,64,(1,5),1,),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d((1,4),4),
            )
        self.L1 = nn.Sequential(nn.Linear(1*39*64,512),
                                nn.ReLU())
        self.trend = nn.Sequential(nn.Linear(512,128),
                                   nn.ReLU(),
                                   nn.Linear(128,1),
                                   )
        self.fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 50),
            nn.ReLU()
            )
        
        self.out = nn.Linear(50, 4)

    def forward(self, x):           
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0),-1)
        hx = self.L1(x)
        treHI = self.trend(hx)
        hfc = self.fc(hx)
        out = self.out(hfc)
        return hfc, out, treHI
