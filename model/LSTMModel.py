# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


class LSTMmodel(nn.Module,):
    def __init__(self,input_size,hidden_size,output_size,num_layers,time_steps):
        super(LSTMmodel, self).__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.output_size=output_size
        self.num_layers=num_layers 
        self.time_steps=time_steps
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=False)
        
        linear_list = [100,64,25]
        linear_seq = []
        for i in range(0, len(linear_list)-1):
            linear_seq = linear_seq + self.linear_block(
                linear_list[i], linear_list[i+1])
        self.linear_Layer = nn.Sequential(*linear_seq)
        self.linear_out = nn.Linear(linear_list[-1], self.output_size)
        
        # self.linear = nn.Linear(self.hidden_size, self.output_size)
        
    def linear_block(self, in_features, out_features):
        block  =  [nn.Linear(in_features, out_features)]
        block +=  [nn.BatchNorm1d(out_features)]
        block +=  [nn.ReLU()]
        return block
    
    def forward(self, X, future=0):
        
        out0, (h0, c0) = self.lstm(X)
        outh = self.linear_Layer(out0[:,-1,:])
        # denseh=torch.cat([outh,hhx],1)
        output=self.linear_out(outh)

        return output,outh,out0[:,-1,:]