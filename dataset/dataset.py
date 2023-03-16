# -*- coding: utf-8 -*-


from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np

class MyDataset(Dataset):
    """
        下载数据、初始化数据，都可以在这里完成
    """
 
    def __init__(self,xo,xi,xc,xm,block_size=50,samples=30):
        self.data={'xo':xo,
                   'xi':xi,
                   'xc':xc,
                   'xm':xm}
        
        self.class_label=self.get_class_label(self.data)
        self.regress_label=self.get_regress_label(self.data)
        self.samples=samples
        self.block_size=block_size
        data_block_dict,class_label_block_dict,regress_label_block_dict=self.split_block()
        self.data_block_dict=data_block_dict
        self.class_label_block_dict=class_label_block_dict
        self.regress_label_block_dict=regress_label_block_dict
        self.all_data=self.split_data()
        
    
    
    def get_all(self):
        return self.data,self.class_label,self.regress_label
        
    def get_class_label(self,data):
        self.label_sign={'xo':0,
                         'xi':1,
                         'xc':2,
                         'xm':3,}
        class_label={}
        for key,values in data.items(): #遍历故障种类
            class_label[key]=[]
            for value in values:#遍历轴承
                label=np.ones((value.shape[0],1))*self.label_sign[key]
                class_label[key].append(label)
        return class_label
    
    def get_regress_label(self,data):
        regress_label={}
        for key,values in data.items(): #遍历故障种类
            regress_label[key]=[]
            for value in values:#遍历轴承
                label=np.linspace(1,0,(value.shape[0]-8))
                regress_label[key].append(label)
        return regress_label
    
    def __getitem__(self, index):
        if self.block_size==-1:
            return self.data,self.class_label,self.regress_label
        return self.all_data[index]
 
    def __len__(self):
        if self.block_size==-1:
            return 1
        return len(self.all_data)
    
    def split_block(self):
        data_block_dict={}
        class_label_block_dict={}
        regress_label_block_dict={}
        for key,values in self.data.items():
            print(key)
            datas=self.data[key]
            class_labels=self.class_label[key]
            regress_labels=self.regress_label[key]
            
            datas_block=[]  #记录每个轴承的分块
            class_labels_block=[]  
            regress_labels_block=[]
            for i in range(len(datas)): #遍历轴承
                data=datas[i]
                class_label=class_labels[i]
                regress_label=regress_labels[i]
                sample_len=int(data.shape[0]/self.block_size)
                data_block=[]
                class_label_block=[]
                regress_label_block=[]
                for j in range(sample_len):
                    data_block.append(data[j*self.block_size:(j+1)*self.block_size,:])
                    class_label_block.append(class_label[j*self.block_size:(j+1)*self.block_size,:])
                    regress_label_block.append(regress_label[j*self.block_size:(j+1)*self.block_size])
                
            
                data_block.append(data[-self.block_size:,:])
                class_label_block.append(class_label[-self.block_size:,:])
                regress_label_block.append(regress_label[-self.block_size:])
                
                datas_block.append(data_block)
                class_labels_block.append(class_label_block)
                regress_labels_block.append(regress_label_block)
            
            data_block_dict[key]=datas_block
            class_label_block_dict[key]=class_labels_block
            regress_label_block_dict[key]=regress_labels_block
        return data_block_dict,class_label_block_dict,regress_label_block_dict

    def split_data(self):
        all_data=[]
        
        for num in range(self.samples):
            data_block_sel_dict={}
            class_label_block_sel_dict={}
            regress_label_block_sel_dict={}
            for key,values in self.data_block_dict.items(): #遍历故障类型
                data_blocks=values
                class_label_blocks=self.class_label_block_dict[key]
                regress_label_blocks=self.regress_label_block_dict[key]
                
                data_block_sel=[]
                class_label_block_sel=[]
                regress_label_block_sel=[]
                for i in range(len(data_blocks)): #遍历轴承
                    data_block=data_blocks[i]
                    class_label_block=class_label_blocks[i]
                    regress_label_block=regress_label_blocks[i]
                    
                    data_block_sel.append(data_block[np.random.randint(len(data_block))])
                    class_label_block_sel.append(class_label_block[np.random.randint(len(data_block))])
                    regress_label_block_sel.append(regress_label_block[np.random.randint(len(data_block))])
                
                data_block_sel_dict[key]=data_block_sel
                class_label_block_sel_dict[key]=class_label_block_sel
                regress_label_block_sel_dict[key]=regress_label_block_sel
            all_data.append([data_block_sel_dict,class_label_block_sel_dict,regress_label_block_sel_dict])
        return all_data