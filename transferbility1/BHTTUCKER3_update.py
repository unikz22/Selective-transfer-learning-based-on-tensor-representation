# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 20:46:59 2021

@author: WJ
"""
import tensorly as tl
tl.set_backend('pytorch')
import torch

class BHTTUCKER3_update(object):
    def __init__(self,X1,X2, Us):
        """store all parameters in the class and do checking on taus"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._trans_data1 = X1
        self._trans_data2 = X2
        self._Us = Us         
    
    def _get_Xs(self, trans_data):

        T_hat = trans_data.shape[-1]
        Xs = [ trans_data[..., t] for t in range(T_hat)]

        return Xs    
    def _get_cores(self, Xs, Us):
        cores = [ tl.tenalg.multi_mode_dot( x, [u.T for u in Us], modes=[i for i in range(len(Us))] ) for x in Xs]

        return cores
        
    
    def run(self):
        
        coress, corest    = self._run()
                  
        return  coress, corest
    
    def _run(self):
        # trans_data1, mdt1 = self._forward_MDT(self._ts1, self._taus)
        # trans_data2, mdt2 = self._forward_MDT(self._ts2, self._taus)
        
        trans_data1=self._trans_data1
        trans_data2=self._trans_data2

        X_source=torch.cat([trans_data1],axis=2) # 在dim=2 处拼接
        X_target=torch.cat([trans_data2],axis=2) # 在dim=2 处拼接

#将三维数组变成列表形式        
        Xss = self._get_Xs(X_source)
        Xst = self._get_Xs(X_target)

#得到更新后的源域和目标域的核张量             
        coress = self._get_cores(Xss, self._Us)
        corest = self._get_cores(Xst, self._Us)
       
        # coress = torch.cat([torch.unsqueeze(i,0) for i in coress], dim=0)
        # corest = torch.cat([torch.unsqueeze(i,0) for i in corest], dim=0)
        # X_S = coress.permute(1,2,0)
        # X_T = corest.permute(1,2,0)
        # X_S = tl.tenalg.multi_mode_dot(coress, self._Us)
        # X_T = tl.tenalg.multi_mode_dot(corest, self._Us)

      
        return coress, corest