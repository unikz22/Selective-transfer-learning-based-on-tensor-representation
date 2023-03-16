# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 20:46:59 2021

@author: WJ
"""

import copy
import scipy.io as sio
import tensorly as tl
tl.set_backend('pytorch')
import scipy as sp
import numpy as np
import torch

from tensorly.decomposition import tucker

from util.MDT import MDTWrapper
from util.functions import svd_init

class BHTTUCKER3(object):
    def __init__(self,X1,X2, Rs, K, tol, seed=None, Us_mode=4, \
        verbose=0, convergence_loss=False):
        """store all parameters in the class and do checking on taus"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._trans_data1 = X1
        self._trans_data2 = X2
        

        # self._ts_ori1_shape = X1.shape
        # self._ts_ori2_shape = X2.shape
        
#        self._N = len(ts.shape) - 1
 #       self.T = ts.shape[-1]
        self._Rs = Rs
        self._K = K
        self._tol = tol
        self._Us_mode = Us_mode
        self._verbose = verbose
        self._convergence_loss = convergence_loss
        
        if seed is not None:
            torch.random.seed()
        
        # # check Rs parameters
        # M = 0
       
        # for dms,tau in zip(self._ts_ori1_shape, taus):
        #     if dms == tau:
        #         M += 1
        #     elif dms > tau:
        #         M += 2
                
        # if M-1 != len(Rs):
        #     raise ValueError("the first element of taus should be equal to the num of series")
        

    
    
    
    
    def _initilizer(self, T_hat, Js, Rs, Xs):
        # initilize Us
        U = [ torch.rand([j,r]).to(self.device) for j,r in zip( list(Js), Rs )]
        return U
    
    def _initilize_U(self, T_hat, Xs, Rs):

        haveNan = True
        while haveNan:
            factors = svd_init(Xs[0], range(len(Xs[0].shape)), ranks=Rs)
            haveNan = torch.any(torch.isnan(factors))
        return factors  
    
    def _get_Xs(self, trans_data):

        T_hat = trans_data.shape[-1]
        Xs = [ trans_data[..., t] for t in range(T_hat)]

        return Xs    
    def _get_cores(self, Xs, Us):
        cores = [ tl.tenalg.multi_mode_dot( x, [u.T for u in Us], modes=[i for i in range(len(Us))] ) for x in Xs]

        return cores
    
    def _get_H(self, Us, n):

        Hs = tl.tenalg.kronecker([u.T for u, i in zip(Us[::-1], reversed(range(len(Us)))) if i!= n ])
        return Hs   
    
    def _update_cores(self, n, Us, Xs, cores, lam=1):
        T_hat = len(Xs)
        unfold_cores = self._get_unfold_tensor(cores, n)
        # H = self._get_H(Us, n)
        # for t in range(0, T_hat):
        #     unfold_Xs = self._get_unfold_tensor(Xs[t].to(self.device), n)
        #     unfold_cores[t] = 1/(1+lam) * (lam * torch.matmul( torch.matmul(Us[n].T.to(self.device), unfold_Xs), H.T))
        return unfold_cores
    
     
    def _get_fold_tensor(self, tensor, mode, shape):
        if isinstance(tensor,list):
            return [ tl.base.fold(ten, mode, shape) for ten in tensor ]
        elif isinstance(tensor, torch.Tensor):
            return tl.base.fold(tensor, mode, shape)
        else:
            raise TypeError(" 'tensor' need to be a list or numpy.ndarray")
            
    def _get_unfold_tensor(self, tensor, mode):
        
        if isinstance(tensor, list):
            return [ tl.base.unfold(ten, mode) for ten in tensor]
        elif isinstance(tensor, torch.Tensor):
            return tl.base.unfold(tensor, mode)
        else:
            raise TypeError(" 'tensor' need to be a list or numpy.ndarray")              
            
    def _update_Us(self, Us, Xs, unfold_cores, n):

        T_hat = len(Xs)
        M = len(Us)
#        begin_idx = self._p + self._q

        H = self._get_H(Us, n)
        # orth in J3
        if self._Us_mode == 1:
            if n<M-1:
                As = []
                Bs = []
                for t in range(0, T_hat):
                    unfold_X = self._get_unfold_tensor(Xs[t], n)
                    As.append(torch.dot(torch.dot(unfold_X, H.T), torch.dot(unfold_X, H.T).T))
                    Bs.append(torch.dot(torch.dot(unfold_X, H.T), unfold_cores[t].T))
                a = sp.linalg.pinv(torch.sum(As, axis=0))
                b = torch.sum(Bs, axis=0)
                temp = torch.dot(a, b)
                Us[n] = temp / torch.linalg.norm(temp)
            else:
                Bs = []
                for t in range(0, T_hat):
                    unfold_X = self._get_unfold_tensor(Xs[t], n)
                    Bs.append(torch.dot(torch.dot(unfold_X, H.T), unfold_cores[t].T))
                b = torch.sum(Bs, axis=0)
                U_, _, V_ = torch.linalg.svd(b, full_matrices=False)
                Us[n] = torch.dot(U_, V_)
        # orth in J1 J2
        elif self._Us_mode == 2:
            if n<M-1:
                Bs = []
                for t in range(0, T_hat):
                    unfold_X = self._get_unfold_tensor(Xs[t], n)
                    Bs.append(torch.dot(torch.dot(unfold_X, H.T), unfold_cores[t].T))
                b = torch.sum(Bs, axis=0)
                U_, _, V_ = torch.linalg.svd(b, full_matrices=False)
                Us[n] = torch.dot(U_, V_)
            else:
                As = []
                Bs = []
                for t in range(0, T_hat):
                    unfold_X = self._get_unfold_tensor(Xs[t], n)
                    As.append(torch.dot(torch.dot(unfold_X, H.T), torch.dot(unfold_X, H.T).T))
                    Bs.append(torch.dot(torch.dot(unfold_X, H.T), unfold_cores[t].T))
                a = sp.linalg.pinv(torch.sum(As, axis=0))
                b = torch.sum(Bs, axis=0)
                temp = torch.dot(a, b)
                Us[n] = temp / torch.linalg.norm(temp)
        # no orth      
        elif self._Us_mode == 3:
            As = []
            Bs = []
            for t in range(0, T_hat):
                unfold_X = self._get_unfold_tensor(Xs[t], n)
                As.append(torch.dot(torch.dot(unfold_X, H.T), torch.dot(unfold_X, H.T).T))
                Bs.append(torch.dot(torch.dot(unfold_X, H.T), unfold_cores[t].T))
            a = sp.linalg.pinv(torch.sum(As, axis=0))
            b = torch.sum(Bs, axis=0)
            temp = torch.dot(a, b)
            Us[n] = temp / torch.linalg.norm(temp)
        # all orth
        elif self._Us_mode == 4:
            Bs = []
            for t in range(0, T_hat):
                unfold_X = self._get_unfold_tensor(Xs[t], n)
                # Bs.append(torch.matmul(torch.matmul(unfold_X, H.T), unfold_cores[t].T).cpu().detach().numpy())
                Bs.append(torch.matmul(torch.matmul(unfold_X, H.T), unfold_cores[t].T))
            b = torch.sum(torch.cat([torch.unsqueeze(i,0) for i in Bs], dim=0), dim=0)
            # Bs = torch.tensor(Bs)
            # b = torch.sum(Bs, axis=0)
            
            #b = b.replace(torch.inf, torch.nan).replace(-torch.inf, torch.nan).dropna()
            U_, _, V_ = torch.svd(b)
            Us[n] = torch.matmul(U_, V_)
        # only orth in J1
        elif self._Us_mode == 5:
            if n==0:
                Bs = []
                for t in range(0, T_hat):
                    unfold_X = self._get_unfold_tensor(Xs[t], n)
                    Bs.append(torch.dot(torch.dot(unfold_X, H.T), unfold_cores[t].T))
                b = torch.sum(Bs, axis=0)
                U_, _, V_ = torch.linalg.svd(b, full_matrices=False)
                Us[n] = torch.dot(U_, V_)
            else:
                As = []
                Bs = []
                for t in range(0, T_hat):
                    unfold_X = self._get_unfold_tensor(Xs[t], n)
                    As.append(torch.dot(torch.dot(unfold_X, H.T), torch.dot(unfold_X, H.T).T))
                    Bs.append(torch.dot(torch.dot(unfold_X, H.T), unfold_cores[t].T))
                a = sp.linalg.pinv(torch.sum(As, axis=0))
                b = torch.sum(Bs, axis=0)
                temp = torch.dot(a, b)
                Us[n] = temp / torch.linalg.norm(temp)
        # only orth in J2
        elif self._Us_mode == 6:
            if n==1:
                Bs = []
                for t in range(0, T_hat):
                    unfold_X = self._get_unfold_tensor(Xs[t], n)
                    Bs.append(torch.dot(torch.dot(unfold_X, H.T), unfold_cores[t].T))
                b = torch.sum(Bs, axis=0)
                U_, _, V_ = torch.linalg.svd(b, full_matrices=False)
                Us[n] = torch.dot(U_, V_)
            else:
                As = []
                Bs = []
                for t in range(0, T_hat):
                    unfold_X = self._get_unfold_tensor(Xs[t], n)
                    As.append(torch.dot(torch.dot(unfold_X, H.T), torch.dot(unfold_X, H.T).T))
                    Bs.append(torch.dot(torch.dot(unfold_X, H.T), unfold_cores[t].T))
                a = sp.linalg.pinv(torch.sum(As, axis=0))
                b = torch.sum(Bs, axis=0)
                temp = torch.dot(a, b)
                Us[n] = temp / torch.linalg.norm(temp)
        return Us 
    def _compute_convergence(self, new_U, old_U):
        
        new_old = [ n-o for n, o in zip(new_U, old_U)]
        aa = [torch.sqrt(tl.tenalg.inner(e,e)) for e in new_old]
        a_ =torch.cat([torch.unsqueeze(i,0) for i in aa],dim = 0)
        
        # a = torch.sum([torch.sqrt(tl.tenalg.inner(e,e)) for e in new_old], axis=0)
        a = torch.sum(a_, axis=0)
        
        bb = [torch.sqrt(tl.tenalg.inner(e,e)) for e in new_U]
        b_ =torch.cat([torch.unsqueeze(i,0) for i in bb],dim = 0)
        b = torch.sum(b_, axis=0)
        return a/b
        
    def _inverse_MDT(self, mdt, data, taus, shape):
        return mdt.inverse(data, taus, shape)       

    def _forward_MDT(self, data, taus):
        self.mdt = MDTWrapper(data,taus)
        trans_data = self.mdt.transform()
        self._T_hat = self.mdt.shape()[-1]
        return trans_data, self.mdt
    
    
    
    def run(self):
        """run the program

        Returns
        -------
        result : torch.ndarray, shape (num of items, num of time step +1)
            prediction result, included the original series

        loss : list[float] if self.convergence_loss == True else None
            Convergence loss

        """

        
        Us,coress, corest,loss    = self._run()
          
        if self._convergence_loss:
            
            return Us,coress, corest,loss      
        
        return  Us,coress, corest,None
    
    def _run(self):
        # trans_data1, mdt1 = self._forward_MDT(self._ts1, self._taus)
        # trans_data2, mdt2 = self._forward_MDT(self._ts2, self._taus)
        
        trans_data1=self._trans_data1
        trans_data2=self._trans_data2

        X_source=torch.cat([trans_data1],axis=2) # 在dim=2 处拼接
        X_target=torch.cat([trans_data2],axis=2) # 在dim=2 处拼接
        trans_data=torch.cat([X_source,X_target],axis=2) # 在dim=2 处拼接   

#将三维数组变成列表形式        
        Xs = self._get_Xs(trans_data)
        Xss = self._get_Xs(X_source)
        Xst = self._get_Xs(X_target)

        # initialize Us
        Us = self._initilizer(len(Xs), Xs[0].shape, self._Rs, Xs)
        
        con_loss = []
        for k in range(self._K):

            old_Us = Us.copy()            
            # get cores
            cores = self._get_cores(Xs, Us)
            for n in range(len(self._Rs)):
                
                cores_shape = cores[0].shape
                unfold_cores = self._update_cores(n, Us, Xs, cores, lam=1)
                cores = self._get_fold_tensor(unfold_cores, n, cores_shape)
                
                # update Us 
                Us = self._update_Us(Us, Xs, unfold_cores, n)                            

            # convergence check:
            convergence = self._compute_convergence(Us, old_Us)
            con_loss.append(convergence)
            
            if k%10 == 0:
                if self._verbose == 1:             
                    print("iter: {}, convergence: {}, tol: {:.10f}".format(k, convergence, self._tol))
                    #print("alpha: {}, beta: {}".format(alpha, beta))

            if self._tol > convergence:
                if self._verbose == 1: 
                    print("iter: {}, convergence: {}, tol: {:.10f}".format(k, convergence, self._tol))
                break

        
#得到更新后的源域和目标域的核张量             
        coress = self._get_cores(Xss, Us)
        corest = self._get_cores(Xst, Us)
       
        return Us,coress, corest,con_loss 