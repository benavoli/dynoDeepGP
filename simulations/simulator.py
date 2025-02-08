#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 16:35:45 2024

@author: benavoli
"""
import torch 
import numpy as np
from model.commonsLTI import build_A, calculate_Pinf, trajectoryLTI

def wiener(deltat,N,N_train,num_states=3,number_in_b=1,number_in_l=1,jitter=1e-8,nonlinear=torch.exp,deterministic=False,add_trend=False):
    batch_shape = 1
    t = torch.tensor([(deltat*np.arange(N)).astype(np.float64)]).T
    f_nyq = torch.tensor(0.5/deltat)
    #init1 = -f_nyq+torch.linspace(0,f_nyq/1.01,num_states)
    a_d = -1.0+torch.rand(batch_shape, num_states,1)*0.99
    a_d = f_nyq*a_d
    #shape = (batch_shape, num_states)
    #a_d = init1.expand(*shape)
    
    #a_d = -abs(torch.randn(batch_shape, num_states, 1))-0.7 #-f_nyq/10+torch.rand(batch_shape, num_states, 1)*f_nyq/20
    #a_d = -f_nyq+torch.linspace(0,f_nyq/1.01,pairs)
    #a_d = -torch.exp(torch.randn(batch_shape, num_states, 1))
    a_v = torch.rand(batch_shape, num_states, 1)*f_nyq
    #a_w = torch.rand(batch_shape, num_states, 1)
    C = torch.randn(batch_shape,1, num_states)
    B = torch.randn(batch_shape,num_states, number_in_b)*f_nyq
    if deterministic==True:
        L = torch.zeros(batch_shape,num_states, number_in_l)
    else:
        L = 0.1*torch.randn(batch_shape,num_states, number_in_l)*torch.sqrt(f_nyq)
   



    A = build_A(a_d,a_v,a_v)
    Pinf = calculate_Pinf(A,L,jitter)
    Ad = torch.matrix_exp(A*deltat)
    I = torch.stack([torch.eye(A.shape[1]) for i in range(A.shape[0])])
    Bd = torch.matmul(torch.linalg.inv(A), (Ad - I)).matmul(B)
    Cd = C

    Qd=Pinf-Ad@Pinf@Ad.transpose(1,2)#+I*jitter

    def ufun(t):
        c=torch.tensor(0.0)
        if add_trend==True:
            c=t
        u=[c+torch.sin(t* ((3+i*2) * np.pi)) for i in range(number_in_b)]
        return torch.stack(u).transpose(1,0)

    u,y=trajectoryLTI(N,N_train,t,ufun,Ad,Bd,Cd,Qd)
    paramters=[a_d,a_v,B,C,L]
    return t,u,nonlinear(y),paramters