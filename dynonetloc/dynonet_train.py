#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 09:49:54 2024

@author: benavoli
"""
# !pip install dynonet
from dynonet.lti import MimoLinearDynamicalOperator
from dynonet.static import MimoStaticNonLinearity
import torch
import torch.nn as nn

# a basic dynoNet with Wiener structure
class Wiener(nn.Module):
    def __init__(self,in_channels, out_channels,n_a,n_b,n_hidden):
        super().__init__()

        self.G = MimoLinearDynamicalOperator(in_channels=in_channels, out_channels=out_channels, n_b=n_b[0], n_a=n_a[0], n_k=1)
        self.F = MimoStaticNonLinearity(in_channels=out_channels, out_channels=out_channels, n_hidden=n_hidden[0])

    def forward(self, u):
        x = self.G(u)
        y = self.F(x)
        return y
    

class LTI(nn.Module):
    def __init__(self,in_channels, out_channels,n_a,n_b,n_hidden):
        super().__init__()

        self.G = MimoLinearDynamicalOperator(in_channels=in_channels, out_channels=out_channels, n_b=n_b[0], n_a=n_a[0], n_k=1)
        #self.F = MimoStaticNonLinearity(in_channels=out_channels, out_channels=out_channels, n_hidden=n_hidden[0])

    def forward(self, u):
        y = self.G(u)

        return y
    
class WH(nn.Module):
    def __init__(self, in_channels, out_channels, n_a,n_b,n_hidden):
       super().__init__()
       
  

       # PyTorch class that holds layers in a list
       self.G1 = MimoLinearDynamicalOperator(in_channels = in_channels, out_channels = in_channels, n_b = n_b[0], n_a = n_a[0], n_k=1)
       self.F1 = MimoStaticNonLinearity(in_channels, out_channels, n_hidden=n_hidden[0]) 
       self.G2 = MimoLinearDynamicalOperator(in_channels = out_channels, out_channels = out_channels, n_b = n_b[1] , n_a = n_a[1] , n_k=0)
       
               
    def forward(self, x):
       
       x = self.G1(x)
       x = self.F1(x)
       y = self.G2(x)
       
       return y

class HW(nn.Module):
    def __init__(self, in_channels, out_channels, n_a,n_b,n_hidden):
       super().__init__()
       


       # PyTorch class that holds layers in a list
       self.F1 = MimoStaticNonLinearity(in_channels, in_channels, n_hidden=n_hidden[0]) 
       self.G1 = MimoLinearDynamicalOperator(in_channels = in_channels, out_channels = out_channels, n_b = n_b[0], n_a = n_a[0], n_k=1)
       self.F2 = MimoStaticNonLinearity(out_channels, out_channels, n_hidden=n_hidden[1]) 
 
    def forward(self, x):
       
       x = self.F1(x)
       x = self.G1(x)
       y = self.F2(x)
       
       return y
   
def train_dynonet(in_channels, out_channels, n_a,n_b, n_hidden,train_y,test_y,train_u,test_u,itertr=1,typeD="Wiener",DeepNNmodel=[]):
    #train_u = u[:, :N_train, :]
    if len(train_u.shape)==1:
        test_y = test_y.unsqueeze(0).unsqueeze(-1)
        train_y = train_y.unsqueeze(0).unsqueeze(-1)
    else:
        train_y = train_y[...,None]
        test_y = test_y[...,None]


    if typeD=="Wiener":
        model = Wiener(in_channels, out_channels,n_a,n_b,n_hidden)
    elif typeD=="WH":
        model = WH(in_channels, out_channels,n_a,n_b,n_hidden)
    elif typeD=="HW":
        model = HW(in_channels, out_channels,n_a,n_b,n_hidden)
    elif typeD=="LTI":
        model = LTI(in_channels, out_channels,n_a,n_b,n_hidden)
    elif typeD=="custom":
        model = DeepNNmodel(in_channels, out_channels,n_a,n_b,n_hidden)
        
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


    
    LOSS = []
    for idx in range(itertr):
        for i in range(train_u.shape[0]):
            optimizer.zero_grad()
            train_y_hat = model(train_u[[i]])
            loss = torch.nn.functional.mse_loss(train_y[[i]], train_y_hat)
            LOSS.append(loss.item())
            loss.backward()
            #pbar.set_postfix_str(loss.item())
            optimizer.step()
    
    
   
    
    #test_u = u[:, :, :]
    
    #test_y.shape, test_y.shape
    test_y_hat=[]
    for i in range(train_u.shape[0]):
        test_y_hat.append(model(test_u[[i]]).detach())
    
    test_y_hat = torch.cat(test_y_hat,axis=0)
    return test_y_hat.numpy()