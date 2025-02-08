#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
#torch.set_default_device('cpu')
torch.set_default_dtype(torch.float64)
import sys
sys.path.append("../")
import torch.nn as nn
from simulator import wiener
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import gc
import pickle
# In[2]:


import scoringrules as sr
import nonlinear_benchmarks # used just for metrics
import pandas as pd

import train

torch.manual_seed(123)
device = torch.device("cuda")

architecture =  'jordan'

f_train_ds =  "../data/WH_EstimationExample.csv" #os.path.join(folder, 'F16Data_SineSw_Level4_Validation.csv')

f_test_ds = "../data/WH_TestDataset.csv" # "WH_TestDataset.csv"  # "WH_EstimationExample.csv" # "WH_SineSweepInput_meas.csv" #"WH_SineInput_meas.csv" # "WH_MultisineFadeOut.csv" # "WH_TestDataset.csv" #os.path.join(folder, 'F16Data_SineSw_Level4_Validation.csv')



dict_ds = {'train': [], 'test': [],}
dict_ds['train'] = pd.read_csv(f_train_ds)#.iloc[0:3000,:]
dict_ds['test'] = pd.read_csv(f_test_ds)#.iloc[0:3000,:]


#if f_test_ds == "WH_EstimationExample.csv":
    #dict_ds['test'].rename(columns={'u': 'u0', 'y': 'y0'}, inplace=True)
    #dict_ds['test'].rename(columns={'Unnamed: 13': 'u', 'Unnamed: 23': 'y'}, inplace=True)

u = torch.tensor(dict_ds['train']['u'].values[:,None][None,:])  #(dict_ds['train']['u'] - dict_ds['train']['u'].mean())/ dict_ds['train']['u'].std()
#u = torch.tensor(u.values, dtype = torch.float32).view(-1,1)
y = torch.tensor(dict_ds['train']['y'].values)[None,:]

# In[5]:
N = u.shape[1]  # total number of data points
N_train = 6500  #
deltat = 1 / 100.0 # dict_ds['train']['fs'].iloc[0]  # sampling rate

as_test=True
if as_test==True: 
    
    u = torch.cat([u,torch.tensor(dict_ds['test']['u'].values[:,None][None,:])],axis=1)
    y = torch.cat([y,torch.tensor(dict_ds['test']['y'].values)],axis=0)
    N=u.shape[1]
    N_train=len(dict_ds['train']['y'].values)



t = torch.linspace(0,y.shape[1],y.shape[1])[:,None]*deltat

name_exp='wiener_hamemrstein'
path_exp ='../results/WinerHammersteinReal/'
train.run(device,y,u,t,deltat,N_train,name_exp,path_exp,
          n_a=[10,10],n_hidden_NN=[32],
          num_inducing_dynlayer=[1000,1000],num_inducing_nonlin=[300],
          shiftp = 0,
          itertr_NN=1,itertr_GP=1200,typeD="WH",batch_size=1024,strategy_inducing_init_dynamic="multinomial",
          learning_rate_GP=0.002,
          initialization_type="normal") #WH
