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
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.models.deep_gps import DeepGP
from model.deep_gp import DeepGPLayer
from gpytorch.mlls import DeepApproximateMLL
from gpytorch.mlls import VariationalELBO, AddedLossTerm
import gpytorch
from model.layers import NonLinearLayer, DynamicLinearLayer
from dynonet.lti import MimoLinearDynamicalOperator
from dynonet.static import MimoStaticNonLinearity

class DeepGP_custom(DeepGP):
    def __init__(self,  number_in,number_out, time,deltat,batch_size=300,
                 num_states_dynlayer=[3,3],
                 num_inducing_dynlayer=[500,200],
                 number_in_l_dynlayer=[1,1],
                 parametrisation_dynlayer=['jordan','jordan'],
                 num_inducing_nonlin=[150],
                 strategy_inducing_init_dynamic="multinomial",
                 initialization_type="normal",
                 number_outputs_layers1=[3],
                 number_outputs_layers2=[3]):

        #Linear LTI layer
        lti_layer1 = DynamicLinearLayer(
            input_dims=number_in,
            output_dims=number_outputs_layers1[0], #number of outputs
            time=time,
            deltat=deltat,
            num_states=num_states_dynlayer[0],
            number_in_l=number_in_l_dynlayer[0],
            num_inducing=np.minimum(len(time),num_inducing_dynlayer[0]),# number of inducing points
            batch_size=batch_size,
            parametrisation=parametrisation_dynlayer[0],
            strategy_inducing_init = strategy_inducing_init_dynamic,
            include_D=True,
            initialization_type=initialization_type
        )
        
        

        #Nonlinear layer
        nonlin_layer = NonLinearLayer(
            input_dims=lti_layer1.output_dims,
            output_dims=number_outputs_layers2[0], #number_out, #number_out, #None, if last layer
            mean_type='constant',
            num_inducing=num_inducing_nonlin[0], #number of inducing points
        )

                
        lti_layer2 = DynamicLinearLayer(
            input_dims=nonlin_layer.output_dims,
            output_dims=number_out,
            time=time,
            deltat=deltat,
            num_states=num_states_dynlayer[1],
            number_in_l=number_in_l_dynlayer[1],
            num_inducing=np.minimum(len(time),num_inducing_dynlayer[1]),# number of inducing points
            batch_size=batch_size,
            parametrisation=parametrisation_dynlayer[1],
            include_D=True,
            strategy_inducing_init = strategy_inducing_init_dynamic,
            initialization_type=initialization_type
        )
          
        super().__init__()
        

        self.lti_layer1 = lti_layer1
        self.lti_layer2 = lti_layer2
        self.nonlin_layer = nonlin_layer
        self.likelihood = GaussianLikelihood()
        #this is used for training, to save the previous-batch-mean
        self.batch_restart = True
        self.batch_size=batch_size

        
    
    def forward(self, inputs):
        
        #this is the DEEP model
        
        #the first column is time
        time = inputs[...,[0]]
        
        #1st layer - LTI layer        
        self.lti_layer1.batch_restart=self.batch_restart #only for the linear layer
        out1 = self.lti_layer1(time,inputs[...,1:])  
        
        #2ns layer - nonlinear layer
        out2 = self.nonlin_layer(out1)    
        
        #3rd layer - LTI layer
        self.lti_layer2.batch_restart=self.batch_restart #only for the linear layer
        self.lti_layer2.is_last_layer=True #so the model knows this is the last layer
        out3 = self.lti_layer2(time,out2)
        
        # Output
        output = out3 
                                                          
        return output
    
    def predict(self, x_batch,*args):
        if self.training==False:
            self.lti_layer1.training=False 
            self.lti_layer2.training=False  
        else:
            raise ValueError
            
        with torch.no_grad():
            mus = []
            variances = []
            preds = self.likelihood(self(x_batch,*args))
            mus.append(preds.mean)
            variances.append(preds.variance)

        
        return torch.cat(mus, dim=-1), torch.cat(variances, dim=-1)

class NN_custom(nn.Module):
    def __init__(self, in_channels, out_channels, n_a,n_b,n_hidden):
       super().__init__()
       
       innlay=3
       midlay=3
  

       # PyTorch class that holds layers in a list
       self.G1 = MimoLinearDynamicalOperator(in_channels = in_channels, out_channels = innlay, n_b = n_b[0], n_a = n_a[0], n_k=1)
       self.F1 = MimoStaticNonLinearity(innlay, midlay, n_hidden=n_hidden[0]) 
       self.G2 = MimoLinearDynamicalOperator(in_channels = midlay, out_channels = out_channels, n_b = n_b[1] , n_a = n_a[1] , n_k=0)
       
               
    def forward(self, x):
       
       x = self.G1(x)
       x = self.F1(x)
       y = self.G2(x)
       
       return y
   
    
torch.manual_seed(123)
device = torch.device("cuda")

architecture =  'jordan'

# load data
data = pd.read_csv("../data/vic_elec.csv",index_col=0)
#data = data.astype('float32')
deltat = 1/(365*48)#halh-hour

    


#scale time
#for i in range(2,6):
#    data.iloc[:,i] = data.iloc[:,i]
#    data.iloc[:,i] = (data.iloc[:,i])*deltat
    
    
#data = data.iloc[0:2500,:]
#plt.plot(data.iloc[:,0])
#plt.plot(data.iloc[:,1])

#data
cutn=365*48*2
y = torch.tensor(data.iloc[:,0].values[0:cutn])
u =  torch.tensor(np.hstack([data.iloc[:,[1]].values[0:cutn],data.iloc[:,[3]].values[0:cutn]+0.0])) #,3,4,5
t = torch.linspace(0,len(y),len(y))[:,None]*deltat
#add covariates
m1 = 1/(48*7*deltat)
m2 = 1/(48*deltat)
u = torch.hstack([u,torch.cos(2*np.pi*t),torch.sin(2*np.pi*t)
                  ,torch.cos(4*np.pi*t),torch.sin(4*np.pi*t)
                  #,torch.cos(6*np.pi*t),torch.sin(6*np.pi*t)
                  #,torch.cos(8*np.pi*t),torch.sin(8*np.pi*t)
              ,torch.cos(2*np.pi*t*m2),torch.sin(2*np.pi*t*m2)
              ,torch.cos(4*np.pi*t*m2),torch.sin(4*np.pi*t*m2)
              ,torch.cos(6*np.pi*t*m2),torch.sin(6*np.pi*t*m2)
              ,torch.cos(8*np.pi*t*m2),torch.sin(8*np.pi*t*m2)
              ,torch.cos(2*np.pi*t*m1),torch.sin(2*np.pi*t*m1)
              ,torch.cos(4*np.pi*t*m1),torch.sin(4*np.pi*t*m1)
              ,torch.cos(6*np.pi*t*m1),torch.sin(6*np.pi*t*m1)
              ,torch.cos(8*np.pi*t*m1),torch.sin(8*np.pi*t*m1)])

u = u[None,:]
y = y[None,:]
# Loop to visualize the input-output relationships for the training data


N = u.shape[1]
N_train=N-48*182#int(0.8*N)



#t = torch.linspace(0,y.shape[1],y.shape[1])[:,None]*deltat

name_exp='ele'
path_exp ='../results/electricity/'
train.run(device,y,u,t,deltat,N_train,name_exp,path_exp,
          n_a=[10,10],n_hidden_NN=[32],
          num_inducing_dynlayer=[700,700],num_inducing_nonlin=[200],
          shiftp = 2000,
          itertr_NN=3000,itertr_GP=500,typeD="custom",batch_size=1053,strategy_inducing_init_dynamic="multinomial",
          learning_rate_GP=0.002,
          initialization_type="normal",
          scale_u="all", scale_y="all",num_samples_likelihood_tr=10,
          NN_custom=NN_custom,GP_custom=DeepGP_custom) 
