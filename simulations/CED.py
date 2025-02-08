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
                 number_outputs_layers1=[2],
                 number_outputs_layers2=[2]):

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
            #include_D=True,
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
       
       innlay=2
       midlay=2
  

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


# Load the training and testing datasets using the official dataloader
trains, tests = nonlinear_benchmarks.CED()

# initialize input and output sequences
u_train = np.zeros((len(trains), len(trains[0].u)))
y_train = np.zeros((len(trains), len(trains[0].y)))
u_test = np.zeros((len(tests), len(tests[0].u)))
y_test = np.zeros((len(tests), len(tests[0].y)))



# Unpack input and output data from the training and testing datasets
for i in range(len(trains)):
    u_train[i,:] = trains[i].u  # Accessing `u` and `y` attributes of `train`
    y_train[i,:] = trains[i].y  # Accessing `u` and `y` attributes of `train`

    u_test[i,:] = tests[i].u  # Accessing `u` and `y` attributes of `test`
    y_test[i,:] = tests[i].y  # Accessing `u` and `y` attributes of `test`

plt.plot(u_train[0])
plt.plot(u_train[1])
# Loop to visualize the input-output relationships for the training data
'''
for i in range(len(u_train)):
    print(f"Output: {i}")
    plt.figure()
    plt.subplot(2, 1, 1)  # Create a subplot for input data
    plt.plot(u_train[i,:], label="Input (u)")
    plt.legend()
    plt.subplot(2, 1, 2)  # Create a subplot for output data
    plt.plot(y_train[i,:], label="Output (y)")
    plt.legend()
    plt.show()  # Display the plot for each instance
'''


i = 1
N_train= len(u_train[i,:])

u_train = u_train[[i]]
u_test = u_test[[i]]
y_train = y_train[[i]]
y_test = y_test[[i]]
'''
u = np.concatenate([u_train[i,:],u_test[i,:]])
y = np.concatenate([y_train[i,:],y_test[i,:]])

N_train= len(u_train[i,:])
N = u.shape[1]
deltat = trains[0].sampling_time
u = torch.tensor(u[:,None][None,:])  #(dict_ds['train']['u'] - dict_ds['train']['u'].mean())/ dict_ds['train']['u'].std()
y = torch.tensor(y)
t = torch.linspace(0,len(y),len(y))[:,None]*deltat

'''
u = torch.tensor(np.concatenate([u_train[...,None],u_test[...,None]],axis=1))
y = torch.tensor(np.concatenate([y_train,y_test],axis=1))

N = u.shape[1]
deltat = trains[0].sampling_time



t = torch.linspace(0,y.shape[1],y.shape[1])[:,None]*deltat

name_exp='CED'+str(i)
path_exp ='../results/CED/'
train.run(device,y,u,t,deltat,N_train,name_exp,path_exp,
          n_a=[5,5],n_hidden_NN=[32],
          num_inducing_dynlayer=[250,250],num_inducing_nonlin=[175],
          shiftp = 400,
          itertr_NN=3000,itertr_GP=3000,typeD="custom",batch_size=400,strategy_inducing_init_dynamic="multinomial",
          learning_rate_GP=0.001,
          initialization_type="normal",
          scale_u=False, scale_y=False,num_samples_likelihood_tr=40,
          NN_custom=NN_custom,GP_custom=DeepGP_custom) 
