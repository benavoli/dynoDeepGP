#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 09:59:33 2024

@author: benavoli
"""

from model.layers import NonLinearLayer, DynamicLinearLayer
import numpy as np

from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.models.deep_gps import DeepGP
from model.deep_gp import DeepGPLayer
from gpytorch.mlls import DeepApproximateMLL
from gpytorch.mlls import VariationalELBO, AddedLossTerm
import gpytorch
from gpytorch.distributions import MultivariateNormal
from torch.utils.data import TensorDataset, DataLoader
import torch
import tqdm
#num_hidden_dims = 1 #only one output (It is a MISO)


class DeepGP_LTI_only(DeepGP):
    def __init__(self,  number_in,number_out, time,deltat,batch_size=300,
                 num_states_dynlayer=[3],
                 num_inducing_dynlayer=[500],
                 number_in_l_dynlayer=[1],
                 parametrisation_dynlayer=['jordan'],
                 num_inducing_nonlin=[150],
                 strategy_inducing_init_dynamic="multinomial",
                 initialization_type="sampling_time_scaled"):

        #Linear LTI layer
        lti_layer = DynamicLinearLayer(
            input_dims=number_in,
            output_dims=number_out,
            time=time,
            deltat=deltat,
            num_states=num_states_dynlayer[0],
            number_in_l=number_in_l_dynlayer[0],
            num_inducing=np.minimum(len(time),num_inducing_dynlayer[0]),# number of inducing points
            batch_size=batch_size,
            parametrisation=parametrisation_dynlayer[0],
            strategy_inducing_init = strategy_inducing_init_dynamic,
            initialization_type=initialization_type
        )
        
        


          
        super().__init__()
        

        self.lti_layer = lti_layer
        self.likelihood = GaussianLikelihood()
        #this is used for training, to save the previous-batch-mean
        self.batch_restart = True
        self.batch_size=batch_size

        
    
    def forward(self, inputs):
        
        #this is the DEEP model
        
        #time
        time = inputs[...,[0]]
        
        #LTI layer
        self.lti_layer.batch_restart=self.batch_restart #only for the linear layer
        self.lti_layer.is_last_layer = True #so the model knows this is the last layer
        out1 = self.lti_layer(time,inputs[...,1:])  
           
        #Output
        output = out1
          

        return output
    
    def predict(self, x_batch,*args):
        if self.training==False:
            self.lti_layer.training=False  
        else:
            raise ValueError
            
        with torch.no_grad():
            mus = []
            variances = []
            preds = self.likelihood(self(x_batch,*args))
            mus.append(preds.mean)
            variances.append(preds.variance)

        
        return torch.cat(mus, dim=-1), torch.cat(variances, dim=-1)

class DeepGP_Wiener(DeepGP):
    def __init__(self,  number_in,number_out, time,deltat,batch_size=300,
                 num_states_dynlayer=[3],
                 num_inducing_dynlayer=[500],
                 number_in_l_dynlayer=[1],
                 parametrisation_dynlayer=['jordan'],
                 num_inducing_nonlin=[150],
                 strategy_inducing_init_dynamic="multinomial",
                 initialization_type="normal"): #"sampling_time_scaled"

        #Linear LTI layer
        lti_layer = DynamicLinearLayer(
            input_dims=number_in,
            output_dims=number_out,
            time=time,
            deltat=deltat,
            num_states=num_states_dynlayer[0],
            number_in_l=number_in_l_dynlayer[0],
            num_inducing=np.minimum(len(time),num_inducing_dynlayer[0]),# number of inducing points
            batch_size=batch_size,
            parametrisation=parametrisation_dynlayer[0],
            strategy_inducing_init = strategy_inducing_init_dynamic,
            initialization_type=initialization_type
        )
        
        

        #Nonlinear layer
        nonlin_layer = NonLinearLayer(
            input_dims=lti_layer.output_dims,
            output_dims=number_out, #number_out, #number_out, #None, if last layer
            mean_type='constant',
            num_inducing=num_inducing_nonlin[0], #number of inducing points
        )

          
        super().__init__()
        

        self.lti_layer = lti_layer
        self.nonlin_layer = nonlin_layer
        self.likelihood = GaussianLikelihood()
        #this is used for training, to save the previous-batch-mean
        self.batch_restart = True
        self.batch_size=batch_size

        
    
    def forward(self, inputs):
        
        #this is the DEEP model
        
        #time
        time = inputs[...,[0]]
        
        #LTI layer
        self.lti_layer.batch_restart=self.batch_restart #only for the linear layer
        out1 = self.lti_layer(time,inputs[...,1:])  
        
        #Nonlinear layer
        self.nonlin_layer.is_last_layer = True #so the model knows this is the last layer
        out2 = self.nonlin_layer(out1)    
        
        #Output
        output = out2    

        return output
    
    def predict(self, x_batch,*args):
        if self.training==False:
            self.lti_layer.training=False  
        else:
            raise ValueError
            
        with torch.no_grad():
            mus = []
            variances = []
            preds = self.likelihood(self(x_batch,*args))
            mus.append(preds.mean)
            variances.append(preds.variance)

        
        return torch.cat(mus, dim=-1), torch.cat(variances, dim=-1)



class DeepGP_Wiener_Hammerstein(DeepGP):
    def __init__(self,  number_in,number_out, time,deltat,batch_size=300,
                 num_states_dynlayer=[3,3],
                 num_inducing_dynlayer=[500,200],
                 number_in_l_dynlayer=[1,1],
                 parametrisation_dynlayer=['jordan','jordan'],
                 num_inducing_nonlin=[150],
                 strategy_inducing_init_dynamic="multinomial",
                 initialization_type="normal"):

        #Linear LTI layer
        lti_layer1 = DynamicLinearLayer(
            input_dims=number_in,
            output_dims=number_in,
            time=time,
            deltat=deltat,
            num_states=num_states_dynlayer[0],
            number_in_l=number_in_l_dynlayer[0],
            num_inducing=np.minimum(len(time),num_inducing_dynlayer[0]),# number of inducing points
            batch_size=batch_size,
            parametrisation=parametrisation_dynlayer[0],
            strategy_inducing_init = strategy_inducing_init_dynamic,
            initialization_type=initialization_type
        )
        
        

        #Nonlinear layer
        nonlin_layer = NonLinearLayer(
            input_dims=lti_layer1.output_dims,
            output_dims=number_out, #number_out, #number_out, #None, if last layer
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


class DeepGP_Hammerstein_Wiener(DeepGP):
    def __init__(self,  number_in,number_out, time,deltat,batch_size=300,
                 num_states_dynlayer=[3],
                 num_inducing_dynlayer=[500],
                 number_in_l_dynlayer=[1],
                 parametrisation_dynlayer=['jordan'],
                 num_inducing_nonlin=[150,150],
                 strategy_inducing_init_dynamic="multinomial",
                 initialization_type="normal"):
        
        #Nonlinear layer
        nonlin_layer1 = NonLinearLayer(
            input_dims=number_in,
            output_dims=number_in,
            mean_type='linear',
            num_inducing=num_inducing_nonlin[0], #number of inducing points
        )

        #Linear LTI layer
        lti_layer= DynamicLinearLayer(
            input_dims=nonlin_layer1.output_dims,
            output_dims=number_out, #
            time=time,
            deltat=deltat,
            num_states=num_states_dynlayer[0],
            number_in_l=number_in_l_dynlayer[0],
            num_inducing=np.minimum(len(time),num_inducing_dynlayer[0]),# number of inducing points
            batch_size=batch_size,
            parametrisation=parametrisation_dynlayer[0],
            strategy_inducing_init = strategy_inducing_init_dynamic,
            initialization_type=initialization_type
        )
        
        

        #Nonlinear layer
        nonlin_layer2 = NonLinearLayer(
            input_dims=lti_layer.output_dims,
            output_dims=number_out,
            mean_type='constant',
            num_inducing=num_inducing_nonlin[1], #number of inducing points
        )

                

          
        super().__init__()
        

        self.lti_layer = lti_layer
        self.nonlin_layer1 = nonlin_layer1
        self.nonlin_layer2 = nonlin_layer2
        self.likelihood = GaussianLikelihood()
        #this is used for training, to save the previous-batch-mean
        self.batch_restart = True
        self.batch_size=batch_size

        
    
    def forward(self, inputs):
        
        #this is the DEEP model
        
        #time
        time = inputs[...,[0]]
        
        #Input nonlinear layer
        out1 = self.nonlin_layer1(inputs[...,1:]) 
        
        # LTI layer
        self.lti_layer.batch_restart=self.batch_restart #only for the linear layer
        out2 = self.lti_layer(time,out1)  
        
        #Output nonlinear layer
        self.nonlin_layer2.is_last_layer = True #the deep model needs to know this is the last layer
        out3 = self.nonlin_layer2(out2)    
        

        
        output = out3 
        

        return output
    
    def predict(self, x_batch,*args):
        if self.training==False:
            self.lti_layer.training=False  
        else:
            raise ValueError
            
        with torch.no_grad():
            mus = []
            variances = []
            preds = self.likelihood(self(x_batch,*args))
            mus.append(preds.mean)
            variances.append(preds.variance)

        
        return torch.cat(mus, dim=-1), torch.cat(variances, dim=-1)
    

class DeepGP_nonLin2(DeepGP):
    def __init__(self,  number_in,number_out, time,deltat,batch_size=300,
                 num_states_dynlayer=[],
                 num_inducing_dynlayer=[],
                 number_in_l_dynlayer=[],
                 parametrisation_dynlayer=[],
                 num_inducing_nonlin=[150,150]):
        
        #Nonlinear layer
        nonlin_layer1 = NonLinearLayer(
            input_dims=number_in,
            output_dims=number_in, 
            mean_type='linear',
            num_inducing=num_inducing_nonlin[0], #number of inducing points
        )

        

        #Nonlinear layer
        nonlin_layer2 = NonLinearLayer(
            input_dims=nonlin_layer1.output_dims,
            output_dims=number_out,#
            mean_type='linear',
            num_inducing=num_inducing_nonlin[1], #number of inducing points
        )

                

          
        super().__init__()
        


        self.nonlin_layer1 = nonlin_layer1
        self.nonlin_layer2 = nonlin_layer2
        self.likelihood = GaussianLikelihood()
        #this is used for training, to save the previous-batch-mean
        self.batch_restart = True
        self.batch_size=batch_size

        
    
    def forward(self, inputs):
        
        #this is the DEEP model
        
        #time
        time = inputs[...,[0]]
        
        #Input nonlinear layer
        out1 = self.nonlin_layer1(inputs[...,1:]) 
        

        
        #Output nonlinear layer
        self.nonlin_layer2.is_last_layer = True #the deep model needs to know this is the last layer
        out2 = self.nonlin_layer2(out1)    
        

        
        output = out2
        

        return output
    
    def predict(self, x_batch,*args):
        if self.training==False:
            1
            #self.lti_layer.training=False  
        else:
            raise ValueError
            
        with torch.no_grad():
            mus = []
            variances = []
            preds = self.likelihood(self(x_batch,*args))
            mus.append(preds.mean)
            variances.append(preds.variance)

        
        return torch.cat(mus, dim=-1), torch.cat(variances, dim=-1)
    