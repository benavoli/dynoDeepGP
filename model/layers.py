#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 18:00:40 2024

@author: benavoli
"""
import torch
import gpytorch
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.distributions import MultivariateNormal
from gpytorch.models import ApproximateGP, GP
from gpytorch.mlls import VariationalELBO, AddedLossTerm
from gpytorch.likelihoods import GaussianLikelihood
from model.utility import select_elements
import numpy as np
from botorch.models.kernels.infinite_width_bnn import InfiniteWidthBNNKernel


from model.deep_gp import DeepGPLayer
from model.kernel_lowrank import LTIlowrank, LTIlowrankMean, set_right_to_left_params_lowrank
from model.kernel_jordan import LTIJordanCC, LTIJordanCCMean, set_right_to_left_params_jordan

from model.commonsLTI import zoh


import torch



class NonLinearLayer(DeepGPLayer):
    def __init__(self, input_dims, output_dims, num_inducing=128, mean_type='linear', base_nonlinearity=torch.tanh,is_last_layer=False):
        self.base_nonlinearity = base_nonlinearity

        if output_dims is None:
            inducing_points = 0.3*torch.randn(num_inducing, input_dims)
            batch_shape = torch.Size([])
        else:
            inducing_points = 0.3*torch.randn(output_dims, num_inducing, input_dims)
            batch_shape = torch.Size([output_dims])

        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=batch_shape
        )

        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )

        super(NonLinearLayer, self).__init__(variational_strategy, input_dims, output_dims, is_last_layer)

        if mean_type == 'constant':
            self.mean_module = ConstantMean(batch_shape=batch_shape)
        else:
            self.mean_module = LinearMean(input_dims,batch_shape=batch_shape)
        '''  
        self.covar_module = ScaleKernel(
            InfiniteWidthBNNKernel(2,batch_shape=batch_shape), #gpytorch.kernels.PiecewisePolynomialKernel(batch_shape=batch_shape, ard_num_dims=input_dims)
           batch_shape=batch_shape, ard_num_dims=None
        )
        '''
        self.covar_module =       ScaleKernel(gpytorch.kernels.MaternKernel(1.5,
                      #angle_prior=gpytorch.priors.GammaPrior(0.5,1),
                      #radius_prior=gpytorch.priors.GammaPrior(3,2),
                      batch_shape=batch_shape,
                      ard_num_dims=input_dims),batch_shape=batch_shape, ard_num_dims=None
                   )
        
        
        '''
        self.covar_module = gpytorch.kernels.LinearKernel(batch_shape=batch_shape, ard_num_dims=input_dims,
                                                          variance_constraint=gpytorch.constraints.Interval(1, 1.001),
                                                          offset_constraint=gpytorch.constraints.Interval(0, 0.001))
        '''

    def forward(self, x0):
        x = x0 # self.base_nonlinearity(x0)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def __call__(self, x, *other_inputs, **kwargs):
        """
        Overriding __call__ isn't strictly necessary, but it lets us add concatenation based skip connections
        easily. For example, hidden_layer2(hidden_layer1_outputs, inputs) will pass the concatenation of the first
        hidden layer's outputs and the input data to hidden_layer2.
        """
        if len(other_inputs):
            if isinstance(x, gpytorch.distributions.MultitaskMultivariateNormal):
                x = x.rsample()

            processed_inputs = [
                inp.unsqueeze(0).expand(gpytorch.settings.num_likelihood_sales.value(), *inp.shape)
                for inp in other_inputs
            ]

            x = torch.cat([x] + processed_inputs, dim=-1)

        return super().__call__(x, are_samples=bool(len(other_inputs)))
    
    
    
class DynamicLinearLayer(DeepGPLayer):
    def __init__(self, time,deltat, input_dims, output_dims,
                 num_inducing=128, num_states=3, number_in_l=1,
                 batch_size=300,parametrisation='jordan',is_last_layer=False,include_D=False,
                 strategy_inducing_init="multinomial",initialization_type="sampling_time_scaled"):
        #zoh
        self.time = time
        self.deltat = deltat
        number_in_b = input_dims
        input_dims = 1 #the only dimension for the kernel is time
        lent=torch.max(time.flatten())
        if strategy_inducing_init=="end":
            ind = torch.arange(len(time.flatten())-num_inducing+1,len(time.flatten()))
        elif strategy_inducing_init=="multinomial":
            ind = torch.multinomial(torch.log1p(time.flatten()/lent)/torch.log1p(torch.tensor(1)),num_inducing-1)
        self.time_inducing_points = torch.vstack([time[[0],:],time[torch.sort(ind)[0]]])

        # this is important - time must start from zero for convolution in the mean function
        assert self.time_inducing_points[0]==0.0
        
       
        self.output_dims = output_dims
        
        num_inducing = len(ind) +1 #self.time_inducing_points.flatten())
        
        if output_dims is None:
            self.time_inducing_points = self.time_inducing_points.expand(*self.time_inducing_points.shape)
            inducing_points =  torch.rand(num_inducing, input_dims)
            batch_shape = torch.Size([])
        else:
            self.time_inducing_points = self.time_inducing_points.expand(output_dims,*self.time_inducing_points.shape)
            inducing_points = torch.rand(output_dims, num_inducing, input_dims)
            batch_shape = torch.Size([output_dims])

        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=batch_shape
        )

        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=False,
            jitter_val=1e-6
        )

        super(DynamicLinearLayer, self).__init__(variational_strategy, input_dims, output_dims,is_last_layer)
        if include_D:
            D_constraint=None
        else: 
            D_constraint=gpytorch.constraints.Interval(0.0, 1e-6)
        if parametrisation=='lowrank':
            None  
            #self.covar_module = LTIlowrank(deltat,num_states=num_states, number_in_l=number_in_l, number_in_b=number_in_b, batch_shape=batch_shape,
            #                            ard_num_dims=1, active_dims=[0],d_constraint=D_constraint)
                                         ##l_constraint=gpytorch.constraints.Interval(0.0, 1e-6))
            #self.mean_module = LTIlowrankMean(deltat,add_steps_ahead=np.maximum(200,batch_size))
            #self.set_right_to_left_params=set_right_to_left_params_lowrank       
        elif parametrisation=='jordan':
            self.covar_module = LTIJordanCC(pairs=num_states,samplingtime=deltat, batch_shape=batch_shape,
                                            number_in_l=number_in_l,
                                            number_in_b=number_in_b,
                                            ard_num_dims=1, active_dims=[0],D_constraint=D_constraint,
                                            initialization_type=initialization_type)
                                            #L_re_constraint=gpytorch.constraints.Interval(0.0, 1e-6),
                                            #L_im_constraint=gpytorch.constraints.Interval(0.0, 1e-6))
            #copy covar paramters into  the mean
            self.mean_module = LTIJordanCCMean(samplingtime=deltat,num_inducing=num_inducing,add_steps_ahead=np.maximum(200,batch_size),
                                               batch_shape=batch_shape)
                            
            self.set_right_to_left_params=set_right_to_left_params_jordan
        self.set_right_to_left_params(self.mean_module,self.covar_module)
        self.training=True
        self.batch_restart=True
        self.batch_size=batch_size
       

    def forward(self, timex):
        n_dims = timex.ndimension()
        # Build the index: [0, ..., 0, :, :]
        indices = [0] * (n_dims - 2) + [slice(None), slice(None)]
        #concatenate time and u
        self.u_current = self.ufun(timex[indices]) #torch.stack([self.ufun[i] (timex[indices]) for i in range(len(self.ufun))]) #  self.ufun is defined in _call below
        #if len(timex.shape)>3:
        #timex=timex.expand(self.u_current.shape[0],*timex.shape)
        if self.u_current.ndimension()>2:
            ushape = self.u_current
            self.u_current=self.u_current.unsqueeze(1)
            self.u_current=self.u_current.expand(self.u_current.shape[0],timex.shape[-3],
                                  *self.u_current.shape[2:])
            
        else:
            self.u_current=self.u_current.expand(timex.shape[0],
                                  *self.u_current.shape[0:])
        
        
        xx = torch.cat([timex,self.u_current],axis=-1)
        #torch.cat([timex[0,...].expand(self.u_current.shape[0],timex.shape[1],1), self.u_current],dim=-1) 
        
        covar_x = self.covar_module(timex[indices])
        covar_x = covar_x.expand(*timex.shape[0:-3],*covar_x.shape[-3:])
        #copy covar paramters into  the mean
        self.set_right_to_left_params(self.mean_module,self.covar_module)
        
        
        #store or not previous-time values of the  mean function
        #if self.training==False:
        #    self.mean_module.add_steps_ahead=2000
        if self.batch_restart==True:
            self.mean_module.previous_state = torch.zeros((2,*self.u_current.shape[0:-2],self.mean_module.add_steps_ahead))
   
            #self.mean_module.previous_state[0,...]=self.mean_module.previous_state[1,...]
            
        mean_x= self.mean_module(xx)
        #if mean_x.shape[0]>1: #AAAAAAAA
        #    mean_x = mean_x[:,None,:]
        #if( mean_x.shape[0]>1)&(covar_x.shape[0]==1):
        #    covar_x = covar_x.repeat(mean_x.shape[0],1,1)
        return MultivariateNormal(mean_x, covar_x)
    

        
    def __call__(self, time, *other_inputs, **kwargs):
        """
        Overriding __call__ isn't strictly necessary, but it lets us add concatenation based skip connections
        easily. For example, hidden_layer2(hidden_layer1_outputs, inputs) will pass the concatenation of the first
        hidden layer's outputs and the input data to hidden_layer2.
        """
        

        timex = time

        if self.training==True:
            self.variational_strategy.inducing_points = self.time_inducing_points

        if len(other_inputs):
            if isinstance(other_inputs[0], gpytorch.distributions.MultitaskMultivariateNormal):
                u = other_inputs[0].rsample()
                are_samples=True
                if torch.any(torch.isnan(u)):                    
                    v=other_inputs[0]
                    v=v.add_jitter(1e-6)
                    u = v.rsample()
                if torch.any(torch.isnan(u)):                    
                    v=other_inputs[0]
                    v=v.add_jitter(1e-5)
                    u = v.rsample()
                if torch.any(torch.isnan(u)):                    
                    v=other_inputs[0]
                    v=v.add_jitter(1e-4)
                    u = v.rsample()
                if len(other_inputs)>1:
                    inp=other_inputs[1]
                    processed_inputs = [
                        inp.unsqueeze(0).expand(gpytorch.settings.num_likelihood_samples.value(), *inp.shape)
                    ]

                    u = torch.cat([u] + processed_inputs, dim=-1)
            else:
                
                u = other_inputs[0]
                
                #if len(u.shape)==2:
                #    u = u[None,:]
                are_samples=False
            ushape = u.shape
            if len(ushape)>2:
                timex=time.expand(*u.shape[0:-2],*time.shape)
        
        #if len(u.shape)==3:
        #    time = torch.cat([time.unsqueeze(0) for i in range(u.shape[0])],axis=0)
            
        # be sure time and u have same dimensions
        #timex=timex.repeat(self.u_current.shape[0],1,1
        if len(u.shape)>2:
            self.ufun = zoh(time,u) # [zoh(time,u[i]) for i in range(u.shape[0])]
        else:
            self.ufun = zoh(time,u) # [zoh(time,u)]
        

        return super().__call__(timex,  are_samples=are_samples)

