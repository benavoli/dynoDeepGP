#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 15:34:03 2024

@author: benavoli
"""

import torch
import gpytorch
from gpytorch.constraints import Positive, LessThan, Interval
from torchaudio.functional import convolve
from model.commonsLTI import build_A, build_leftconvolutionterm, calculate_Pinf
from gpytorch.constraints import Interval
import math
from model.utility import IdentityTrasf, split_increasing_sequences, extract_diagonal, calculate_kernel_br
from model.priors import LaplacePrior, CauchyPrior, HorseshoePrior
#import control as ct


def set_right_to_left_params_jordan(left,right):
    left.nu=right.nu
    left.pairs=right.pairs
    left.a_re = right.a_re
    left.a_im = right.a_im
    left.C_re = right.C_re
    left.C_im = right.C_im
    left.B_re = right.B_re
    left.B_im = right.B_im
    left.D = right.D


class LTIJordanCC(gpytorch.kernels.Kernel):
    # the sinc kernel is stationary
    is_stationary = False

    # We will register the parameter when initializing the kernel
    def __init__(self, samplingtime=1.0, pairs=1, number_in_b=1,  number_in_l=1, 
                 a_re_prior=None, a_re_constraint=None, 
                 a_im_prior=None, a_im_constraint=None, #s.HorseshoePrior(0.01), 
                   L_re_prior=HorseshoePrior(0.1), L_re_constraint=None, #gpytorch.constraints.Interval(0.0, 1e-6), #gpytorch.constraints.Interval(0.0, 1e-6), #gpytorch.priors.HorseshoePrior(0.1)
                   L_im_prior=HorseshoePrior(0.1),
                   L_im_constraint=None, #gpytorch.constraints.Interval(0.0, 1e-6),
                     B_re_prior=None, B_re_constraint=None,
                     B_im_prior=None, B_im_constraint=None, #gpytorch.constraints.Interval(0.0, 1e-6),
                     C_re_prior=None, C_re_constraint=None,
                     C_im_prior=None, C_im_constraint=None, #gpytorch.constraints.Interval(0.0, 1e-6),
                     D_prior=None, D_constraint=gpytorch.constraints.Interval(0.0, 1e-6), initialization_type="sampling_time_scaled",
                     
                    **kwargs):  #
        super().__init__(**kwargs)

        self.nu = number_in_b
        self.nl = number_in_l
        self.pairs = pairs
        self.ts = samplingtime
        self.jitter = 1e-6
        self.initialization_type = initialization_type
        

        if initialization_type=="sampling_time_scaled":
            f_nyq = torch.tensor(0.5/self.ts)
            init1 = (-2+torch.quasirandom.SobolEngine(*self.batch_shape).draw(pairs).T*1.95)*f_nyq
        else:

            init1 = (-20+torch.quasirandom.SobolEngine(*self.batch_shape).draw(pairs).T*19.5)
            f_nyq = torch.tensor(1.0)
            
        self.f_nyq = f_nyq
        #init1 = -f_nyq+torch.linspace(0,f_nyq/1.01,pairs)

        
        
        #init1 = -(5+torch.rand(*self.batch_shape, pairs)*4.95)*f_nyq
        #init1 = -2.0+torch.rand(*self.batch_shape, pairs)*1.9
        #shape = (*self.batch_shape, pairs)
        #init1 = init1.expand(*shape)
        
        self.register_parameter(
            name="raw_a_re",
            parameter=torch.nn.Parameter(init1),
        )
        self.register_parameter(
            name="raw_a_im",
            parameter=torch.nn.Parameter(f_nyq*1e-3*torch.randn(*self.batch_shape, pairs)),
        )
#ATTTENTION I have multipled L by torch.sqrt(f_nyq) in Q.
        self.register_parameter(
            name="raw_L_re",
            parameter=torch.nn.Parameter(1e-3*torch.randn(*self.batch_shape, pairs, self.nl)),
        )
        self.register_parameter(
            name="raw_L_im",
            parameter=torch.nn.Parameter(1e-3*torch.randn(*self.batch_shape, pairs, self.nl)),
        )
        self.register_parameter(
            name="raw_C_re",
            parameter=torch.nn.Parameter(3e-1*torch.randn(*self.batch_shape, 1, pairs)),
        )
        self.register_parameter(
            name="raw_C_im",
            parameter=torch.nn.Parameter(1e-3*torch.randn(*self.batch_shape, 1, pairs)),
        )
        self.register_parameter(
            name="raw_B_re",
            parameter=torch.nn.Parameter(f_nyq*3e-1*torch.randn(*self.batch_shape, pairs, self.nu)),
        )  # unused here
        self.register_parameter(
            name="raw_B_im",
            parameter=torch.nn.Parameter(f_nyq*1e-3*torch.randn(*self.batch_shape, pairs, self.nu)),
        )  # unused here
        self.register_parameter(
            name="raw_D",
            parameter=torch.nn.Parameter(1e-12*torch.randn(*self.batch_shape, 1, self.nu)),
        )  # unused here
        #self.register_parameter(
        #    name="raw_x0",
        #    parameter=torch.nn.Parameter(1e-12*torch.rand(*self.batch_shape, pairs, 1)),
        #) 


#        self.register_constraint("raw_a", LessThan(0.0))
        if a_re_constraint is None:
            a_re_constraint = LessThan(0)
        self.register_constraint("raw_a_re",  a_re_constraint)#Interval(-f_nyq, -f_nyq/10.0))
       
        if a_im_constraint is None:
            a_im_constraint = IdentityTrasf()
        self.register_constraint("raw_a_im",  a_im_constraint)
        
        if B_re_constraint is None:
           B_re_constraint = IdentityTrasf()
        if B_im_constraint is None:
           B_im_constraint = IdentityTrasf()
        self.register_constraint("raw_B_re", B_re_constraint)
        self.register_constraint("raw_B_im", B_im_constraint)
        
        if C_re_constraint is None:
           C_re_constraint = IdentityTrasf()
        if C_im_constraint is None:
           C_im_constraint = IdentityTrasf()
        self.register_constraint("raw_C_re", C_re_constraint)
        self.register_constraint("raw_C_im", C_im_constraint)
        
        if L_re_constraint is None:
           L_re_constraint = IdentityTrasf()
        if L_im_constraint is None:
           L_im_constraint = IdentityTrasf()
        self.register_constraint("raw_L_re", L_re_constraint)
        self.register_constraint("raw_L_im", L_im_constraint)
       
        if D_constraint is None:
           D_constraint = IdentityTrasf()
        self.register_constraint("raw_D", D_constraint)

        
        if a_re_prior is not None:
            self.register_prior(
                   "a_re_prior",
                   a_re_prior,
                   lambda m: m.a_re,
                   lambda m, v : m._set_a_re(v),
               )
        if a_im_prior is not None:
             self.register_prior(
                    "a_im_prior",
                    a_im_prior,
                    lambda m: m.a_im,
                    lambda m, v : m._set_a_im(v),
                )
        if L_re_prior is not None:
             self.register_prior(
                    "L_re_prior",
                    L_re_prior,
                    lambda m: m.L_re,
                    lambda m, v : m._set_L_re(v),
                )
        if L_im_prior is not None:
             self.register_prior(
                    "L_im_prior",
                    L_im_prior,
                    lambda m: m.L_im,
                    lambda m, v : m._set_L_im(v),
                )
        if B_re_prior is not None:
             self.register_prior(
                    "B_re_prior",
                    B_re_prior,
                    lambda m: m.B_re,
                    lambda m, v : m._set_B_re(v),
                )
        if B_im_prior is not None:
             self.register_prior(
                    "B_im_prior",
                    B_im_prior,
                    lambda m: m.B_im,
                    lambda m, v : m._set_B_im(v),
                )
        if C_re_prior is not None:
             self.register_prior(
                    "C_re_prior",
                    C_re_prior,
                    lambda m: m.C_re,
                    lambda m, v : m._set_C_re(v),
                )
        if C_im_prior is not None:
             self.register_prior(
                    "C_im_prior",
                    C_im_prior,
                    lambda m: m.C_im,
                    lambda m, v : m._set_C_im(v),
                )
        if D_prior is not None:
             self.register_prior(
                    "D_prior",
                    D_prior,
                    lambda m: m.D,
                    lambda m, v : m._set_D(v),
                )


    @property
    def a_re(self):
        # when accessing the parameter, apply the constraint transform
        return self.raw_a_re_constraint.transform(self.raw_a_re)    
    @a_re.setter
    def a_re(self, value):
        return self._set_a(value)
    def _set_a_re(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value)
        # when setting the paramater, transform the actual value to a raw one by applying the inverse transform
        self.initialize(raw_a_re=self.raw_a_re_constraint.inverse_transform(value))

    @property
    def a_im(self):
        # when accessing the parameter, apply the constraint transform
        return self.raw_a_im_constraint.transform(self.raw_a_im)
    @a_im.setter
    def a_im(self, value):
        return self._set_a_im(value)
    def _set_a_im(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value)
        # when setting the paramater, transform the actual value to a raw one by applying the inverse transform
        self.initialize(raw_a_im=self.raw_a_im_constraint.inverse_transform(value))

    @property
    def L_re(self):
        # when accessing the parameter, apply the constraint transform
        return self.raw_L_re_constraint.transform(self.raw_L_re)
    @L_re.setter
    def L_re(self, value):
        return self._set_L_re(value)
    def _set_L_re(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self._set_L_re)
        # when setting the paramater, transform the actual value to a raw one by applying the inverse transform
        self.initialize(raw_L_re=self.raw_L_re_constraint.inverse_transform(value))

    @property
    def L_im(self):
        # when accessing the parameter, apply the constraint transform
        return self.raw_L_im_constraint.transform(self.raw_L_im)
    @L_im.setter
    def L_im(self, value):
        return self._set_L_im(value)
    def _set_L_im(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self._set_L_im)
        # when setting the paramater, transform the actual value to a raw one by applying the inverse transform
        self.initialize(raw_L_im=self.raw_L_im_constraint.inverse_transform(value))
   
    @property
    def B_re(self):
        # when accessing the parameter, apply the constraint transform
        return self.raw_B_re_constraint.transform(self.raw_B_re)
    @B_re.setter
    def B_re(self, value):
        return self._set_B_re(value)
    def _set_B_re(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self._set_B_re)
        # when setting the paramater, transform the actual value to a raw one by applying the inverse transform
        self.initialize(raw_B_re=self.raw_B_re_constraint.inverse_transform(value))

    @property
    def B_im(self): 
        # when accessing the parameter, apply the constraint transform
        return self.raw_B_im_constraint.transform(self.raw_B_im)
    @B_im.setter
    def B_im(self, value):
        return self._set_B_im(value)
    def _set_B_im(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self._set_B_im)
        # when setting the paramater, transform the actual value to a raw one by applying the inverse transform
        self.initialize(raw_B_im=self.raw_B_im_constraint.inverse_transform(value))
    
    @property
    def C_re(self): 
        # when accessing the parameter, apply the constraint transform
        return self.raw_C_re_constraint.transform(self.raw_C_re)
    @C_re.setter
    def C_re(self, value):
        return self._set_C_re(value)
    def _set_C_re(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self._set_C_re)
        # when setting the paramater, transform the actual value to a raw one by applying the inverse transform
        self.initialize(raw_C_re=self.raw_C_re_constraint.inverse_transform(value))
   
    @property
    def C_im(self): 
        # when accessing the parameter, apply the constraint transform
        return self.raw_C_im_constraint.transform(self.raw_C_im)
    @C_im.setter
    def C_im(self, value):
        return self._set_C_im(value)
    def _set_C_im(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self._set_C_im)
        # when setting the paramater, transform the actual value to a raw one by applying the inverse transform
        self.initialize(raw_C_im=self.raw_C_im_constraint.inverse_transform(value))
    
    @property
    def D(self): 
        # when accessing the parameter, apply the constraint transform
        return self.raw_D_constraint.transform(self.raw_D)
    @D.setter
    def D(self, value):
        return self._set_D(value)
    def _set_D(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self._set_D)
        # when setting the paramater, transform the actual value to a raw one by applying the inverse transform
        self.initialize(raw_D=self.raw_D_constraint.inverse_transform(value))


    def forward(self, t1, t2, diag=False, **params):

        # b, n1, d = t1.shape
        # b_, n2, d_ = t2N\left({\bar{\bf y}};f(\mathbf{t}),\tfrac{1}{n}D(\mathbf{t},\mathbf{t};\boldsymbol{\theta})\right).shape
        nx = 2 * self.pairs

        lambdas = self.a_re + 1j * self.a_im
        lambdas_cc = torch.zeros(*self.batch_shape, nx, dtype=torch.complex64)
        lambdas_cc[..., ::2] = lambdas
        lambdas_cc[..., 1::2] = lambdas.conj()
        
        #print("l=",lambdas)

        # Assemble complex-conjugate matrix L
        Lc = self.L_re + 1j * self.L_im
        Lcc = torch.zeros((*self.batch_shape, nx, self.nl), dtype=torch.complex64)
        Lcc[..., ::2, :] = Lc
        Lcc[..., 1::2, :] = Lc.conj()
        #print("L=",torch.max(abs(Lcc.flatten())))
        # Assemble complex-conjugate matrix C
        Cc = self.C_re + 1j * self.C_im
        Ccc = torch.zeros(
            (*self.batch_shape, 1, nx), dtype=lambdas.dtype
        )  # batch_shape, ny, nx
        Ccc[..., :, ::2] = (
            Cc  # we take the real part of the complex conjugate system as output...
        )
        Ccc[..., :, 1::2] = Cc.conj()

        # Solve Lyapunov equation
        Qcc = -Lcc @ (Lcc.mH)*self.f_nyq #AAAAAAAAAAAAAAAAAAAA

        O = torch.ones(nx)
        lhs = torch.kron(O, lambdas_cc.conj()) + torch.kron(lambdas_cc, O)
        rhs = Qcc.flatten(start_dim=-2, end_dim=-1)
        Scc = rhs / lhs
        Scc = Scc.reshape(*self.batch_shape, nx, nx)  # *batch_shape, nx, nx

        # Compute the elements of the LTI kernel
        dt = t1 - t2.transpose(-2, -1)
        max_delay = torch.max(torch.abs(dt) / self.ts).int() + 1

        elambdas_cc = torch.exp(lambdas_cc * self.ts)  # 

        repeats = [1] * len(self.batch_shape) + [max_delay, 1]
        elambdas_pows = torch.cumprod(elambdas_cc.unsqueeze(-2).repeat(repeats), dim=-2)
        elambdas_pows = torch.cat((torch.ones_like(elambdas_pows[..., [0], :]), elambdas_pows.conj()#AAAA
                                   ), dim=-2)
        elambdas_pows = elambdas_pows.unsqueeze(-1) #*batch_shape, max_delay, nx, 1

        tmp1 = Ccc @ Scc  # batch_shape, ny, nx
        tmp2 = elambdas_pows * Ccc.mH.unsqueeze(-3)  # batch_shape, max_delay, nx, 1
        elems = (tmp1.unsqueeze(-3) @ tmp2).real  # batch_shape, max_delax, ny, ny

        elems = elems.squeeze(-1).transpose(
            -1, -2
        )  
        
        #K = calculate_kernel(dt, elems, self.ts)
        K = calculate_kernel_br(dt, elems, self.ts)
        
        
       
        #if K.shape[1]==K.shape[2]:
        #    eee = torch.min(torch.linalg.eig(K)[0].real)
        #    print(eee)
        #    if eee<-1e-4:
        #        print(eee)
        if dt.shape!=K.shape:
            K = K.expand(*dt.shape[0:-3], *K.shape)
        if diag:
            #n_dims = K.ndimension()
            # Build the index: [0, ..., 0, :, :]
            #indices = [0] * (n_dims - 2) + [slice(None), slice(None)] 
            return extract_diagonal(K)
            #if len(K.shape)==4:
            #    return torch.cat([torch.cat([K[i,j,:,:].diag()[None,:] for i in range(K.shape[0])],axis=0).unsqueeze(1) for j in range(K.shape[1]) ],axis=1)
            #elif len(K.shape)==3:
            #    return torch.cat([K[i,:,:].diag()[None,:] for i in range(K.shape[0])],axis=0)
        else:
            return K
    
    

class LTIJordanCCMean(gpytorch.means.Mean):
    def __init__(self,samplingtime,num_inducing, add_steps_ahead=500, batch_shape=torch.Size(())):
        super().__init__()

        self.samplingtime = samplingtime
        self.batch_shape = batch_shape

        self.nu=[]
        self.pairs=[]
        self.a_re = []
        self.a_im = []
        self.B_re = []
        self.B_im = []
        self.C_re = []
        self.C_im = []
        self.D = []
        self.previous_state = torch.tensor([0.0,0.0])
        self.add_steps_ahead = add_steps_ahead
        self.num_inducing = num_inducing

  
    def forward(self, x):
        device = x.device 
        n_dims = x.ndimension()
        # Build the index: [0, ..., 0, :, :]
        indices = [0] * (n_dims - 2) + [slice(None), slice(None)]
        t = x[...,0:1][indices] # x[...,0:1]
        u = x[...,1:]
        
        nx = 2*self.pairs
        
        lambdas = self.a_re + 1j * self.a_im
        #print(lambdas)
        lambdas_cc = torch.zeros(*self.batch_shape, nx, dtype=torch.cdouble)
        lambdas_cc[..., ::2] = lambdas
        lambdas_cc[..., 1::2] = lambdas.conj()
        
        elambdas = torch.exp(lambdas_cc * self.samplingtime)
        #elambdas = torch.exp(lambdas * self.ts)
        #print(lambdas)
        
        Ad = torch.diag_embed(elambdas)
        I = torch.stack([torch.eye(Ad.shape[1], dtype=torch.cdouble) for i in range(Ad.shape[0])])
        Bc = self.B_re + 1j * self.B_im
        Bcc = torch.zeros((*self.batch_shape, nx, self.nu), dtype=lambdas.dtype)
        Bcc[..., ::2, :] = Bc
        Bcc[..., 1::2, :] = Bc.conj()
        #print("B=",Bc)


        # discrete-time parameters
        Bd = ((1/lambdas_cc)*(elambdas-1)).unsqueeze(-1) * Bcc
        #(1/lambdas)*(elambdas-1)*Bc
        Cc = self.C_re + 1j * self.C_im
        Ccc = torch.zeros(
            (*self.batch_shape, 1, nx), dtype=lambdas.dtype
        )  # batch_shape, ny, nxself.a_re + 1j * self.a_im
        Ccc[..., :, ::2] = (
            Cc  # we take the real part of the complex conjugate system as output...
        )
        Ccc[..., :, 1::2] = Cc.conj()
        
        

        t_split,u_split=split_increasing_sequences(t,u,self.num_inducing )#check if there are consecutvie tme sequences
        
        
        out=[]
        vnext=[]
        self.previous_state = self.previous_state.to(device)
        for s in range(len(t_split)):
            #if s!=0: #len(t_split[s].flatten())!=self.num_inducing:
            #add steps ahead
            timef = t_split[s]- t_split[s][0,0]
            timeahead = torch.linspace(1,self.add_steps_ahead,self.add_steps_ahead,device=device)[:,None]
                #timeahead = timeahead.expand(timef.shape[0],timeahead.shape[0],1)
                #timeahead = timef[:,0,self.a_re + 1j * self.a_im-1][None,:][None,:].T+timeahead*self.samplingtime
            timeahead = timef[-1]+timeahead*torch.tensor(self.samplingtime,device=device)
            timeext = torch.cat([timef,
                                   timeahead
                                   ],dim=0)
            elems = (build_leftconvolutionterm(timeext,torch.tensor(self.samplingtime,device=device),Ccc,I,Ad)@Bd).transpose(1,2).real
            indrow=torch.abs(timeext)/torch.tensor(self.samplingtime,device=device)
            indrow = indrow.round().to(int).to(elems.device)
            #if len(indrow.shape)==4:
            #    cext= torch.stack([elems[i,:,indrow[0,:,:,0]] for i in range(elems.shape[0])])
            #elif len(indrow.shape)==3:
            cext= elems[...,indrow[:,0]] #torch.stack([elems[...,indrow[:,0]] for i in range(elems.shape[0])])
            ushape = list(u_split[s].shape)
            ushape[-2]= self.add_steps_ahead

            us = torch.cat([u_split[s],torch.zeros(ushape,device=device)],dim=-2)
            us = us.transpose(-1,-2)
            if len(us.shape)==4:
                cext = cext.expand(*ushape[0:-3],cext.shape[0],cext.shape[1],cext.shape[2])
            cext=cext.to(us.device)
            val = convolve(us,cext)[...,0:us.shape[-1]]
            v = val.sum(axis=-2)[...,0:timef.shape[0]]
            dimv = torch.minimum(torch.tensor(v.shape[-1]),torch.tensor(self.add_steps_ahead))
            v[...,0:dimv] = v[...,0:dimv]+self.previous_state[s,...,0:dimv]
            
            vnext.append((val.sum(axis=-2)[...,timef.shape[0]:])[None,:])
           
            out.append(v)
        self.previous_state= torch.cat(vnext,axis=0)
        self.previous_state = self.previous_state.detach()
        #for i in vnext.keys():
        #    self.previous_state[i] = vnext[i].detach() #retain values only
        out = torch.cat(out,dim=-1)    
        out = out+((self.D)@u.transpose(-2,-1))[...,0,:]
        return out
