import torch
import gpytorch
from gpytorch.constraints import Positive, LessThan, Interval
#from utility import solve_lyapunov
from torchaudio.functional import convolve
from model.commonsLTI import build_A, build_leftconvolutionterm, calculate_Pinf
from gpytorch.constraints import Interval
import math
from model.utility import IdentityTrasf, split_increasing_sequences, calculate_kernel_br,extract_diagonal
from model.priors import LaplacePrior

def set_right_to_left_params_lowrank(left,right):
    left.a_d = right.a_d
    left.a_v = right.a_v
    #left.a_w = right.a_w
    left.b = right.b
    left.c = right.c
    left.d = right.d





class LTIlowrank(gpytorch.kernels.Kernel):
    # the sinc kernel is stationary
    is_stationary = False

    # We will register the parameter when initializing the kernel
    def __init__(self, samplingtime, num_states=3, number_in_b=1,number_in_l=1,
                 a_d_prior=None, a_d_constraint=None, 
                 a_v_prior=None, a_v_constraint=None,
                 #a_w_prior=None, a_w_constraint=None,
                 b_prior=None, b_constraint=None,
                 l_prior=gpytorch.priors.HorseshoePrior(0.01), l_constraint=None,
                 c_prior=None, c_constraint=None,
                 d_prior=None, d_constraint=gpytorch.constraints.Interval(0.0, 1e-6),
                 **kwargs): #
        super().__init__(**kwargs)
        self.number_output = self.batch_shape.numel()
        
        f_nyq=0.5/samplingtime

        # register the raw parameter
        #init = -2*torch.rand(*self.batch_shape, num_states, 1)
        init1 = -torch.rand(*self.batch_shape, num_states,1)-f_nyq/5
        self.register_parameter(
            name='raw_a_d', parameter=torch.nn.Parameter(init1)
        )
        # register the raw parameter
        self.register_parameter(
            name='raw_a_v', parameter=torch.nn.Parameter(-1*torch.ones(*self.batch_shape, num_states, 1)-f_nyq/5)
        )
        # register the raw parameter
        #self.register_parameter(
        #    name='raw_a_w', parameter=torch.nn.Parameter(0.1*torch.rand(*self.batch_shape, num_states, 1))
        #)
        # register the raw parameter
        init2 =  torch.rand(*self.batch_shape,1, num_states)
        #init[...,0,0] = torch.ones(*self.batch_shape)
        self.register_parameter(
            name='c', parameter=torch.nn.Parameter(init2)
        )
        # register the raw parameter
        init = torch.rand(*self.batch_shape,num_states, number_in_b)
        #init[...,0,0] = torch.ones(*self.batch_shape)
        self.register_parameter(
            name='b', parameter=torch.nn.Parameter(init)
        )
        # register the raw parameter
        self.register_parameter(
            name='raw_l', parameter=torch.nn.Parameter(1e-12*torch.rand(*self.batch_shape,num_states, number_in_l))
        )
        # register the raw parameter
        self.register_parameter(
            name='raw_d', parameter=torch.nn.Parameter(1e-12*torch.rand(*self.batch_shape,1, number_in_b))
        )


        if a_d_constraint is None:
            a_d_constraint = LessThan(0)#diagonal of A
        if a_v_constraint is None:
            a_v_constraint = Positive()#lowrank left-matrix
        #if a_w_constraint is None:
        #    a_w_constraint = Positive()#lowrank right-matrix
        if l_constraint is None:
           l_constraint = IdentityTrasf()
        if d_constraint is None:
           d_constraint = IdentityTrasf()


        # register the constraint
        self.register_constraint("raw_a_d", a_d_constraint)
        #self.register_constraint("raw_a_w", a_w_constraint)
        self.register_constraint("raw_a_v", a_v_constraint)
        self.register_constraint("raw_l", l_constraint)
        self.register_constraint("raw_d", d_constraint)
        
        # set the parameter prior, see
        # https://docs.gpytorch.ai/en/latest/module.html#gpytorch.Module.register_prior
        if a_d_prior is not None:
            raise NotImplementedError
        if l_prior is not None:
             self.register_prior(
                    "l_prior",
                    l_prior,
                    lambda m: m.l,
                    lambda m, v : m._set_l(v),
                )
             
        if d_prior is not None:
             self.register_prior(
                    "d_prior",
                    d_prior,
                    lambda m: m.d,
                    lambda m, v : m._set_d(v),
                )
            
        self.jitter = 1e-6
        self.samplingtime = samplingtime
        self.num_states=num_states
        self.left_convolution = []
            
    # now set up the 'actual' paramter
    @property
    def a_d(self):
        # when accessing the parameter, apply the constraint transform
        return self.raw_a_d_constraint.transform(self.raw_a_d)

    @a_d.setter
    def a_d(self, value):
        return self._set_a_d(value)

    def _set_a_d(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self._set_a_d)
        # when setting the paramater, transform the actual value to a raw one by applying the inverse transform
        self.initialize(raw_a_d=self.raw_a_d_constraint.inverse_transform(value))
        
    @property
    def a_v(self):
        # when accessing the parameter, apply the constraint transform
        return self.raw_a_v_constraint.transform(self.raw_a_v)

    @a_v.setter
    def a_v(self, value):
        return self._set_a_v(value)

    def _set_a_v(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self._set_a_v)
        # when setting the paramater, transform the actual value to a raw one by applying the inverse transform
        self.initialize(raw_a_v=self.raw_a_v_constraint.inverse_transform(value))

    @property
    def l(self):
        # when accessing the parameter, apply the constraint transform
        return self.raw_l_constraint.transform(self.raw_l)

    @l.setter
    def l(self, value):
        return self._set_l(value)

    def _set_l(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self._set_l)
        # when setting the paramater, transform the actual value to a raw one by applying the inverse transform
        self.initialize(raw_l=self.raw_l_constraint.inverse_transform(value))
    
    
    @property
    def d(self):
        # when accessing the parameter, apply the constraint transform
        return self.raw_d_constraint.transform(self.raw_d)
    @d.setter
    def d(self, value):
        return self._set_d(value)
    def _set_d(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self._set_d)
        # when setting the paramater, transform the actual value to a raw one by applying the inverse transform
        self.initialize(raw_d=self.raw_d_constraint.inverse_transform(value))
    
    
    
    

    
    
        
    # this is the kernel function
    def forward(self, t1, t2, diag = False, **params):
        #calculate time difference
        dt =t1-t2.transpose(-2, -1)
        #print(dt.shape)
        #print("a_d",self.a_d)
        #print("a_v",self.a_v)
        #print("a_w",self.a_w)
        #print("l",self.l)
        
        
        ######################################################
        A = build_A(self.a_d,self.a_v,self.a_v) ###########
        Pinf = calculate_Pinf(A,self.l,self.jitter)

        #Ad = torch.matrix_exp(A*self.samplingtime)
        
        M = (1.0 / ((2.0 / self.samplingtime) - self.a_d))
        M = torch.stack([torch.diag_embed(M[i][:,0]) for i in range(M.shape[0])])
        Vt = self.a_v.transpose(2,1)
        V = self.a_v
        A1 = M - (M @ V * (1.0 / (1 + (Vt @ M @ V))) * Vt @ M)
        I = torch.stack([torch.eye(self.a_d.shape[1]) for i in range(self.a_d.shape[0])])
        A0 = (2.0 / self.samplingtime) * I + A
        Ad= A1 @ A0
        
        #Cp=self.c@Pinf
        Ct= self.c.transpose(1, 2).contiguous()
        elems = (build_leftconvolutionterm(dt,self.samplingtime,self.c,Pinf,Ad.transpose(1,2))@Ct).transpose(1,2)

      
        #K0 = calculate_kernel(dt, elems, self.samplingtime)
        K = calculate_kernel_br(dt, elems, self.samplingtime)
        if dt.shape!=K.shape:
            K = K.expand(*dt.shape[0:-3], *K.shape)
        if diag:
            return extract_diagonal(K)
            #if len(K.shape)==4:
            #    return torch.cat([K[i,0,:,:].diag()[None,:] for i in range(K.shape[0])],axis=0).unsqueeze(1)
            #elif len(K.shape)==3:
            #    return torch.cat([K[i,:,:].diag()[None,:] for i in range(K.shape[0])],axis=0)
        else:
            return K
        


class LTIlowrankMean(gpytorch.means.Mean):
    def __init__(self,samplingtime,add_steps_ahead=500):
        super().__init__()
        self.samplingtime = samplingtime
        self.a_d=[]
        self.a_v=[]
        self.a_w=[]
        self.b=[]
        self.c=[]
        self.d=[]
        self.previous_state = torch.tensor([0.0,0.0])
        self.add_steps_ahead = add_steps_ahead
        

    def forward(self, x):
        n_dims = x.ndimension()
        # Build the index: [0, ..., 0, :, :]
        indices = [0] * (n_dims - 2) + [slice(None), slice(None)]
        t = x[...,0:1][indices] # x[...,0:1]
        u = x[...,1:]
        
        A = build_A(self.a_d,self.a_v,self.a_v) 

        I = torch.stack([torch.eye(self.a_d.shape[1]) for i in range(self.a_d.shape[0])])
        #Ad = torch.matrix_exp(A*self.samplingtime)
        #Bd = torch.matmul(torch.linalg.inv(A), (Ad - I)).matmul(self.b)
        
        M = (1.0 / ((2.0 / self.samplingtime) - self.a_d))
        M = torch.stack([torch.diag_embed(M[i][:,0]) for i in range(M.shape[0])])
        Vt = self.a_v.transpose(2,1)
        V = self.a_v
        A1 = M - (M @ V * (1.0 / (1 + (Vt @ M @ V))) * Vt @ M)
        I = torch.stack([torch.eye(self.a_d.shape[1]) for i in range(self.a_d.shape[0])])
        A0 = (2.0 / self.samplingtime) * I + A
        Ad= A1 @ A0
        
 
 
        #
        #Ad = torch.matrix_exp(A*self.samplingtime)
        Bd = 2*A1@self.b #torch.matmul(torch.linalg.inv(A), (Ad - I)).matmul(self.b)
        

        t_split,u_split=split_increasing_sequences(t,u)#check if there are consecutvie tme sequences


        out=[]
        vnext=[]
        for s in range(len(t_split)):
            #if s!=0: #len(t_split[s].flatten())!=self.num_inducing:
            #add steps ahead
            timef = t_split[s]- t_split[s][0,0]
            timeahead = torch.linspace(1,self.add_steps_ahead,self.add_steps_ahead)[:,None]
            timeahead = timef[-1]+timeahead*self.samplingtime
            timeext = torch.cat([timef,
                                   timeahead
                                   ],dim=0)
            elems = (build_leftconvolutionterm(timeext,self.samplingtime,self.c,I,Ad)@Bd).transpose(1,2)
            indrow=torch.abs(timeext)/self.samplingtime
            indrow = indrow.round().to(int)
            #if len(indrow.shape)==4:
            #    cext= torch.stack([elems[i,:,indrow[i,:,:,0]] for i in range(elems.shape[0])])
            #elif len(indrow.shape)==3:
            #    cext= torch.stack([elems[i,:,indrow[i,:,0]] for i in range(elems.shape[0])])
            cext= elems[...,indrow[:,0]]
            ushape = list(u_split[s].shape)
            ushape[-2]= self.add_steps_ahead
            
            us = torch.cat([u_split[s],torch.zeros(ushape)],dim=-2)
            us = us.transpose(-1,-2)
            
            if len(us.shape)==4:
                cext = cext.expand(*ushape[0:-3],cext.shape[0],cext.shape[1],cext.shape[2])
            val = convolve(us,cext)[...,0:us.shape[-1]]
            v = val.sum(axis=-2)[...,0:timef.shape[0]]
            dimv = torch.minimum(torch.tensor(v.shape[-1]),torch.tensor(self.add_steps_ahead))
            v[...,0:dimv] = v[...,0:dimv]+self.previous_state[s,...,0:dimv]
            
            vnext.append((val.sum(axis=-2)[...,timef.shape[0]:])[None,:])

            out.append(v)
        self.previous_state= torch.cat(vnext,axis=0)
        self.previous_state = self.previous_state.detach() #retain values only
        out = torch.cat(out,dim=-1)    
        out = out+((self.d)@u.transpose(-2,-1))[...,0,:]
        return out
