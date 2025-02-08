#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 11:06:32 2024

@author: benavoli
"""

import torch
from torch.autograd import Variable
import numpy as np
import gpytorch
import scipy
import torch
from gpytorch.constraints import Interval
import math

def extract_diagonal(K):
    if K.ndim < 2:
        raise ValueError("Tensor must have at least 2 dimensions")
    
    # Get the size of the batch dimensions
    batch_shape = K.shape[:-2]
    inner_shape = K.shape[-2:]  # The last two dimensions (matrices)
    
    if len(batch_shape) == 0:  # If there's no batch dimension
        return K.diag()[None, :]
    
    # Iterate over the batch dimensions dynamically
    diagonals = []
    for index in torch.cartesian_prod(*[torch.arange(s) for s in batch_shape]):
        index = tuple(index.tolist())  # Convert Tensor to tuple for indexing
        diag = K[index].diag()[None, :]  # Extract diagonal for the 2D matrix
        diagonals.append(diag)
    
    # Combine results into a single tensor
    return torch.cat(diagonals).view(*batch_shape, inner_shape[0])

def donothing(x):
    return x

class IdentityTrasf(Interval):
    def __init__(self, transform=donothing, inv_transform=donothing, initial_value=None):
        super().__init__(
            lower_bound=-math.inf,
            upper_bound=math.inf,
            transform=transform,
            inv_transform=inv_transform,
            initial_value=initial_value,
        )
    '''
    def __repr__(self) -> str:
        if self.lower_bound.numel() == 1:
            return self._get_name() + f"({self.lower_bound:.3E})"
        else:
            return super().__repr__()
    '''
    def transform(self, tensor):
        #transformed_tensor = self._transform(tensor) + self.lower_bound if self.enforced else tensor
        return tensor

    def inverse_transform(self, transformed_tensor):
        #tensor = self._inv_transform(transformed_tensor - self.lower_bound) if self.enforced else transformed_tensor
        return transformed_tensor

def split_increasing_sequences(t,u,hard_split=[]):
    
    # Calculate the differences between consecutive elements
    diffs = torch.diff(t,dim=0)
    
    # Check if there's any jump (large difference) indicating the start of a new sequence
    split_index = torch.where(diffs<0)[0] 
    if len(split_index)==0:
        split_index = [hard_split-1]

        

    # If the maximum difference is less than or equal to zero, then there's only one sequence
    if len(split_index)>1:
        raise ValueError("this should not happen")
    if len(split_index)>0:
        seq1=[]
        seq2=[]
        u1=[]
        u2=[]
        for i in range(len(split_index)):
            #if len(split_index[i])>0:
            s = split_index[i]+1
            # Split the tensor into two sequences
            seq1.append(t[:s])
            seq2.append(t[s:])
            u1.append(u[...,:s,:])
            u2.append(u[...,s:,:])
        #seq1 = torch.stack(seq1)
        #seq2 = torch.stack(seq2)
        #u1 = torch.stack(u1)
        #u2 = torch.stack(u2)
        return [seq1[0],seq2[0]], [u1[0],u2[0]]
    else:
        return [t], [u]
    
def select_elements(arr, k):
    # Initialize the result list
    selected_elements = []
    
    # Iterate through the array using i*k as the index
    for i in range(0, len(arr) // k + 1):
        index = i * k
        if index < len(arr):
            selected_elements.append(arr[index])
    
    return selected_elements


def calculate_kernel(dt, elems, samplingrate):

     if len(dt.shape) == 4:
         flat=dt[0, 0, 0, :].flatten()            
     elif len(dt.shape) == 3:
         flat = dt[0, 0, :].flatten()
        
     dt_1strow = torch.abs(flat) / samplingrate
     ee = elems[..., dt_1strow.round().to(int)]
     #ind=torch.where(flat>0)[0]
     #ee[ind]=ee[ind].conj()
     rows = [ee]

     for i in range(1, dt.shape[-2]):
         if len(dt.shape) == 4:
             flat=dt[0, :, i, :].flatten() 
         elif len(dt.shape) == 3:
             flat=dt[0, i, :].flatten()
         indrow = torch.abs(flat)/ samplingrate
         ee = elems[..., indrow.round().to(int)]
         #ind=torch.where(flat>0)[0]
         #ee[ind]=ee[ind].conj()
         rows.append(ee)

     # Stack the rows to form the matrix
     K = torch.cat(rows, dim=-2)  # dim=1
     return K
 
def calculate_kernel_br(dt, elems, samplingrate):
    elems = elems.to(dt.device)
    n_dims = dt.ndimension()
    # Build the index: [0, ..., 0, :, :]
    indices = [0] * (n_dims - 2) + [slice(None), slice(None)]
    
    # Calculate indices for all rows at once using broadcasting
    #if len(dt.shape) == 4:
    #    dt_flat = dt[0, 0, :, :].reshape(dt.shape[2], dt.shape[3])  
    #elif len(dt.shape) == 3:
    #    dt_flat = dt[0, :, :].reshape(dt.shape[1], dt.shape[2])
    dt_flat=dt[indices]
    # Compute the indices for all rows based on the dt values
    dt_indices = torch.abs(dt_flat) / samplingrate
    dt_rounded_indices = dt_indices.round().to(torch.int).to(dt.device)

    # Use the rounded indices to select the corresponding elements from `elems`
    rows = elems[..., dt_rounded_indices]
    
    # Reshape the result to form the final matrix K
    #K = rows[0,...] #.reshape(*dt.shape[:-2], elems.shape[-1])
    # if len(rows.shape)==4:
    #    K = rows[:,0,...] 
    #elif len(rows.shape)==3:
    #    K = rows[0,...] 
    K = rows[...,0,:,:]
    return K

class MixtureNorm:
    def __init__(self, mus, sigmas, alphas):
        self.mus = mus
        self.sigmas = sigmas
        self.alphas = alphas
    def cdf(self, x):
        cdf = 0
        for mu, sigma, alpha in zip(self.mus, self.sigmas, self.alphas):
            cdf += scipy.stats.norm.cdf(x, loc=mu, scale=sigma) * alpha
        return cdf
def crps_mixnorm(x, mus, sigmas, alphas):
    M = len(mus)
    
    def F(x):
        total = 0
        for i in range(M):
            total += alphas[i] * scipy.stats.norm.cdf((x-mus[i])/sigmas[i])
        return total
    
    def A(mu1, mu2, sig1, sig2):
        sig = np.sqrt(sig1**2 + sig2**2)
        sx = (mu1-mu2)/sig
        return sx * sig * (2*scipy.stats.norm.cdf(sx) - 1) + 2*sig*scipy.stats.norm.pdf(sx)
        
    total = 0
    for i in range(M):
        total += alphas[i] * A(x,mus[i], 0,sigmas[i])
    
    for i in range(M):
        for j in range(M):
            total -= 0.5 * alphas[i] * alphas[j] * A(mus[i],mus[j], sigmas[i],sigmas[j])
    return total