#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 16:45:07 2024

@author: benavoli
"""

import torch


def last_positive_or_zero_indices(tensor):
    pos_zero_mask = tensor >= 0  # Mask where elements are positive or zero
    device = tensor.device
    idx =  torch.arange(tensor.size(1), device=device).unsqueeze(0).expand_as(tensor) + torch.tensor(1, device=device)  # Row indices
    pos_zero_indices = pos_zero_mask * idx  # Multiply mask with indices
    pos_zero_indices[pos_zero_indices == 0] = - \
        1  # Set negative element indices to -1
    # Get the max index (last positive or zero)
    last_pos_zero_indices = pos_zero_indices.max(dim=1).values
    return last_pos_zero_indices- torch.tensor(1, device=device)


class zoh_old:
    def __init__(self, time, u):
        self.time = time.flatten()
        self.rest = u

    def __call__(self, t):
        # ind=torch.where(t.flatten()>=self.time)[0]
        ind = last_positive_or_zero_indices((t.flatten()[:, None]-self.time))
        if len(ind) == 0:
            return torch.tensor([0.0])
        return self.rest[..., ind, :]
    
    
class zoh:
    def __init__(self, time, u):
        self.time = time.flatten()
        self.rest = u

    def __call__(self, t):
        # ind=torch.where(t.flatten()>=self.time)[0]
        ind = last_positive_or_zero_indices((t.flatten()[:, None]-self.time))
        if len(ind) == 0:
            return torch.tensor([0.0])
        return self.rest[..., ind, :]


def trajectoryLTI(N, N_train, t, ufun, Ad, Bd, Cd, Qd):
    x = torch.zeros(1, Ad.shape[1], 1)
    u = []
    y = []
    for k in range(N):
        u.append(ufun(t[[k]]))
        w = torch.randn(1, Ad.shape[1], 1)
        if k>N_train:
            w=w*0.0
        x = Ad@x+Bd@u[-1] + \
            torch.linalg.cholesky(Qd)@w
        y.append(Cd@x)
    u = torch.cat(u, dim=2).transpose(1, 2)
    y = torch.cat(y)[:, 0, 0]
    return u, y


def solve_lyapunov(A, Q, type="continuous"):
    """
    Solves the continuous-time Lyapunov equation AX + XA^T + Q = 0
    using the Kronecker product method.

    Parameters:
        A (torch.Tensor): The matrix A of shape (n, n).
        Q (torch.Tensor): The matrix Q of shape (n, n).

    Returns:
        X (torch.Tensor): The solution matrix X of shape (n, n).
    """
    # Ensure A and Q are square matrices and of the same size

    num_systems = A.shape[0]
    n = A.shape[1]
    # assert A.shape == (num_systems,n, n), "A must be a square matrix"
    # assert Q.shape == (num_systems,n, n), "Q must be a square matrix"

    # Compute the Kronecker product of A and I, and I and A^T
    # At = A.transpose(1, 2).contiguous()
    # Vectorize the matrix Q
    Q_vec = Q.reshape(num_systems, -1)
    if type == "discrete":
        I = torch.stack([torch.eye(n**2) for i in range(num_systems)])
        A_kron = I-torch.kron(A, A)
    elif type == "continuous":
        I = torch.stack([torch.eye(n) for i in range(num_systems)])

        A_kron = (torch.stack([torch.kron(A[i], I[i]) for i in range(A.shape[0])]) +
                  torch.stack([torch.kron(I[i], A[i])
                              for i in range(A.shape[0])])
                  )  # .transpose(1,2).contiguous()
        Q_vec = - Q_vec

    '''     
    # Solve the linear system A_kron @ X_vec = -Q_vec
    #print(A_kron.shape, Q_vec.shape)
    # Iterate over the first dimension
    kron_list=[]
    for i in range(A.shape[0]):
        # Compute the Kronecker product of the i-th slice of A and I
        kron_prod = torch.kron(A[i], I[i])
        # Append the result to the list
        kron_list.append(kron_prod)

    # Stack the list along the first dimension to form the final tensor
    A_kron = torch.stack(kron_list)
    '''
    X_vec = torch.linalg.solve(A_kron, Q_vec)

    # Reshape the vectorized solution back to a matrix
    X = X_vec.reshape(num_systems, n, n)

    return X


def build_A(a_d, a_v, a_w):
    Lambda = torch.stack([torch.diag_embed(a_d[i][:, 0])
                         for i in range(a_d.shape[0])])
    # n = Lambda.shape[1] # number of states
    V = a_v
    W = a_w
    A = Lambda-V@W.transpose(1, 2).contiguous()
    return A


def build_leftconvolutionterm(t, samplingtime, c, Pinf, Ad):
    Cp = c@Pinf
    t_unique = torch.unique((torch.abs(t.flatten())/samplingtime).round())
    # elems = torch.cat([compute_ct_ad_c(Cp, Ad.transpose(1,2), torch.abs(d))@Ct for d in dt_unqiue],dim=-1)
    left_convolution_term = (
        torch.cat(compute_ct_ad(Cp, Ad, torch.max(t_unique)), dim=-2))
    return left_convolution_term


def build_leftconvolutionterm_lowrank(t, samplingtime, c, Pinf, a_d, a_v):
    Cp = c@Pinf
    tildeLamb = (1.0 / ((2.0 / samplingtime) - a_d))
    aux = tildeLamb.transpose(2, 1)*a_v.transpose(2, 1)
    r0 = 1/(1+torch.sum(a_v.transpose(2, 1)*aux, axis=2))
    R1 = tildeLamb*((2/samplingtime)+a_d)
    t_unique = torch.unique((torch.abs(t.flatten())/samplingtime).round())
    # elems = torch.cat([compute_ct_ad_c(Cp, Ad.transpose(1,2), torch.abs(d))@Ct for d in dt_unqiue],dim=-1)
    left_convolution_term = (torch.cat(compute_ct_ad_lowrank(
        Cp, r0, R1.transpose(2, 1), tildeLamb, a_v, torch.max(t_unique)), dim=-2))
    return left_convolution_term


def calculate_Pinf(A, l, jitter):
    L = l

    Q = torch.stack([L[i]@L[i].T+torch.eye(L[i].shape[0])
                    * jitter for i in range(L.shape[0])])
    # print(Q.shape)
    Pinf = solve_lyapunov(A, Q, type="continuous")
    return Pinf


def compute_ct_ad_lowrank(C, r0, R1t, a_v, tildeLamb, d):
    """
    Computes C^T A^d C efficiently using iterative matrix-vector multiplication.

    Parameters:
    C (torch.Tensor): 1D tensor of size n.
    A (torch.Tensor): 2D tensor of size nxn.
    d (int): The exponent to which matrix A is raised.

    Returns:
    torch.Tensor: The scalar result of C^T A^d C.
    """
    # Step 1: Start with the initial vector as C
    result_vector = [C]

    # Step 2: Iteratively multiply by A, d times
    for _ in range(d.round().to(torch.int)):
        c1 = result_vector[-1]*R1t
        c2 = (result_vector[-1]@a_v)*a_v.transpose(2, 1) * \
            tildeLamb.transpose(2, 1)*r0
        result_vector.append(c1-c2*R1t-c2)

    # Step 3: Compute the final dot product C^T * result_vector
    # result = torch.dot(C, result_vector)

    return result_vector


def compute_ct_ad(C, A, max_d):
    """
    Computes C^T A^d C efficiently using iterative matrix-vector multiplication.

    Parameters:
    C (torch.Tensor): 1D tensor of size n.
    A (torch.Tensor): 2D tensor of size nxn.
    d (int): The exponent to which matrix A is raised.

    Returns:
    torch.Tensor: The scalar result of C^T A^d C.
    """
    # Step 1: Start with the initial vector as C
    result_vector = [C]

    # Step 2: Iteratively multiply by A, d times
    for _ in range(max_d.round().to(torch.int)):
        result_vector.append(torch.matmul(result_vector[-1], A))

    # Step 3: Compute the final dot product C^T * result_vector
    # result = torch.dot(C, result_vector)

    return result_vector
