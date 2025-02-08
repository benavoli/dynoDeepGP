#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 09:59:33 2024

@author: benavoli
"""




import gpytorch
from gpytorch.mlls import DeepApproximateMLL
from gpytorch.mlls import VariationalELBO
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import TensorDataset, DataLoader
import torch
import tqdm
from model.utility import IdentityTrasf
import math
    
def train_gp(model,train_x,train_y,test_x,training_iter=1500,batch_size=350,learning_rate=0.002,num_samples_likelihood_tr=10):
    
    device= train_x.device
       
    mll = DeepApproximateMLL(VariationalELBO(model.likelihood, model, train_x.shape[-2]))#
    
    model.train()
    
    #make dataset    
    train_dataset = TensorDataset(train_x.transpose(0,1), train_y.transpose(0,1))
    #sequential batch
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    
      
    
    num_samples=num_samples_likelihood_tr# it uses samples to propagate the GP posterior from a layer to the next one
    
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
    ], lr=learning_rate) #0.002

    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(0.9*training_iter)], gamma=0.5)
    #model.allx = train_x
    for i in range(training_iter):
        minibatch_iter = tqdm.tqdm(train_loader, desc=f"Iter {i + 1}/{training_iter}", unit="batch")
       
        #if i==200:
        #    model.lti_layer.covar_module.raw_L_re_constraint.lower_bound=torch.tensor(-1e2,device=device)
        #    model.lti_layer.covar_module.raw_L_re_constraint.upper_bound=torch.tensor(+1e2,device=device)
        #    model.lti_layer.covar_module.raw_L_im_constraint.lower_bound=torch.tensor(-1e2,device=device)
        #    model.lti_layer.covar_module.raw_L_im_constraint.upper_bound=torch.tensor(+1e2,device=device)
        for ui in range(train_x.shape[0]):
            model.batch_restart=True#IMPORTANT to tell the model is running a batch
            for x_batch, y_batch in minibatch_iter:
    
                with gpytorch.settings.num_likelihood_samples(num_samples):
                    # Zero gradients from previous iteration
                    optimizer.zero_grad()
                    # Output from model
                    output = model(x_batch.transpose(0,1)[ui])
                    loss = -mll(output, y_batch.transpose(0,1)[ui])
                    loss.backward()
                    optimizer.step()
                    model.batch_restart=False
                    minibatch_iter.set_postfix(loss=loss.item())
                    #print(loss.item())
                    
        
            torch.cuda.empty_cache()
        scheduler.step()
    model.eval()#THIS IS IMPORTANT. The forward model is different in prediction mode
    model.batch_restart=True #THIS IS IMPORTANT, it erases past mean
    
    bsize = 2000
    pp_means=[]
    pp_variances=[]
    for ui in range(test_x.shape[0]):
        ppm=[]
        ppc=[]
        for i in range(200):
            with gpytorch.settings.num_likelihood_samples(1):#we can use more samples for testing
                predictive_means=[]
                predictive_variances=[]
                model.batch_restart=True
                for b in range(int(torch.ceil(torch.tensor(test_x[ui].shape[0]/bsize)))):
                    p_means, p_variances = model.predict(test_x[ui][b*bsize:(b+1)*bsize,:])
                    model.batch_restart=False
                    if b==0:
                        predictive_means=p_means
                        predictive_variances = p_variances
                    else:
                        predictive_means = torch.cat([predictive_means,p_means],axis=-1)
                        predictive_variances = torch.cat([predictive_variances,p_variances],axis=-1)
                ppm.append(predictive_means)
                ppc.append(predictive_variances)
        pp_means.append(torch.cat(ppm,axis=0))
        pp_variances.append(torch.cat(ppc,axis=0))
    predictive_means = torch.cat([pp_means[i].unsqueeze(0) for i in range(len(pp_means))],axis=0)
    predictive_variances = torch.cat([pp_variances[i].unsqueeze(0) for i in range(len(pp_variances))],axis=0)
    return predictive_means.detach().cpu().numpy(),predictive_variances.detach().cpu().numpy()
