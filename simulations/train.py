#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 09:35:30 2024

@author: benavoli
"""
import torch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nonlinear_benchmarks # used just for metrics
import pandas as pd

from model.gp_train_batch import train_gp
from model.models import DeepGP_Wiener, DeepGP_Wiener_Hammerstein, DeepGP_Hammerstein_Wiener, DeepGP_nonLin2, DeepGP_LTI_only
from dynonetloc.dynonet_train import train_dynonet
from functions.scaler import StandardScaler
import pickle
def run(device,y,u,t,deltat,N_train,name_exp,path_exp,n_a,n_hidden_NN,
        num_inducing_dynlayer,num_inducing_nonlin,
        shiftp = 200,itertr_NN=10000,itertr_GP=1000,typeD='Wiener',batch_size=500,
        strategy_inducing_init_dynamic="multinomial",GP_custom=[],NN_custom=[],learning_rate_GP=0.002,
        initialization_type="sampling_time_scaled",scale_u=True,scale_y=True,num_samples_likelihood_tr=10):
    

    #prepare data   
    
    time_tr = t[0:N_train, :]
    time_te = t
    
    uscaler=StandardScaler()
    if scale_u==True:
        uscaler.fit(u[:, 0:N_train, :],dim=1)
    elif scale_u=='all':
        uscaler.fit(u[:, :, :],dim=1)
    uscaler.transform(u[:, 0:N_train, :])
    train_x_orig = torch.cat(
        [time_tr.expand(u.shape[0],*time_tr.shape), u[:, 0:N_train, :]], dim=2
    )
    train_x = torch.cat(
        [time_tr.expand(u.shape[0],*time_tr.shape), uscaler.transform(u[:, 0:N_train, :])], dim=2
    )  # the input is the stacking of time and u [t,u]
    yscaler = StandardScaler()
    if scale_y==True:
        yscaler.fit(y[:,0:N_train],dim=1)
    elif scale_y=='all':
        yscaler.fit(y[:,:],dim=1)
    train_y_orig=y[:,0:N_train]
    train_y = yscaler.transform(y[:,0:N_train])
    train_u = uscaler.transform(u[:, 0:N_train, :])
    
    # we test it on all the data
    test_x_origin = torch.cat([time_te.expand(u.shape[0],*time_te.shape), u], dim=2)
    test_x = torch.cat([time_te.expand(u.shape[0],*time_te.shape), uscaler.transform(u)], dim=2)
    test_y_orig= y
    test_y = yscaler.transform(y)
    
    
    shiftp0 = shiftp
    shiftp = np.minimum(shiftp0,N_train)
    test_x = test_x[:,N_train-shiftp:,...]
    test_x_orig = test_x_origin[:,N_train-shiftp:,...]
    test_y = test_y[:,N_train-shiftp:,...]
    test_y_orig = test_y_orig[:,N_train-shiftp:,...]
    test_u = uscaler.transform(u[:,N_train-shiftp:, ...])
    
    Data=[train_x_orig,train_y_orig,test_x_orig,test_y_orig]
    
    with open(path_exp+name_exp+"data", 'wb') as handle:
        pickle.dump(Data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
    number_in_b = u.shape[2]
    n_input=number_in_b
    n_output=1
    n_a=n_a#[10,10]
    n_b=n_a#[10,10]
    n_hidden=n_hidden_NN#[32]
    
    if (NN_custom!=[])&(typeD!="custom"):
        raise ValueError("Are you runnning a custom dynoNET model?")

    pred_NN = train_dynonet(n_input,n_output,n_a,n_b,n_hidden,train_y,test_y,train_u,test_u,itertr=itertr_NN,typeD=typeD,DeepNNmodel=NN_custom)#10000
    test_y_hat_NN = yscaler.inverse_transform( torch.tensor(pred_NN[...,0])).numpy()

    np.savetxt(path_exp+name_exp+"dyno.csv", test_y_hat_NN, delimiter=",")
    
        
    #test_y_hat = test_y_hat_NN.copy()
    #plt.figure()
    #plt.plot(test_y.numpy().squeeze())
    #plt.plot(test_y_hat,'C2')
    #plt.savefig(path_exp+name_exp+"_NN.png")

    #mc = 0
    #Res=pd.DataFrame(columns=['Dyno_fit_index_tr','Dyno_rmse_tr','Dyno_fit_index_te','Dyno_rmse_te',
    #                             'GP_fit_index_tr','GP_rmse_tr','GP_fit_index_te','GP_rmse_te'])
    #Res.loc[mc,'Dyno_fit_index_tr']=None #nonlinear_benchmarks.error_metrics.fit_index(test_y.numpy().squeeze()[0:shiftp], test_y_hat[0:shiftp])
    #Res.loc[mc,'Dyno_rmse_tr']=nonlinear_benchmarks.error_metrics.RMSE(test_y.numpy().squeeze()[0:shiftp], test_y_hat[0:shiftp])
    #Res.loc[mc,'Dyno_fit_index_te']=None #nonlinear_benchmarks.error_metrics.fit_index(test_y.numpy().squeeze()[shiftp:], test_y_hat[shiftp:])
    #Res.loc[mc,'Dyno_rmse_te']=nonlinear_benchmarks.error_metrics.RMSE(test_y.numpy().squeeze()[shiftp:], test_y_hat[shiftp:])
    print("finished NN")
    
    number_in=number_in_b
    number_out=1
     
    batch_size = int(np.minimum(N_train,batch_size))
    
    if (GP_custom!=[])&(typeD!="custom"):
        raise ValueError("Are you runnning a custom dynoGP  model?")

    
    if typeD=="WH":
        DeeGPmodel = DeepGP_Wiener_Hammerstein
    elif typeD=="Wiener":
        DeeGPmodel = DeepGP_Wiener
    elif typeD=="HW":
        DeeGPmodel = DeepGP_Hammerstein_Wiener
    elif typeD=="LTI":
        DeeGPmodel=DeepGP_LTI_only
    elif typeD=="custom":
        DeeGPmodel=GP_custom
        

           
        
        
        
    model=DeeGPmodel(number_in,number_out, time_tr.to(device),deltat,batch_size=batch_size,
                     num_states_dynlayer=n_a,
                     #number_in_l_dynlayer=[1]*len(n_a),
                     num_inducing_dynlayer=num_inducing_dynlayer,   #800                      
                     #parametrisation_dynlayer=['jordan'],
                     num_inducing_nonlin=num_inducing_nonlin,
                     strategy_inducing_init_dynamic=strategy_inducing_init_dynamic,
                     initialization_type=initialization_type) #200
    
    predictive_means,predictive_variances= train_gp(model.to(device),train_x.to(device),train_y.to(device),test_x.to(device),training_iter=itertr_GP,batch_size=batch_size,
                                                    learning_rate = learning_rate_GP, num_samples_likelihood_tr=num_samples_likelihood_tr)
    predictive_means = predictive_means[...,0,:]
    predictive_variances = predictive_variances[...,0,:]
    nsamples = predictive_means.shape[1]
    for i in range(nsamples):
        predictive_means[:,i,:] = yscaler.inverse_transform(torch.tensor(predictive_means[:,i,:])).numpy()
        predictive_variances[:,i,:] = ((yscaler.std**2)*torch.tensor(predictive_variances[:,i,:])).numpy()
    #test_y_hat_GP = np.median(predictive_means,axis=0).flatten()
    
    #test_y_hat = test_y_hat_GP.copy()
    print("finished GP")
    #Res.loc[mc,'GP_fit_index_tr']=None #nonlinear_benchmarks.error_metrics.fit_index(test_y.numpy().squeeze()[0:shiftp], test_y_hat[0:shiftp])
    #Res.loc[mc,'GP_rmse_tr']=nonlinear_benchmarks.error_metrics.RMSE(test_y.numpy().squeeze()[0:shiftp], test_y_hat[0:shiftp])
    #Res.loc[mc,'GP_fit_index_te']=None# nonlinear_benchmarks.error_metrics.fit_index(test_y.numpy().squeeze()[shiftp:], test_y_hat[shiftp:])
    #Res.loc[mc,'GP_rmse_te']=nonlinear_benchmarks.error_metrics.RMSE(test_y.numpy().squeeze()[shiftp:], test_y_hat[shiftp:])
    #meanr = Res.mean()
    #medianr=Res.median()
    #Res.loc['mean'] = np.round(meanr,4)
    #Res.loc['median'] = np.round(medianr,4)
    #Res.to_csv(path_exp+name_exp+"_results.csv")
    #plt.figure()
    #plt.plot(test_y.numpy().squeeze())
    #plt.plot(test_y_hat)
    #plt.savefig(path_exp+name_exp+"_GP.png")
    #pickle.dump((shiftp,test_y.numpy().squeeze(),test_y_hat_NN,test_y_hat_GP),open(path_exp+name_exp+"_data_predictions","wb"))
    
    

    with open(path_exp+name_exp+"gp_means", 'wb') as handle:
        pickle.dump(predictive_means, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(path_exp+name_exp+"gp_vars", 'wb') as handle:
        pickle.dump(predictive_variances, handle, protocol=pickle.HIGHEST_PROTOCOL)
