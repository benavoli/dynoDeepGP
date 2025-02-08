#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
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

from functions.scaler import StandardScaler
import scoringrules as sr
import nonlinear_benchmarks # used just for metrics
import pandas as pd
from model.gp_train_batch import train_gp
from model.models import DeepGP_Wiener, DeepGP_LTI_only
from dynonetloc.dynonet_train import train_dynonet

# In[3]:



device = torch.device("cuda")


# # Generate data from a linear system followed by a static nonlinearity

# In[5]:
N = 2500  # total number of data points
N_train = 1500  #
deltat = 1/100.0 # sampling rate

number_in_b = 2  # number of inputs - they are shifted cosine functions
num_states = 5
number_in_l = 1  # number of columns of the matrix L which defines the covariance of the noise as LL^T

# add measreument noise
noisestd = 0.1 #0.025

# static nonlinearity
def nonlinear(x):
    return x*(x>0) #x**2# x*abs(torch.sigmoid(x)-0.5)#x**2 #(x>0)*x  #x**2  # torch.sigmoid(x)

#deterministic = False  # if this is set to True then the linear system is deterministic
architecture =  'jordan' #sys.argv[1] # 'jordan'

for deterministic in [False]:
    Data=[]
    Dyno=[]
    GP_means=[]
    GP_vars=[]
    Res=pd.DataFrame(columns=['Dyno_fit_index_tr','Dyno_rmse_tr','Dyno_fit_index_te','Dyno_rmse_te',
                              'GP_fit_index_tr','GP_rmse_tr','GP_fit_index_te','GP_rmse_te'])
    for mc in range(0,50): #(3,50)
        print("mc=",mc)
        torch.manual_seed(mc)    
        
              

        t, u, y, parameters = wiener(
            deltat,
            N,
            N_train,
            num_states=num_states,
            number_in_b=number_in_b,
            number_in_l=1,
            jitter=1e-8,
            nonlinear=nonlinear,
            deterministic=deterministic,
        )  # torch.exp)
        
        # normalise data for fitting
        #ym = y.mean()
        #ys = y.std()
        #y = (y - ym) / ys
        #um = u.mean(axis=1)
        #us = u.std(axis=1)
        #u = (u - um) / us
        
        ys = y.std()
        
        #add noise
        y = y + noisestd*ys* torch.randn(*y.shape)
        y = y[None,:]

        
        time_tr = t[0:N_train, :]
        time_te = t
        
        uscaler=StandardScaler()
        uscaler.fit(u[:, 0:N_train, :],dim=1)
        #uscaler.transform(u[:, 0:N_train, :])
        train_x_orig = torch.cat(
            [time_tr.expand(u.shape[0],*time_tr.shape), u[:, 0:N_train, :]], dim=2
        )
        train_x = torch.cat(
            [time_tr.expand(u.shape[0],*time_tr.shape), uscaler.transform(u[:, 0:N_train, :])], dim=2
        )  # the input is the stacking of time and u [t,u]
        yscaler = StandardScaler()
        yscaler.fit(y[:,0:N_train],dim=1)
        train_y_orig=y[:,0:N_train]
        train_y = yscaler.transform(y[:,0:N_train])
        train_u = uscaler.transform(u[:, :N_train, :])
        
        # we test it on all the data
        test_x_origin = torch.cat([time_te.expand(u.shape[0],*time_te.shape), u], dim=2)
        test_x = torch.cat([time_te.expand(u.shape[0],*time_te.shape), uscaler.transform(u)], dim=2)
        test_y_orig= y
        test_y = yscaler.transform(y)
        
        
        shiftp0 = 1000
        shiftp = np.minimum(shiftp0,N_train)
        test_x = test_x[:,N_train-shiftp:,...]
        test_x_orig = test_x_origin[:,N_train-shiftp:,...]
        test_y = test_y[:,N_train-shiftp:,...]
        test_y_orig = test_y_orig[:,N_train-shiftp:,...]
        test_u = uscaler.transform(u[:,N_train-shiftp:, ...])
        
        
        
        Data.append([train_x_orig,train_y_orig,test_x_orig,test_y_orig])
        
        
        n_input=number_in_b
        n_output=1
        n_a=[5]#10
        n_b=[5]#10
        n_hidden=[32]
        pred_NN = train_dynonet(n_input,n_output,n_a,n_b,n_hidden,train_y,test_y,train_u,test_u,itertr=5000,typeD="Wiener")#10000
        test_y_hat = yscaler.inverse_transform( torch.tensor(pred_NN).squeeze()).numpy()
        test_y_hat=test_y_hat[0]
        
        Dyno.append(test_y_hat)
        deterministi = str(deterministic)
        np.savetxt("../results/Wiener/"+architecture+"/dyno_deterministic_"+deterministi+".csv", Dyno, delimiter=",")
        with open("../results/Wiener/"+architecture+"/data_deterministic_"+deterministi, 'wb') as handle:
            pickle.dump(Data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        '''
        
        Res.loc[mc,'Dyno_fit_index_tr']=nonlinear_benchmarks.error_metrics.fit_index(test_y_orig[0].numpy().squeeze()[0:shiftp], test_y_hat[0:shiftp])
        Res.loc[mc,'Dyno_rmse_tr']=nonlinear_benchmarks.error_metrics.RMSE(test_y_orig[0].numpy().squeeze()[0:shiftp], test_y_hat[0:shiftp])
        Res.loc[mc,'Dyno_fit_index_te']=nonlinear_benchmarks.error_metrics.fit_index(test_y_orig[0].numpy().squeeze()[shiftp:], test_y_hat[shiftp:])
        Res.loc[mc,'Dyno_rmse_te']=nonlinear_benchmarks.error_metrics.RMSE(test_y_orig[0].numpy().squeeze()[shiftp:], test_y_hat[shiftp:])
        #Res.loc[mc,'Dyno_fit_index_te']=nonlinear_benchmarks.error_metrics.fit_index(test_y.numpy().squeeze(), test_y_hat)
        #es.loc[mc,'Dyno_rmse_te']=nonlinear_benchmarks.error_metrics.RMSE(test_y.numpy().squeeze(), test_y_hat)
        print("finished NN")
        number_in=number_in_b
        number_out=1
         
        batch_size = int(np.minimum(N_train,375))

        model=DeepGP_Wiener(number_in,number_out, time_tr.to(device),deltat,batch_size=batch_size,
                         num_states_dynlayer=[5],#10
                         number_in_l_dynlayer=[1],
                         num_inducing_dynlayer=[800],   #800                      
                         parametrisation_dynlayer=[architecture],
                         num_inducing_nonlin=[200],
                             initialization_type="normal"
                         ) #200

        predictive_means,predictive_variances= train_gp(model.to(device),train_x.to(device),train_y.to(device),test_x.to(device),
                                                        training_iter=2000,batch_size=batch_size,learning_rate=0.002)#900
        predictive_means = predictive_means[...,0,:]
        predictive_variances = predictive_variances[...,0,:]
        nsamples = predictive_means.shape[1]
        for i in range(nsamples):
            predictive_means[:,i,:] = yscaler.inverse_transform(torch.tensor(predictive_means[:,i,:])).numpy()
            predictive_variances[:,i,:] = ((yscaler.std**2)*torch.tensor(predictive_variances[:,i,:])).numpy()
        
        test_y_hat = np.median(predictive_means[0],axis=0).flatten()
        #mus = predictive_means
        #sigmas = np.sqrt(predictive_variances)
        #ytestf = test_y.flatten()
        #crps=[sr.crps_mixnorm(ytestf[i],mus[:,i],sigmas[:,i],np.ones(mus.shape[0])/mus.shape[0]) for i in range(len(ytestf))]


        print("finished GP")
        GP_means.append(predictive_means)
        GP_vars.append(predictive_variances)
        Res.loc[mc,'GP_fit_index_tr']=nonlinear_benchmarks.error_metrics.fit_index(test_y_orig[0].numpy().squeeze()[0:shiftp], test_y_hat[0:shiftp])
        Res.loc[mc,'GP_rmse_tr']=nonlinear_benchmarks.error_metrics.RMSE(test_y_orig[0].numpy().squeeze()[0:shiftp], test_y_hat[0:shiftp])
        Res.loc[mc,'GP_fit_index_te']=nonlinear_benchmarks.error_metrics.fit_index(test_y_orig[0].numpy().squeeze()[shiftp:], test_y_hat[shiftp:])
        Res.loc[mc,'GP_rmse_te']=nonlinear_benchmarks.error_metrics.RMSE(test_y_orig[0].numpy().squeeze()[shiftp:], test_y_hat[shiftp:])

        meanr = Res.mean()
        medianr=Res.median()
        Res.loc['mean'] = np.round(meanr,4)
        Res.loc['median'] = np.round(medianr,4)

        Res.to_csv("../results/Wiener/"+architecture+"/metrics_deterministic_"+deterministi+".csv")
        #np.savetxt("../results/Wiener/"+architecture+"/data_deterministic_"+deterministi+".csv", Data, delimiter=",")

        with open("../results/Wiener/"+architecture+"/gp_means_deterministic_"+deterministi, 'wb') as handle:
            pickle.dump(GP_means, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open("../results/Wiener/"+architecture+"/gp_vars_deterministic_"+deterministi, 'wb') as handle:
            pickle.dump(GP_vars, handle, protocol=pickle.HIGHEST_PROTOCOL)
        #np.savetxt("../results/Wiener/"+architecture+"/gp_means_deterministic_"+deterministi+".csv", GP_means, delimiter=",")
        #np.savetxt("../results/Wiener/"+architecture+"/gp_vars_deterministic_"+deterministi+".csv", GP_vars, delimiter=",")
        '''
        gc.collect()
        
        
        #torch.empty_cache()
        #plt.plot(test_y.numpy().squeeze())
        #plt.plot(test_y_hat)
   
