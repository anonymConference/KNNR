# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 10:22:05 2022

@author: Administrator
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from scipy.linalg import solve_discrete_are
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
import time


np.random.seed(seed=1)

Nsample_list = [10000]#[2500,5000,7500,10000]#[100,250,500,750,1000]#[100,250,500,750,1000]#[2500,5000,7500,10000]#, ,50000,100000] # Number of sample trajectories
Nsample_target = 100000
resample_ratio = 4

get_target = False
sample_data = True

T = 10 # Length of episode
state_dim = 2
control_dim = 1

n_reps = 100

# System matrices
A = np.array([[1,1],[0,1]])
B = np.expand_dims(np.array([0,1]),axis=1)#np.extend_dim(np.array([0,1]))

Q = np.array([[1,0],[0,0]])

r0 = 0.5

# create target strategy path



#xsum = 0
# # Solve Riccati equation
# P_end = np.array([[0,0],[0,0]])
# P_list = []
# P_list.append()


P = solve_discrete_are(A, B, Q, r0) 
K = np.linalg.inv(r0+B.T@P@B)@B.T@P@A
# Get Target reward 

if get_target:
    
    cum_reward = np.zeros(Nsample_target)
    
    x_init = np.vstack((-1+0.5*np.random.randn(Nsample_target), np.zeros(Nsample_target))).T#0.1*np.random.randn()
    w = 10e-2*np.random.randn(Nsample_target,T, state_dim)
    
    for i in range (Nsample_target):
        # K_sample1 = np.random.rand(Nsample_target,T)
        # K_sample2 = 1 + np.random.rand(Nsample_target,T)
        
        # K_sample = np.concatenate((K_sample1[...,np.newaxis],K_sample2[...,np.newaxis]),axis=2)
        
        xt = x_init[i,:]
        reward = 0
        for t in range(T):
            # K = K_sample[i,t,:]
            u_sample = -K@xt
            xt_1 = xt
            xt = A@xt + u_sample + w[i,t]
            rt= xt.T@Q@xt+r0*u_sample**2
            reward +=rt  
        cum_reward[i] = reward
    target_reward = np.mean(cum_reward)
    print('The average target award is ')
    print(target_reward)
        


rel_error_list = []

reward_stored = np.zeros([n_reps,len(Nsample_list)])
N_ind = 0
time_list = []


for Nsample in Nsample_list:
    print(Nsample)
    total_time = 0
    
    for i_rep in range(n_reps):
        if sample_data:
            current_state1 = []
            current_state2 = []
            next_state1 = []
            next_state2 = []
            reward = []
            control = []
            time_model = []
            
            # Sampling
            x_init = np.vstack((-1+0.5*np.random.randn(Nsample), np.zeros(Nsample))).T#0.1*np.random.randn()
            
            K_sample1 = np.random.rand(Nsample,T)
            K_sample2 = 1 + np.random.rand(Nsample,T)
            
            K_sample = np.concatenate((K_sample1[...,np.newaxis],K_sample2[...,np.newaxis]),axis=2)
            w = 10e-2*np.random.randn(Nsample,T, state_dim)
            
            for i in range(Nsample):
                xt = x_init[i,:]
            
                for t in range(T):
                    u_sample = -K_sample[i,t,:]@xt
                    xt_1 = xt
                    xt = A@xt + u_sample + w[i,t]
                    rt= xt.T@Q@xt+r0*u_sample**2
                    current_state1.append(xt_1[0])
                    current_state2.append(xt_1[1])
                    next_state1.append(xt[0])
                    next_state2.append(xt[1])
                    control.append(u_sample)
                    time_model.append(t+1)
                    reward.append(rt)
                        
                        
                #xsum += xt
                
            #xavg = xsum/Nsample
            
            df = pd.DataFrame(list(zip(current_state1,current_state2, control,reward, next_state1,next_state2, time_model)),
                           columns =['current_state1','current_state2', 'control', 'reward' ,'next_state1', 'next_state2', 'time' ])
        
        starting_df = df[df['time']==1]  
        match_columns = ['current_state1','current_state2', 'control']
        
        
        poly = PolynomialFeatures(degree=2)
        
        reward = np.array(df['reward'])
        
        Q_target = np.zeros(df['reward'].size)
        features =np.array(df[match_columns])

        features_state = np.array(df[['next_state1','next_state2']])
        actions = np.matmul(features_state, -K.T)
        features_modified = np.concatenate((features_state,actions), axis=1)
        
        n_neighbors= int(Nsample**0.25)
        for t in range(T,0,-1):
            targets = reward + Q_target
            reg = KNeighborsRegressor(n_neighbors=n_neighbors).fit(features, targets)
            #GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0).fit(features, targets)
            Q_target = reg.predict(features_modified)
            
        init_state = np.array(starting_df[['current_state1','current_state2']])   
        actions = np.matmul(init_state , -K.T)
        init_features = np.concatenate((init_state ,actions), axis=1)
        V = reg.predict(init_features)
        reward_stored[i_rep, N_ind] = np.mean(V)
    N_ind +=1 
            
     