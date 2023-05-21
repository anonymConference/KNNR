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
import time


np.random.seed(seed=1)#First 30 seed=1# 70 after seed=10

Nsample_list = [2500,5000,7500,10000]##[75000]##[100,250,500,750,1000]#[25000]#[50000]#, ,50000,100000] # Number of sample trajectories
Nsample_target = 100000
resample_ratio = 4

get_target = False
sample_data = True

T = 10 # Length of episode
state_dim = 2
control_dim = 1

n_reps = 30

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
time_stored = np.zeros([n_reps,len(Nsample_list)])


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
        
        start_time = time.time()
        
        n_neighbors= int(Nsample**0.25)
        nbrs = NearestNeighbors(n_neighbors=n_neighbors+T, algorithm='ball_tree').fit(df[match_columns])
        #distances, indices = nbrs.kneighbors(df[match_columns])
        Nresample = int(Nsample/resample_ratio)
        reward_est = 0
        sample_paths = np.zeros([Nresample, T+1, state_dim]) 
        actual_control_paths = np.zeros([Nresample, T]) 
        
        start_ind = np.random.randint(0,Nsample-1,size = Nresample)
        NN_ind = np.random.randint(0,n_neighbors-1,size = [Nresample,T])
        
        for i in range(Nresample):
            xt = np.array([starting_df.iloc[start_ind[i],0],starting_df.iloc[start_ind[i],1]])
            sample_paths[i,0,:] = xt 
            previous_indices =np.array([])
            for t in range(T):
                u_target  = -K@xt
                org_state = np.concatenate([xt,u_target]).reshape(1, -1)
                
                distances, indices = nbrs.kneighbors(org_state)
                mask = np.in1d(indices,previous_indices,invert=True)
                masked_indices = indices[:,mask] 
                previous_indices = np.append(previous_indices,masked_indices[0,NN_ind[i,t]])
                xt = np.array([df.iloc[masked_indices[0,NN_ind[i,t]],4],df.iloc[masked_indices[0,NN_ind[i,t]],5]])
                
                rt = df.iloc[indices[0,NN_ind[i,t]],3]
                sample_paths[i,t+1,:] = xt 
                actual_control_paths[i,t] = df.iloc[indices[0,NN_ind[i,t]],2]
                reward_est += rt
        reward_est = reward_est/Nresample
        reward_stored[i_rep, N_ind] = reward_est
        if i_rep%10 == 0:
            print(i_rep)
        end_time = time.time()
        time_stored[i_rep, N_ind] = end_time-start_time
        
        
        # rel_error = np.abs(reward_est-target_reward)/np.abs(target_reward)
        # rel_error_list.append(rel_error)      
        # print(np.abs(reward_est-target_reward)/np.abs(target_reward))
    #total_time = total_time/n_reps
    #time_list.append(total_time) 
    N_ind+=1 