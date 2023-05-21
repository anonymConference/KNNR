# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 14:29:31 2021

@author: Administrator
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
#import multiprocessing as mp
#from sample_backward_ind import sample

np.random.seed(seed=2)


N_traj = 1000000
Nsample_list = [100000]#[100,250,500,750,1000]#[10000]#[2500,5000,7500,10000]#[100,250,500,750,1000]#[50000]#,,100,250,500,750,1000,#[100,250,500,750,1000]#
resample_ratio = 5
n_reps = 30


sub_m = 1
r = 0.0001
decay = 0.99
size_feature = 5
D = size_feature*1
eta_org = 0.001
eta = eta_org
training_steps = 200
pre_factor = 1

T = 20

model_lambda = 50/60 
kappa = 100
alpha = 0.01
N0 = size_feature
S0 = 30
sig = 0.1


get_target  = False
def inner_sum(t,q):
    summand = 0
    for n in range(int(q+1)):
        summand += model_lambda**n*np.exp(-kappa*alpha*(q-n)**2)*(T-t)**n/(np.math.factorial(n)) 
    return summand    


def sol(t,q):
    sol = (1+np.log(inner_sum(t,q)/inner_sum(t,q-1)))/kappa
    
    return sol

def optimal_policy():
    opt_pol = np.ones([N0,T])
    for t in range(T):
        for q in range(N0):
            opt_pol[q,t] = sol(t,q+1)
            
    return opt_pol

def get_action(state,K_samp,t_res, size_feature):
    q_min = int(state[1])
    features_min =  np.zeros(size_feature)#pre_factor*np.array([0, 0, 0, 0, 0])

    features_min[q_min-1] = 1
    if t==0:
        K_min = np.copy(K_samp[:,t_res])
        K_min[q_min-1] = K_samp[q_min-1,t_res]
                               
                #print(K_used)
    else:
        K_min = np.copy(K_samp[:,t_res])
                    
    delta_min = np.matmul(K_min, features_min)#
    return delta_min

def linear_policy(K_opt):
    K_start  = K_opt[:,0]
    K_end  = K_opt[:,-1]
    lin_pol = np.ones([N0,T])
    for t in range(T):
        for q in range(N0):
            lin_pol[q,t] =(1-t/T)* K_start[q]+K_end[q]*t/T
            
    return lin_pol

def constant_policy(K_opt):
    K_start  = K_opt[:,0]
    K_end  = K_opt[:,0]
    lin_pol = np.ones([N0,T])
    for t in range(T):
        for q in range(N0):
            lin_pol[q,t] = K_start[q]*t/T+(1-t/T)*K_end[q]
            
    return lin_pol

def worker_func(random_seed, inital_states, T, nbrs,S0, K,size_feature):
    #print('Started Eval Number: ' + str(random_seed))
    time_horizon = T
    np.random.seed(random_seed)
    rand_ind = np.random.randint(inital_states.shape[0])
    state = inital_states[rand_ind]
    q_init = state[1]
    #action_mask = get_action_mask(state_wo_mask, env_config["bag_capacity"])
    #state = {'action_mask':action_mask, 'real_obs': state_wo_mask}
    total_reward =- S0*q_init

    for t_res in range(time_horizon):
        if state[1]==0:
            break
        else:
            action = get_action(state,K,t_res, size_feature)
            state_match =  np.append(state, action)
            #state_pre = state_wo_mask[:-1]
            
            distances, indices = nbrs.kneighbors(state_match.reshape(1, -1))
    
            rand_NN = np.random.randint(n_neighbors)
            state = next_state_df[indices[0,rand_NN]]
            #state_post = state_wo_mask[:-1]
            #print(reward_df[indices[0,rand_NN]])
            #reward_adj = 0#- get_waste_diff(state_pre, state_post, waste_helper)
            total_reward += reward_df[indices[0,rand_NN]]
            #print('Eval Number ' + str(random_seed)+ ' is at t='+ str(t_res) +'with total reward=' + str(total_reward))
            # action_mask = get_action_mask(state_wo_mask, env_config["bag_capacity"])
            # state = {'action_mask':action_mask, 'real_obs': state_wo_mask}
            
    total_reward +=  state[1]*(state[0]-alpha*state[1])
    # if random_seed%250 ==0:
    #     print('Reward from Eval Number: ' + str(random_seed))
    #     print(total_reward)
    
    return -total_reward


# def _parallel_mc(iter=1000):
#     pool = mp.Pool(4)

#     future_res = [pool.apply_async(sample(K)) for _ in range(iter)]
#     res = [f.get() for f in future_res]

#     return res

# target  = -0.0204845



if __name__ == '__main__':


    

    K_opt = optimal_policy()#+ np.random.randn(size_feature,T)*0.001#
    K_constant = 0.02* np.ones([N0,T])#constant_policy(K_opt)/2
    #K_opt = K_opt[:,1:] 
    #N_traj = 500000
    #pool = mp.Pool(4)
    
    if get_target:
        K_save = 0
    
        t_out = 0
        
        loss_list = []
        #print('Current Period: '+ str(t_out))
    
    
    
        q_counter  = np.zeros([N0])
        #loss = 0
        K_samp = K_opt#0.02* np.ones([N0,T])#constant_policy(K_opt)/2#linear_policy(K_opt)#optimal_policy()
        for n in range(N_traj):  
    
    
            # K_min = K-r
            # loss_min = []
        
            # K_max = K+r
            # loss_max = []
            # rand_perturb = np.random.randn(size_feature)
            # rand_perturb = r*rand_perturb/ np.linalg.norm(rand_perturb)
        
        

            #(np.random.rand(size_feature,T))*0.04#optimal_policy() +
            W = np.random.randn(T-t_out)
            M = np.random.poisson(lam=1.0, size=T-t_out)
            #q = N0
            q_init = 5#np.random.randint(1, high = N0+1)
            q_min = q_init
       
            features_min =  np.zeros(size_feature)#pre_factor*np.array([0, 0, 0, 0, 0])
    
            features_min[q_min-1] = 1
    
                
            St = S0
            # if t_out ==0:
            #     t_start = 0
            # else:
            #     t_start = np.random.randint(0, high = t_out+1)
            #     St = St + sig*np.sum(W[:t_start-1])
           
            X_min = 0
    
            
            q_counter[q_init-1] += 1
            
            for t in range(T-t_out):

                    
                St = St + sig*W[t]


                reward = 0
                if q_min>0:
                    
                    if t==0:
                        K_min = np.copy(K_samp[:,t])
                        K_min[q_min-1] = K_samp[q_min-1,t]
                    
                    
                    #print(K_used)
                    else:
                        K_min = np.copy(K_samp[:,t])
                        
                    delta_min = np.matmul(K_min, features_min)#sol(t,q)#0.1/(1 + np.exp(-np.matmul(K_used, features)))## 

                    
                    if M[t]>0:
                        #delta_min = np.matmul(K_min, features_min)#sol(t,q)#0.1/(1 + np.exp(-np.matmul(K_used, features)))## 
                        #action_list.append(delta_min)
                        
                        prob_min = np.exp(-kappa*delta_min)
        
                        if prob_min >=1:
                            delta_N_min = min(M[t],q_min)
                        else:
                            delta_N_min = min(np.random.binomial(M[t], prob_min),q_min)
                    
         
                        ind_pre_min = int(q_min-1)
        
                        q_min  = q_min - delta_N_min
                        ind_post_min = max(int(q_min-1),0)
                        
                        features_min[ind_pre_min ] = 0
                        features_min[ind_post_min] = pre_factor*1
        
                        # print(ind_pre)
                        # print(ind_post)
                        # print(q)
                        # print(features)
                        reward = (St + delta_min)*delta_N_min
                        X_min += reward
                        
                
          

                elif q_min == 0:

                    delta_min = 0

                    break
                

            loss_min  = -(X_min + q_min*(St-alpha*q_min)-S0*q_init)
    
            
    
            # if not(loss_diff==0) and q_init == 4:
            #     print(loss_diff)
            loss_list.append(loss_min)
            #loss += loss_min
            
            #loss.append(-( X + q*(St-alpha*q)-S0*N0))
            #print(-( X + features[1]*N0*(St-alpha*features[1]*N0)))
          
        target_loss_np = np.array(loss_list)
        target_loss = np.mean(target_loss_np)
        print('The average target loss is ')
        print(target_loss)
        
        
    

    reward_stored = np.zeros([n_reps,len(Nsample_list)])
    N_ind = 0

    for Nsample in Nsample_list:
        print(Nsample)
        Nresample = int(Nsample/resample_ratio)
        for i_rep in range(n_reps):
            K_save = 0
    
            t_out = 0
        
            feature_pre_list = []
            feature_post_list = []
            action_list = []
            reward_list = []
            start_list = []
            end_list = []
            #print('Current Period: '+ str(t_out))
        
        
        
            q_counter  = np.zeros([N0])
            loss = 0
            rel_error_list = []
        
            for n in range(Nsample):
                
        
        
                # K_min = K-r
                # loss_min = []
            
                # K_max = K+r
                # loss_max = []
                # rand_perturb = np.random.randn(size_feature)
                # rand_perturb = r*rand_perturb/ np.linalg.norm(rand_perturb)
            
            
                
                match_feature_pre =  np.zeros(2)
                match_feature_post =  np.zeros(2)
                
                K_samp = (np.random.rand(size_feature,T))*0.05-0.01 #+ optimal_policy()
                W = np.random.randn(T-t_out)
                M = np.random.poisson(lam=1.0, size=T-t_out)
                #q = N0
                q_init = 5#np.random.randint(1, high = N0+1)
                q_min = q_init
           
                features_min =  np.zeros(size_feature)#pre_factor*np.array([0, 0, 0, 0, 0])
        
                features_min[q_min-1] = 1
        
                    
                St = S0
                # if t_out ==0:
                #     t_start = 0
                # else:
                #     t_start = np.random.randint(0, high = t_out+1)
                #     St = St + sig*np.sum(W[:t_start-1])
               
                X_min = 0
        
                
                q_counter[q_init-1] += 1
                
                for t in range(T-t_out):
                    if t !=0:
                        #print('in')
                        match_feature_post[0] = St
                        match_feature_post[1] = q_min
                        #print(match_feature_post)
                        feature_post_list.append(np.copy(match_feature_post))
                        start_list.append(0)
                        end_list.append(0) 
                    else:
                        start_list.append(1)
                        
                    St = St + sig*W[t]
                    match_feature_pre[0] = St
                    match_feature_pre[1] = q_min 
                    feature_pre_list.append(np.copy(match_feature_pre))
                    reward = 0
                    if q_min>0:
                        
                        if t==0:
                            K_min = np.copy(K_samp[:,t])
                            K_min[q_min-1] = K_samp[q_min-1,t]
                        
                        
                        #print(K_used)
                        else:
                            K_min = np.copy(K_samp[:,t])
                            
                        delta_min = np.matmul(K_min, features_min)#sol(t,q)#0.1/(1 + np.exp(-np.matmul(K_used, features)))## 
                        action_list.append(delta_min) 
                        
                        if M[t]>0:
                            #delta_min = np.matmul(K_min, features_min)#sol(t,q)#0.1/(1 + np.exp(-np.matmul(K_used, features)))## 
                            #action_list.append(delta_min)
                            
                            prob_min = np.exp(-kappa*delta_min)
            
                            if prob_min >=1:
                                delta_N_min = min(M[t],q_min)
                            else:
                                delta_N_min = min(np.random.binomial(M[t], prob_min),q_min)
                        
             
                            ind_pre_min = int(q_min-1)
            
                            q_min  = q_min - delta_N_min
                            ind_post_min = max(int(q_min-1),0)
                            
                            features_min[ind_pre_min ] = 0
                            features_min[ind_post_min] = pre_factor*1
            
                            # print(ind_pre)
                            # print(ind_post)
                            # print(q)
                            # print(features)
                            reward = (St + delta_min)*delta_N_min
                            X_min += reward
                            
                    
              
                        reward_list.append(reward) 
                    elif q_min == 0:
                        reward_list.append(reward)
                        delta_min = 0
                        action_list.append(delta_min)
                        break
                    
                match_feature_post[0] = -100
                match_feature_post[1] = -100
                feature_post_list.append(np.copy(match_feature_post))
                end_list.append(1)
                loss_min  = -(X_min + q_min*(St-alpha*q_min)-S0*q_init)
        
                
        
                # if not(loss_diff==0) and q_init == 4:
                #     print(loss_diff)
                
                loss += loss_min
                
                #loss.append(-( X + q*(St-alpha*q)-S0*N0))
                #print(-( X + features[1]*N0*(St-alpha*features[1]*N0)))
               
            poly = PolynomialFeatures(degree=2)
            
            feature_pre_np  =  np.array(feature_pre_list)
            feature_post_np = np.array(feature_post_list) 
            reward_np = np.array(reward_list)
            end_np = np.array(end_list)
            
            start_np = np.array(start_list)
            initial_states =  feature_pre_np[start_np ==1,:]
             
            match_df = feature_pre_np[end_np!=1,:]
            next_state_df = feature_post_np[end_np!=1,:]
            reward_df = reward_np[end_np!=1]
            
            action_np =  np.array([action_list])
            action_df = action_np[:,end_np!=1]
            action_modified = np.zeros(action_df.shape[1])
            
            match_df  =  np.concatenate((match_df, action_df.T), axis=1)
            features =poly.fit_transform( np.concatenate((match_df[:,:2], action_df.T), axis=1))
            
            
            
            Q_target = -next_state_df[:,1]*(next_state_df[:,0]-alpha*next_state_df[:,1])
            for t in range(T,0,-1):
                targets = -reward_df + Q_target
                reg = LinearRegression().fit(features , targets)
                for i_q in range(q_init):
                    action_modified[next_state_df[:,1]==i_q+1] = K_opt[i_q,t-1]
                
                action_modified = action_modified.reshape(action_df.shape).T    
                features_modified = poly.fit_transform(np.concatenate((next_state_df , action_modified), axis=1 ) )  
                #GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0).fit(features, targets)
                Q_target = reg.predict(features_modified)
                Q_target[next_state_df[:,1]==0] = 0
            
            action_init = K_opt[q_init-1,0]*np.ones([initial_states.shape[0],1])        
            
            features_init =poly.fit_transform( np.concatenate((initial_states,action_init), axis=1 ))
            
            V = reg.predict(features_init )+ initial_states[:,0]*q_init
            print(np.mean( V ))
            reward_stored[i_rep, N_ind] = np.mean( V )
         
            # print(loss)
            # print(np.mean(resampled_loss))
            if i_rep%10 == 0:
                print(i_rep)
        # rel_error = np.abs(reward_est-target_reward)/np.abs(target_reward)
        # rel_error_list.append(rel_error)      
        # print(np.abs(reward_est-target_reward)/np.abs(target_reward))         
        N_ind+=1 