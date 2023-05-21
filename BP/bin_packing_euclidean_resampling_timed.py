# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 12:27:08 2022

@author: Administrator
"""
#####~
# TO DO : check why state can become negative!
######


from bin_environment_org import BinPackingActionMaskGymEnvironment
from env_utils import get_action_mask, step_transform
import numpy as np
#import matplotlib.pyplot as plt
from scipy.stats import poisson, uniform
from sklearn.neighbors import NearestNeighbors, DistanceMetric
#import ot
#from joblib import parallel_backend
#import cython_dist  as cd
#from multiprocessing import Pool
from functools import partial
import warnings
import time as tm
warnings.filterwarnings("ignore")


np.random.seed(2423)

def get_action(state):
    state = state['real_obs']
    num_bins_level = state[:-1]
    item_size = state[-1]
    bag_capacity = len(state)-1

    if item_size == bag_capacity:
        return 0 # new bag

    min_difference = bag_capacity 
    chosen_bin_index = 0 #default is new bag
    for i, bins in enumerate(num_bins_level):
        #skip new bag and levels for which bins don't exist
        if bins == 0 or i == 0:
            continue

        #if item fits perfectly into the bag
        elif (i + item_size) == bag_capacity:
            # assuming full bins have count 0
            if -bins < min_difference:
                min_difference = -bins
                chosen_bin_index = i
                return chosen_bin_index
            else:
                continue
        #item should fit in bag and should be at least of size 1
        elif (i + item_size) > bag_capacity:
            continue

        #sum of squares difference that chooses the bin 
        if num_bins_level[i + item_size] - bins < min_difference:
            chosen_bin_index = i 
            min_difference = num_bins_level[i + item_size] - bins 

    return chosen_bin_index


def get_action_eps(state,eps):
    action_mask = state['action_mask']
    state = state['real_obs']
    num_bins_level = state[:-1]
    item_size = state[-1]
    bag_capacity = len(state)-1
    rand = np.random.binomial(1, 1-eps)
    
    if rand ==1:
        if item_size == bag_capacity:
            return 0 # new bag
    
        min_difference = bag_capacity 
        chosen_bin_index = 0 #default is new bag
        for i, bins in enumerate(num_bins_level):
            #skip new bag and levels for which bins don't exist
            if bins == 0 or i == 0:
                continue
    
            #if item fits perfectly into the bag
            elif (i + item_size) == bag_capacity:
                # assuming full bins have count 0
                if -bins < min_difference:
                    min_difference = -bins
                    chosen_bin_index = i
                    return chosen_bin_index
                else:
                    continue
            #item should fit in bag and should be at least of size 1
            elif (i + item_size) > bag_capacity:
                continue
    
            #sum of squares difference that chooses the bin 
            if num_bins_level[i + item_size] - bins < min_difference:
                chosen_bin_index = i 
                min_difference = num_bins_level[i + item_size] - bins 
    else:
        possible_actions = np.where( action_mask == 1)[0]
        if len(possible_actions)>1:
            eps2 = 1.0
            rand2 = np.random.binomial(1,1-eps2)
            if rand2 ==1:
                chosen_bin_index = np.random.choice(possible_actions[1:])
            else:
                chosen_bin_index = np.random.choice(possible_actions)
        else:
            chosen_bin_index = 0 

    return chosen_bin_index

# def step_transform(state, action, env_config):
#     state_bins = state['real_obs']
#     num_bins_levels = state_bins[:-1]
#     item_size = state_bins[-1]
#     if action==0:
#         num_bins_levels[item_size-1] += 1
#     else:
#         if not(env_config["bag_capacity"] == item_size + action):
#             num_bins_levels[action + item_size] += 1
#         num_bins_levels[action] -= 1
#     return num_bins_levels
    
def get_waste_diff(state_pre, state_post, waste_helper):
    waste_pre = np.sum(waste_helper*state_pre)
    waste_post = np.sum(waste_helper*state_post)
    return waste_post-waste_pre
    

def worker_func(random_seed, inital_states, env_config, nbrs, waste_helper):
    # print('Started Eval Number: ' + str(random_seed))
    time_horizon = env_config['time_horizon']
    #np.random.seed(random_seed)
    rand_ind = np.random.randint(inital_states.shape[0])
    state_wo_mask = inital_states[rand_ind]

    action_mask = get_action_mask(state_wo_mask, env_config["bag_capacity"])
    state = {'action_mask':action_mask, 'real_obs': state_wo_mask} 
    total_reward = 0

    for t_res in range(time_horizon):
        action = get_action(state)
        state_match =  np.append(state_wo_mask,action).astype('float64')
        #state_pre = state_wo_mask[:-1]
        
        distances, indices = nbrs.kneighbors(state_match.reshape(1, -1))

        rand_NN = np.random.randint(n_neighbors)
        state_wo_mask = next_state_df[indices[0,rand_NN]]
        #state_post = state_wo_mask[:-1]
        #print(reward_df[indices[0,rand_NN]])
        reward_adj = 0#- get_waste_diff(state_pre, state_post, waste_helper)
        total_reward += reward_df[indices[0,rand_NN]]+reward_adj 
        #print('Eval Number ' + str(random_seed)+ ' is at t='+ str(t_res) +'with total reward=' + str(total_reward))
        action_mask = get_action_mask(state_wo_mask, env_config["bag_capacity"])
        state = {'action_mask':action_mask, 'real_obs': state_wo_mask} 
        
    # print('Reward from Eval Number: ' + str(random_seed))
    # print(total_reward)
    return total_reward

def get_action_best_fit(state):
    state = state['real_obs']
    num_bins_level = state[:-1]
    item_size = state[-1]
    bag_capacity = len(state)-1

    if item_size == 0:
        print('item size should be larger than 0')
        return 0
    
    if item_size == bag_capacity:
        return 0 # new bag
    
    for i in range(len(num_bins_level)-item_size, 0, -1):
        if num_bins_level[i] > 0: #there is at least one bin at this level
            return i
    
    return 0

def extract_state(state):
    state_bins = state['real_obs']
    num_bins_levels = state_bins[:-1]
    return num_bins_levels,state_bins[-1]

def total_waste(state):
    state_bins = state['real_obs']
    num_bins_levels = state_bins[:-1]
    total_waste = np.sum(num_bins_levels[num_bins_levels!=0]*(20-np.where(num_bins_levels!=0)[0]))
    return total_waste


def calc_lambda(state, env_config):
    state_bins = state['real_obs']
    num_bins_levels = state_bins[:-1]
    if np.sum(num_bins_levels[num_bins_levels!=0])==0:
        lambda_poi = np.max(env_config['item_sizes'])
    else:
        #lambda_poi = np.mean((20-np.where(num_bins_levels!=0)[0]))# Not volume weighted
        weights =  num_bins_levels[num_bins_levels!=0]/np.sum(num_bins_levels[num_bins_levels!=0])
        lambda_poi = np.average((20-np.where(num_bins_levels!=0)[0]),weights=weights )
        lambda_poi = np.min([lambda_poi,9])
        
    return lambda_poi

def get_truncated_poi(lambda_poi,env_config,size=1):
    max_poi = np.max(env_config['item_sizes'])
    cutoff_upper = poisson.cdf(max_poi,lambda_poi) 
    cutoff_lower = poisson.cdf(0,lambda_poi) 
    u = uniform.rvs(scale = (cutoff_upper-cutoff_lower), loc = cutoff_lower, size =size)

    truncated_poi = poisson.ppf(u,lambda_poi)
    return truncated_poi

#def step_transform(state, action, n):
#    num_bins_levels = state[:-1]
#    item_size = int(state[-1])
#    
#    if action==0:
#        num_bins_levels[item_size-1] += 1
#    else:
#        if not(n == item_size + action):
#            num_bins_levels[action + item_size] += 1
#        num_bins_levels[action] -= 1
#    return num_bins_levels


# def mydist(x, y):
#     n = len(x)-2
    
#     action_1 = int(x[-1])
#     action_2 = int(y[-1])
    
#     state_1 = x[:-1]
#     state_2 = y[:-1]
    
#     num_bins_level_1 = step_transform(state_1, action_1, n)
#     num_bins_level_2 = step_transform(state_2, action_2, n)
#     # loss matrix
    
#     dist_dum = np.arange(n, dtype=np.float64)
#     M = ot.dist(dist_dum.reshape((n, 1)), dist_dum.reshape((n, 1)))
#     M /= M.max()
    
    
    
#     dist = ot.unbalanced.mm_unbalanced2(num_bins_level_1, num_bins_level_2,M,0.1,stopThr=1e-3,numItermax =500)
#     return dist
        
env_config = {
                "bag_capacity": 10,
                'item_sizes': [1, 2, 3, 4, 5, 6, 7, 8, 9],
                'item_probabilities': [0.14, 0.10, 0.06, 0.13, 0.11, 0.13, 0.03, 0.11, 0.19], #bounded waste
                #'item_probabilities': [0.06, 0.11, 0.11, 0.22, 0, 0.11, 0.06, 0, 0.33], #perfect pack
                #'item_probabilities': [0, 0, 0, 1/3, 0, 0, 0, 0, 2/3], #linear waste
                'time_horizon': 30,
            }

env = BinPackingActionMaskGymEnvironment(env_config)

Nsample_list = [2500,5000,7500,10000]#[100,250,500,750,1000]#, 25000]#[100]#
eps = 1.0
n_reps = 30
reward_stored = np.zeros([n_reps,len(Nsample_list)])
time_stored = np.zeros([n_reps,len(Nsample_list)])

N_ind = 0
for Nsample in Nsample_list:
    print(Nsample)
    
    
    MC_number = Nsample
    for i_rep in range(n_reps):
        reward_list = []
        pre_state_list = []
        post_state_list = []
        initial_state_list = []
        end_state_list = []
        item_size_list = []
        waste_list = []
        total_reward_list = []
        action_list = []
        # rewards = []
        
        for i in range(MC_number):
            state = env.reset()
            #bins, item_size = extract_state(state)
            #bins_list.append(bins)
            #item_size_list.append(item_size)
            pre_state_list.append(state['real_obs'])
            initial_state_list.append(1)
            #waste_list.append(total_waste(state))
            done = False
            total_reward = 0
            iteration_tracker =0 
            while not done:
                action = get_action_eps(state,eps)# get_action_best_fit(state)#get_action_eps(state,eps)#get_action(state)#
                action_list.append(action)
                # transformed_state_pre = step_transform(state, action, env_config)
                # transformed_states_pre.append(transformed_state)
                state, reward, done, _ = env.step(action)
                pre_state_list.append(state['real_obs'])
                post_state_list.append(state['real_obs'])
                reward_list.append(reward)
                initial_state_list.append(0)
                end_state_list.append(0)
                
                
                total_reward += reward
                iteration_tracker +=1
            
            reward_list.append(float('nan'))
            post_state_list.append(np.ones(len(state['real_obs']), dtype= 'int32'))
            total_reward_list.append(total_reward)
            end_state_list.append(1)
            #action.append(-1)
            #print(len(state['real_obs']))
            #print("Total reward for the sum of squares agent: ", total_reward)
         
          
        waste = np.array(waste_list) 
        
        #counts, bins = np.histogram(waste)
        #plt.hist(bins[:-1], bins, weights=counts)
        total_rewards = np.array(total_reward_list)    
        # print('For eps=' +str(eps) + ' the average total reward is:' )
        # print(np.mean(total_rewards))
        # print(np.std(total_rewards))
        # sum(reward_list) / len(reward_list) 
        
        # print('Start Extract NN structure')    
        pre_state_np = np.array(pre_state_list)
        post_state_np = np.array(post_state_list)
        end_state_np =  np.array(end_state_list)
        reward_np = np.array(reward_list)
        action_np = np.array([action_list])
        initial_state_bool = np.array(initial_state_list)
        
        
        inital_states = pre_state_np[initial_state_bool==1]
        
        match_df = pre_state_np[end_state_np!=1,:]
        next_state_df = post_state_np[end_state_np!=1,:]
        reward_df = reward_np[end_state_np!=1] 
        
        match_df = np.concatenate((match_df , action_np.T), axis=1).astype('float64')
        org_state = match_df[110,:]
        # y  = match_df[500,:]
        n = match_df.shape[1]-2
        dist_dum = np.arange(n, dtype=np.float64)
        # M = ot.dist(dist_dum.reshape((n, 1)), dist_dum.reshape((n, 1)))
            
            
        # M /= np.max(M)
        # reg_m = 1
        # K = np.exp(M / - reg_m / 2)   
        
        # import time
        # rho = 0.1
        # # get the start time
        # keywords = {'M':M, 'rho':rho}
        # #metric = DistanceMetric.get_metric(cd.mydist, **keywords)
        
        # #st = time.time()
        # #
        # #i = 0
        # #while i<100:
        # #dist = cd.mydist(match_df[10,:],match_df[15,:], M, rho)
        # #    i+=1
        # #et = time.time()
        
        # # get the execution time
        # #elapsed_time = et - st
        # #print('Execution time:', elapsed_time, 'seconds')    
        n_neighbors= 5#int(Nsample**0.25)#
        # #
        # #
        # #
        # st = time.time()
        start_time = tm.time()
        nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(match_df )
        
        # et = time.time()
        
        # # get the execution time
        # elapsed_time = et - st
        # print('Execution time Cython:', elapsed_time, 'seconds') 
        
        
        #st = time.time()
        #
        #nbrs = NearestNeighbors(n_neighbors=n_neighbors,  algorithm='ball_tree',  metric=mydist,n_jobs=-1).fit(match_df )
        #
        #et = time.time()
        #
        ## get the execution time
        #elapsed_time = et - st
        #print('Execution time Python:', elapsed_time, 'seconds')  
        #
        #
        
        
        
        #
        #
        #org_state = match_df[2030,:]
        #distances, indices = nbrs.kneighbors(org_state.reshape(1, -1))
        ##
        #
        # print('Start Resampling') 
        
        num_resample = int(MC_number/10)
        
        waste_helper = env_config['bag_capacity']-np.array(range(env_config['bag_capacity']))
        
        #partial_work_func = partial(worker_func,inital_states=inital_states, env_config=env_config, nbrs=nbrs, waste_helper = waste_helper)
         
        resampled_reward_list=[]
        rand_ind = np.random.randint(inital_states.shape[0],size=num_resample)
        time_horizon = env_config['time_horizon']
        rand_NN = np.random.randint(n_neighbors,size=[num_resample,time_horizon])
        
        
        
        for i in range(num_resample):
            
            # print('Started Eval Number: ' + str(random_seed))
            
            #np.random.seed(random_seed)
            
            state_wo_mask = inital_states[rand_ind[i]]
        
            action_mask = get_action_mask(state_wo_mask, env_config["bag_capacity"])
            state = {'action_mask':action_mask, 'real_obs': state_wo_mask} 
            total_reward = 0
            
            
        
            for t_res in range(time_horizon):
                action = get_action(state)
                state_match =  np.append(state_wo_mask,action).astype('float64')
                #state_pre = state_wo_mask[:-1]
                
                distances, indices = nbrs.kneighbors(state_match.reshape(1, -1))
        
                
                state_wo_mask = next_state_df[indices[0,rand_NN[i,t_res]]]
                #state_post = state_wo_mask[:-1]
                #print(reward_df[indices[0,rand_NN]])
                reward_adj = 0#- get_waste_diff(state_pre, state_post, waste_helper)
                total_reward += reward_df[indices[0,rand_NN[i,t_res]]]+reward_adj 
                #print('Eval Number ' + str(random_seed)+ ' is at t='+ str(t_res) +'with total reward=' + str(total_reward))
                action_mask = get_action_mask(state_wo_mask, env_config["bag_capacity"])
                state = {'action_mask':action_mask, 'real_obs': state_wo_mask} 
                    
                    
            reward_resampled = total_reward
            resampled_reward_list.append(np.copy(reward_resampled))
        
        
        resampled_reward_estimate =np.mean( np.array(resampled_reward_list))
        
        reward_stored[i_rep, N_ind] = resampled_reward_estimate
        end_time = tm.time()
            
        time_stored[i_rep, N_ind] = end_time-start_time
        print(resampled_reward_estimate)
    N_ind=N_ind+1
    
    
    
