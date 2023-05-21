#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 22:01:13 2023

@author: giegrich
"""

import numpy as np

def __get_valid_actions(state, bag_capacity ):
    #state is num_bins_level + item_size 
    num_bins_levels = state[:-1]
    item_size =  state[-1]
    
    valid_actions = list()
        # get bin levels for which bins exist and item will fit
        
    for x in range(1, bag_capacity):
            if num_bins_levels[x] > 0:
                if x <= (bag_capacity - item_size):
                    valid_actions.append(x)
    valid_actions.append(0)  # open new bag
    return valid_actions
    
def get_action_mask(state, bag_capacity): 
    valid_actions = __get_valid_actions(state, bag_capacity)
    action_mask = [1 if x in valid_actions else 0 for x in range(bag_capacity)]
    return np.array(action_mask)
    
def  step_transform(state, action, bag_capacity):
    
    item_size = int(np.copy(state[-1]))
    num_bins_levels = np.copy(state[:-1])
    action = int(np.copy(action))
    if action==0:
        num_bins_levels[item_size-1] += 1
    else:
        if not(bag_capacity == item_size + action):
            num_bins_levels[action + item_size] += 1
        num_bins_levels[action] -= 1
    return num_bins_levels    