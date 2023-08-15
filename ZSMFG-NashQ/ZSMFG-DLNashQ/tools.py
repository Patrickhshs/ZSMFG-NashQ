import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class ReplayBuffer(object):

    def __init__(self,n_states,n_actions,n_players=2):
        self.n_states = n_states
        self.n_actions = n_actions
        self.max_storage = int(1e6)
        self.count = 0 # count order as a index
        self.size = 0
        if n_players ==2:
            self.state_storage = np.zeros((self.max_storage,self.n_states,2))
            self.action_storage = np.zeros((self.max_storage,self.n_actions,2))
            self.reward_storage = np.zeros((self.max_storage,1,2))
            self.state_next_storage = np.zeros((self.max_storage,self.n_states,2))
        else:
            self.state_storage = np.zeros((self.max_storage,self.n_states))
            self.action_storage = np.zeros((self.max_storage,self.n_actions))
            self.reward_storage = np.zeros((self.max_storage,1))
            self.state_next_storage = np.zeros((self.max_storage,self.n_states))

        

    def store(self,state,action,reward,new_state):
        self.state_storage[self.count] = state
        self.action_storage[self.count] = action
        self.reward_storage[self.count] = reward
        self.state_next_storage[self.count] = new_state
        self.count = (self.count + 1) % self.max_storage # update the memory if out of storage
        self.size = min(self.size + 1, self.max_storage) # keep record

    
    def sample(self,batch_size):
        index = np.random.choice(self.size,size = batch_size)
        batch_state = torch.tensor(self.state_storage[index],dtype=torch.float)
        batch_action = torch.tensor(self.action_storage[index], dtype = torch.float)
        batch_reward = torch.tensor(self.reward_storage[index],dtype=torch.float)
        batch_state_next = torch.tensor(self.state_next_storage[index], dtype = torch.float)

        return batch_state,batch_action,batch_reward,batch_state_next




def solve_cont_stage_game():
    return 