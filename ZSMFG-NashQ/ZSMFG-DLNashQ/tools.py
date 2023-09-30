import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class ReplayBuffer(object):

    def __init__(self,n_states,n_actions,n_players=2):
        self.n_states = n_states
        self.n_actions = n_actions
        self.max_storage = int(1e3)
        self.count = 0 # count order as a index
        self.size = 0
        self.state_storage = np.zeros((self.max_storage,n_players,self.n_states)).astype(np.float32)
        self.action_storage = np.zeros((self.max_storage,n_players,self.n_states,self.n_actions)).astype(np.float32)
        self.reward_storage = np.zeros((self.max_storage,n_players)).astype(np.float32)
        self.state_next_storage = np.zeros((self.max_storage,n_players,self.n_states)).astype(np.float32)
            

        

    def store(self,state,action,reward,new_state):
        # print(len(state))
        # print(self.state_storage[self.count].shape)
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


def recover_equilibriums(Players,model_1,model_2,max_steps):
    Players.Q_1 = Players.Q_1.load_states(model_1)
    Players.Q_2 = Players.Q_2.load_states(model_2)

    mf_states = [[1,0,0],[0,0,1]]
    for i in range(max_steps):



        return

def solve_cont_stage_game():
    return 