from cmath import tanh
import torch 
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


# Actor network used to approximate pi
class Actor(nn.Module):
    
    def __init__(self,n_states,n_actions,hidden_dim,max_action=1):
        super(Actor,self).__init__()
        self.max_action = max_action
        self.layers = nn.Sequential(
            nn.Linear(n_states,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,n_actions),
            nn.Tanh()
        )

    def forward(self,state):
        # scaling up in range (-1,1)
        action = self.max_action*self.layers(state)

        return action




# Critic network approximates Q(s,a_1,a_2)
class Critic(nn.Module):

    def __init__(self,n_states,n_actions,hidden_dim):
        super(Critic,self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_states+n_actions+n_actions,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,1)
        )

    def forward(self,state,action_1,action_2):
        input = torch.cat([state+action_1+action_2],1)
        Q_value = self.layers(input)

        return Q_value

# Same like Critic but prepare for DQN
class ValueNet(nn.Module):

    def __init__(self,n_states,n_actions):
        super(ValueNet,self).__init__()
        self.state_layers = nn.Sequential(
            nn.Linear(n_states,128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,32)
        )

        self.action_layers = nn.Sequential(
            nn.Linear(n_actions*n_states,64),
            nn.ReLU(),
            nn.Linear(64,128),
            nn.ReLU(),
            nn.Linear(128,32)
        )
        self.final_layer = nn.Sequential(
            nn.Linear(96,48),
            nn.ReLU(),
            nn.Linear(48,1)
        )

    def forward(self,state,action_1,action_2):
        state_vec = self.state_layers(state)
        #print(action_1.flatten().shape)
        action1_vec = self.action_layers(action_1.flatten(start_dim=1))
        action2_vec = self.action_layers(action_2.flatten(start_dim=1))
        # print(state_vec.shape)
        # print(action1_vec.shape)
        # print(action2_vec.shape)
        input = torch.cat([state_vec,action1_vec,action2_vec],axis=1)
        #print(input.shape)
        Q_value = self.final_layer(input)

        return Q_value