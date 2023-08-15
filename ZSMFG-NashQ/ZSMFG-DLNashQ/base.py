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

# Same like Critic but preprare for DQN
class ValueNet(nn.Module):

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