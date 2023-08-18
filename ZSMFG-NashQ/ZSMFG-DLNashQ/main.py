from base import Actor,Critic
from nashDQN import NashDQN
from myEnv import my1dGridEnv
from tools import ReplayBuffer
import torch 
import torch.nn as nn

env = my1dGridEnv()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


if __name__ == "__main__":
    # Run Nash DQN
    print(device)
    Players = NashDQN(lr=1e-2,gamma=0.8,batch_size=50,
                    env=env,replay_buffer=ReplayBuffer,max_iteration=1000,state_dim = 4,epsilon=0.5,n_steps_ctrl=10,max_episode=100)
    Players.training()