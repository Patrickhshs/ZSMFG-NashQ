import os
from statistics import mean
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from base import Actor,Critic
from tools import ReplayBuffer
import copy



class DDPG(object):

    def __init__(self,action_dim,states_dim,max_actions,
                hidden_dim,batch_size,learning_rate,gamma,tau,save_rate=100,id=1):
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.action_dim = action_dim
        self.states_dim  = states_dim
        self.lr = learning_rate
        self.gamma = gamma
        self.tau = tau
        self.max_actions = max_actions
        self.training_step = 0
        self.save_rate = save_rate
        self.id = id
        
        # Actor-Critic networks 
        self.actor = Actor(self.states_dim,self.action_dim,self.hidden_dim,self.max_actions)

        # target network should have the same initialized parameters
        self.actor_target = copy.deepcopy(self.actor)

        self.critic = Critic(self.states_dim,self.action_dim,self.hidden_dim)
        self.critic_target = copy.deepcopy(self.critic)

        # Define optimizers
        self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(),lr=self.lr)
        self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(),lr=self.lr)

    
    # def choose_action(self,state):
    #     state = torch.unsqueeze(torch.tensor(state, dtype=torch.float),0)
    #     action = self.actor(state).data.numpy().flatten()

    #     return action


    def train(self,replay_buffer,other_agent):
        batch_states, batch_action,batch_reward,batch_new_states = replay_buffer.sample(self.batch_size)

        # Compute target Q value

        with torch.no_grad():
            Q_ = self.critic_target(batch_new_states[0],self.actor_target(batch_new_states[0]),other_agent.actor_target(batch_new_states[1]))
            Q_prime = batch_reward + self.gamma * Q_

        current_Q = self.critic(batch_states,batch_action[0],batch_action[1])
        critic_loss = nn.MSELoss(Q_prime, current_Q)

        # add noise

        actor_policy_loss = -torch.mean(self.critic(batch_states[0],self.actor(batch_states[0]),self.actor(batch_states[1]))#*self.actor(batch_states))
                    )
        #actor_policy_loss += torch.mean((self.actor(batch_states)**2))*1e-3

        print('critic_loss is {}, actor_loss is {}'.format(critic_loss, actor_policy_loss))
        self.actor_optimizer.zero_grad()
        actor_policy_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        self.update_target_params()

        if self.training_step > 0 and self.training_step % self.save_rate == 0:
            self.save_model(self.training_step)
        self.training_step +=1


    def update_target_params(self):
        for tar_param,param in zip(self.actor_target.parameters(),self.actor.parameters()):
            tar_param.data.copy_((1-self.tau)*tar_param.data + self.tau*param.data)
        for tar_param,param in zip(self.critic_target.parameters(),self.critic.parameters()):
            tar_param.data.copy_((1-self.tau)*tar_param.data + self.tau*param.data)

    def save_model(self,train_step):
        num = str(train_step // self.save_rate)
        model_path = os.path.join("ZSMFG-NashQ/ZSMFG-Nash-DQN/ddpgModels")
        # Target networks are initilized withthe same params
        torch.save(self.actor.state_dict(),model_path +"/player"+self.id+ "/actor_params_"+str(num)+".pt")
        torch.save(self.critic.state_dict(),model_path +"/player"+self.id+ "/critic_params_"+str(num)+".pt")
        #torch.save(self.actor_target.state_dict(),model_path +"target/player"+self.id+ "/actor_params.pt")
        #torch.save(self.critic_target.state_dict(),model_path +"target/player"+self.id+ "/critic_params.pt")

