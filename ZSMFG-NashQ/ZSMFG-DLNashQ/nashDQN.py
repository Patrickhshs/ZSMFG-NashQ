import os
import torch 
import random as rnd
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from base import ValueNet
import itertools
import numpy as np
from tools import ReplayBuffer
from copy import copy



class NashDQN(object):

    def __init__(self,lr,action_dim,gamma,batch_size,env,replay_buffer=ReplayBuffer,max_iteration=1000,state_dim = 4,save_rate = 100,epsilon=0.5,n_steps_ctrl=10,max_episode=100,baseNet=ValueNet):

        self.lr = lr
        self.gamma = gamma
        self.replay_buffer = replay_buffer
        self.env = env
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.Q_1 = baseNet
        self.Q_2 = baseNet
        self.current_states = [[1,0,0,0],[0,0,0,1]]
        self.max_episode = max_episode
        self.n_steps_ctrl = n_steps_ctrl
        self.max_iteration = max_iteration
        self.training_step = 0
        self.save_rate = save_rate
        self.action_dim = action_dim
        #self.action_space = self.init_action_space()
        #self.payoff_mat_1,self.payoff_mat_2 = self.generate_payoff_matrix()
        self.optimizer_1 = torch.optim.AdamW(self.Q_1.parameters(),lr=self.lr)
        self.optimizer_2 = torch.optim.AdamW(self.Q_2.parameters(),lr=self.lr)
        # To-do:
        # set a target network to avoid overfitting

        self.target_Q_1 = copy.deepcopy(baseNet)  
        self.target_Q_2 = copy.deepcopy(baseNet)

    
    def init_action_space(self):
        combi_ctrl = itertools.product(np.linspace(0,self.n_steps_ctrl,self.n_steps_ctrl+1,dtype=int), repeat=3)# n_states_x) # cartesian product; all possible controls as functions of state_x
        controls = np.asarray([el for el in combi_ctrl]) # np.linspace(0,1,n_steps_ctrl+1)
        return controls[np.where(np.sum(controls, axis=1) == self.n_steps_ctrl)] / float(self.n_steps_ctrl)

    def generate_payoff_matrix(self,states):
        m,n = self.action_space.shape
        mat_1 = np.zeros(m,n)
        mat_2 = np.zeros(m,n)
        for i in range(m):
            for j in range(n):
                mat_1 = self.target_Q_1(states[0],self.action_space[i],self.action_space[j])
                mat_2 = self.target_Q_2(states[1],self.action_space[j],self.action_space[i])

        return mat_1,mat_2


    def training(self):
        for l in tqdm(range(self.max_iteration)):
            for i in tqdm(range(self.max_episode)):
                payoff_mat_1, payoff_mat_2 =  self.generate_payoff_matrix(self.current_states)
                random_number = rnd.uniform(0,1)
                if random_number >= self.epsilon:
                    strategies = self.env.solve_stage_game(payoff_mat_1,payoff_mat_2)
                    
                    i_alpha_1 = np.random.choice(len(strategies[0]),p = strategies[0])
                    i_alpha_2 = np.random.choice(len(strategies[1]),p = strategies[1])
                else:
                    i_alpha_1 = rnd.randrange(self.action_space.shape[0])
                    i_alpha_2 = rnd.randrange(self.action_space.shape[0])
                
                next_state_1 = self.env.get_next_mu(self.current_states[0],self.action_space[i_alpha_1])
                next_state_2 = self.env.get_next_mu(self.current_states[1],self.action_space[i_alpha_2])

                r_next_1, r_next_2 = self.env.get_population_level_reward(next_state_1, next_state_2)
                next_states = [next_state_1,next_state_2]
                rewards = [r_next_1,r_next_2]
                self.replay_buffer.store(self.current_states,[i_alpha_1,i_alpha_2],rewards,next_states)
                self.current_states = next_states
                batch_state,batch_actions,batch_rewards,batch_new_states = self.replay_buffer.sample(self.batch_size)
                target_y_1 = torch.zeros((self.batch_size,1))
                target_y_2 = torch.zeros((self.batch_size,1))

                # for i in range(self.batch_size):
                #     payoff_mat_1,payoff_mat_2 = self.generate_payoff_matrix(batch_new_states[i][0],batch_new_states[i][1])
                #     pi_1,pi_2 = self.env.solve_stage_game(payoff_mat_1,payoff_mat_2)
                #     target_y_1[i] = batch_rewards[i] + self.gamma*torch.dot(torch.dot(pi_1,payoff_mat_1),pi_2)
                #     target_y_2[i] = batch_rewards[i] + self.gamma*torch.dot(torch.dot(pi_1,payoff_mat_2),pi_2)

                
                
                loss_1 = nn.MSELoss(target_y_1,self.Q_1(batch_state,batch_actions[0],batch_actions[1]))
                loss_2 = nn.MSELoss(target_y_2,self.Q_2(batch_state,batch_actions[0],batch_actions[1]))

                self.optimizer_1.zero_grad()
                loss_1.backward()
                self.optimizer_1.step()
                self.optimizer_2.zero_grad()
                loss_2.backward()
                self.optimizer_2.step()
                if self.training_step > 0 and self.training_step % self.save_rate == 0:
                    self.save_model(self.training_step)
                    self.target_Q_1 = copy.deepcopy(self.Q_1)
                    self.target_Q_2 = copy.deepcopy(self.Q_2)
                self.training_step +=1

    def save_model(self,train_step):
        num = str(train_step // self.save_rate)
        model_path = os.path.join("ZSMFG-NashQ/ZSMFG-Nash-DQN/nashDQNmodels")
        # Target networks are initilized withthe same params
        torch.save(self.Q_1.state_dict(),model_path +"/player"+1+ "/nn_params.pt")
        torch.save(self.Q_2.state_dict(),model_path +"/player"+2+ "/nn_params.pt")

    # def compute_nash(self,q_values):
    #     q_tables = q_values.reshape(-1,self.)





            




