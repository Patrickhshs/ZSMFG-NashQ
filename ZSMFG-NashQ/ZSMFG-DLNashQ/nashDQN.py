import os
from pickletools import long1
import torch 
import random as rnd
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from base import ValueNet
import itertools
import numpy as np
from tools import ReplayBuffer
import copy



class NashDQN(object):

    def __init__(self,lr,action_dim,gamma,batch_size,env,replay_buffer=ReplayBuffer,max_iteration=1000,state_dim = 3,save_rate = 100,epsilon=0.5,n_steps_ctrl=10,max_episode=100,baseNet=ValueNet,device = None):

        self.device = device

        self.lr = lr
        self.gamma = gamma
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.replay_buffer = replay_buffer(self.state_dim,self.action_dim)
        self.env = env
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.Q_1 = baseNet(self.state_dim,self.action_dim).to(self.device)
        self.Q_2 = baseNet(self.state_dim,self.action_dim).to(self.device)
        self.max_episode = max_episode
        self.n_steps_ctrl = n_steps_ctrl
        self.max_iteration = max_iteration
        self.training_step = 0
        self.save_rate = save_rate
        self.n_steps_ctrl = 2 # Hyperparameters to discretize action space
        
        self.action_space = self.init_action_space()
        self.n_action = self.action_space.shape[0]

        #self.payoff_mat_1,self.payoff_mat_2 = self.generate_payoff_matrix()
        self.optimizer_1 = torch.optim.AdamW(self.Q_1.parameters(),lr=self.lr)
        self.optimizer_2 = torch.optim.AdamW(self.Q_2.parameters(),lr=self.lr)
        # To-do:
        # set a target network to avoid overfitting

        self.target_Q_1 = copy.deepcopy(self.Q_1)  
        self.target_Q_2 = copy.deepcopy(self.Q_2)

    
    def init_action_space(self):
            combi_ctrl = itertools.product(np.linspace(0,self.n_steps_ctrl-1,self.n_steps_ctrl,dtype=int), repeat=self.action_dim)# n_states_x) # cartesian product; all possible controls as functions of state_x
            controls = np.asarray([el for el in combi_ctrl]) # np.linspace(0,1,n_steps_ctrl+1)
            control = controls[np.where(np.sum(controls, axis=1) == self.n_steps_ctrl)] / float(self.n_steps_ctrl) # all possible combination of distributions
            combi_population_level_ctrl = itertools.product(control,repeat=self.state_dim)
            return np.asarray([el for el in combi_population_level_ctrl])


    def generate_payoff_matrix(self,states):
        m=n =self.action_space.shape[0]
        mat_1 = np.zeros((m,n))
        mat_2 = np.zeros((m,n))
        for i in range(m):
            for j in range(n):
                #print(torch.LongTensor(states[0]),torch.LongTensor(self.action_space[i]),torch.LongTensor(self.action_space[j]))
                mat_1[i][j] = float(self.target_Q_1(torch.unsqueeze(torch.tensor(states[0],device=self.device, dtype=torch.float32),0,),torch.unsqueeze(torch.tensor(self.action_space[i],device=self.device, dtype=torch.float32),0),torch.unsqueeze(torch.tensor(self.action_space[j],device=self.device, dtype=torch.float32),0)))
                mat_2[i][j] = float(self.target_Q_2(torch.unsqueeze(torch.tensor(states[1],device=self.device, dtype=torch.float32),0),torch.unsqueeze(torch.tensor(self.action_space[i],device=self.device, dtype=torch.float32),0),torch.unsqueeze(torch.tensor(self.action_space[j],device=self.device, dtype=torch.float32),0)))

        return mat_1,mat_2


    def training(self):
        for l in tqdm(range(self.max_iteration)):
            self.current_states = [[1,0,0],[0,0,1]]
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
                self.replay_buffer.store(self.current_states,[self.action_space[i_alpha_1],self.action_space[i_alpha_2]],rewards,next_states)
                self.current_states = next_states
                batch_state,batch_actions,batch_rewards,batch_new_states = self.replay_buffer.sample(self.batch_size)
                target_y_1 = torch.zeros((self.batch_size,1),device=self.device)
                target_y_2 = torch.zeros((self.batch_size,1), device = self.device)
                loss_func = torch.nn.MSELoss()
                for i in range(self.batch_size):
                        payoff_mat_1,payoff_mat_2 = self.generate_payoff_matrix([batch_new_states[i][0],batch_new_states[i][1]])
                        pi_1,pi_2 = self.env.solve_stage_game(payoff_mat_1,payoff_mat_2)
                        #print((torch.FloatTensor(pi_1)@torch.FloatTensor(payoff_mat_1)@torch.FloatTensor(pi_2)))

                        target_y_1[i] = batch_rewards[i][0] + self.gamma*(torch.tensor(pi_1,device=self.device, dtype=torch.float32)@torch.tensor(payoff_mat_1,device=self.device, dtype=torch.float32)@torch.tensor(pi_2,device=self.device, dtype=torch.float32))
                        target_y_2[i] = batch_rewards[i][1] + self.gamma*(torch.tensor(pi_1,device=self.device, dtype=torch.float32)@torch.tensor(payoff_mat_2,device=self.device, dtype=torch.float32)@torch.tensor(pi_2,device=self.device, dtype=torch.float32))
                        #print(batch_actions[i][0].shape)
                batch_actions = batch_actions.transpose(0,1)
                batch_state = batch_state.transpose(0,1)
                # print(self.Q_1(batch_state[0],batch_actions[0],batch_actions[1]).shape)
                # print(target_y_1.shape)
                loss_1 = loss_func(target_y_1,self.target_Q_1(torch.tensor(batch_state[0],device=self.device),torch.tensor(batch_actions[0],device = self.device),torch.tensor(batch_actions[1],device=self.device)))
                loss_2 = loss_func(target_y_2,self.target_Q_2(torch.tensor(batch_state[1],device=self.device),torch.tensor(batch_actions[0],device = self.device),torch.tensor(batch_actions[1],device=self.device)))
                self.optimizer_1.zero_grad()
                loss_1.backward()
                self.optimizer_1.step()
                self.optimizer_2.zero_grad()
                loss_2.backward()
                self.optimizer_2.step()
                print("Training Loss for player 1 "+str(loss_1))
                print("Training Loss for player 2 "+str(loss_2))
                if self.training_step > 0 and self.training_step % self.save_rate == 0:
                    self.save_model(self.training_step)
                self.training_step +=1
            self.target_Q_1 = copy.deepcopy(self.Q_1)
            self.target_Q_2 = copy.deepcopy(self.Q_2)

    def save_model(self,train_step):
        num = str(train_step // self.save_rate)
        model_path = os.path.join("ZSMFG-DLNashQ/nashDQNmodels")
        # Target networks are initilized withthe same params
        torch.save(self.Q_1.state_dict(),model_path +"/player_"+str(1)+ "_nn_params.pt")
        torch.save(self.Q_2.state_dict(),model_path +"/player_"+str(2)+ "_nn_params.pt")







            




