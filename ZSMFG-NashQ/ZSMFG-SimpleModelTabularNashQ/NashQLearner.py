from math import exp
from random import random
import random as rnd
import nashpy as nash
import numpy as np
from tqdm import tqdm
import copy

class NashQPlayer():

    def __init__(self, env,
                learning_rate = 0.5,
                iterations = 20,
                discount_factor = 0.7,
                decision_strategy = "epsilon-greedy",
                epsilon = 0.5,
                MonteCarlo = False,
                iter_save = 20,
                Q_1_table = None,
                Q_2_table = None
                ):
        self.env = env
        self.lr = learning_rate
        self.max_itrs = iterations
        self.disct_fct = discount_factor
        self.decision_strategy = decision_strategy
        self.epsilon = epsilon
        self.Q_1 = Q_1_table
        self.Q_2 =  Q_2_table
        self.MonteCarlo = MonteCarlo
        self.iter_save = iter_save
        # self.iters = []
        self.Q_1_diff_sup = []
        self.Q_1_diff_L2 = []
        self.Q_2_diff_sup = []
        self.Q_2_diff_L2 = []

        self.Q_1.init_states() #initialize the states
        self.Q_1.init_ctrl()
        self.Q_2.init_states() #initialize the states
        self.Q_2.init_ctrl()
        

        self.Q_1_visited_times = np.ones((self.Q_1.n_states,self.Q_1.n_controls,self.Q_2.n_controls))
        self.Q_2_visited_times = np.ones((self.Q_2.n_states,self.Q_1.n_controls,self.Q_2.n_controls))
        #print(self.Q_1_visited_times.shape)

    def lr_func(self,number_of_visited):

        return 1 / number_of_visited

    def adjust_eps(self,eps_start,eps_end,current_step):
        eps_threshold = eps_start
        eps_threshold = eps_end + (eps_start-eps_end)* exp(-1. *current_step/self.max_itrs)
        
        return eps_threshold
    
    def training(self):

        
        
        #current_state = [self.Q_1.states[random.randint(self.Q_1.n_states)],self.Q_2.states[random.randint(self.Q_2.n_states)]]
        
        
        
        current_states = [self.Q_1.states[-1],self.Q_2.states[0]]
        for i in tqdm(range(1,self.max_itrs+1)):
            if self.MonteCarlo:
                
                self.epsilon = self.adjust_eps(0.9,0.05,i)
                if self.decision_strategy == "random":
                    i_alpha_1 = self.Q_1.controls[rnd.randrange(self.Q_1.n_controls)]
                    i_alpha_2 = self.Q_2.controls[rnd.randrange(self.Q_2.n_controls)]

                if self.decision_strategy == "epsilon-greedy":
                    random_number = rnd.uniform(0,1)
                    if random_number >=self.epsilon:
                        strategies = self.env.solve_stage_game(self.Q_1.Q_table[self.Q_1.get_state_index(current_states[0])],
                        self.Q_2.Q_table[self.Q_2.get_state_index(current_states[1])])

                        i_alpha_1 = np.random.choice(len(strategies[0]),p = strategies[0])
                        i_alpha_2 = np.random.choice(len(strategies[1]),p = strategies[1])

                    else:
                        i_alpha_1 = rnd.randrange(self.Q_1.n_controls)
                        i_alpha_2 = rnd.randrange(self.Q_2.n_controls)

                # epsion = 0
                if self.decision_strategy == "greedy":
                    strategies = self.env.get_nash_Q_value(self.Q_1.Q_table[self.Q_1.get_state_index(current_states[0])],
                    self.Q_2.Q_table[self.Q_2.get_state_index(current_states[1])])

                    i_alpha_1 = np.random.choice(len(strategies[0]),p=strategies[0])
                    i_alpha_2 = np.random.choice(len(strategies[1]),p=strategies[1])


                next_mu_1 = self.env.get_next_mu(current_states[0],self.Q_1.controls[i_alpha_1])
                next_mu_2 = self.env.get_next_mu(current_states[1],self.Q_2.controls[i_alpha_2])

                i_mu_1_next = self.Q_1.proj_W_index(next_mu_1) # find its most nearest mu
                i_mu_2_next = self.Q_2.proj_W_index(next_mu_2)

                r_next_1, r_next_2 = self.env.get_population_level_reward(self.Q_1.states[i_mu_1_next], self.Q_2.states[i_mu_2_next])
                
                pi_1,pi_2 = self.env.solve_stage_game(self.Q_1.Q_table[i_mu_1_next],self.Q_2.Q_table[i_mu_2_next])

                Q_1_old = self.Q_1.Q_table.copy()

                Q_2_old = self.Q_2.Q_table.copy()
                
                # lr as a function of t
                self.lr_Q_1 = self.lr_func(self.Q_1_visited_times[self.Q_1.get_state_index(current_states[0])][i_alpha_1][i_alpha_2])
                self.lr_Q_2 = self.lr_func(self.Q_2_visited_times[self.Q_2.get_state_index(current_states[1])][i_alpha_1][i_alpha_2])

                # self.lr_Q_1 = 1/i
                # self.lr_Q_2 = 1/i

                self.Q_1_visited_times[self.Q_1.get_state_index(current_states[0])][i_alpha_1][i_alpha_2] += 1
                self.Q_2_visited_times[self.Q_2.get_state_index(current_states[1])][i_alpha_1][i_alpha_2] += 1

                self.Q_1.Q_table[self.Q_1.get_state_index(current_states[0])][i_alpha_1][i_alpha_2] = ((1-self.lr_Q_1) * self.Q_1.Q_table[self.Q_1.get_state_index(current_states[0])][i_alpha_1][i_alpha_2]
                    + self.lr_Q_1*(r_next_1 + self.disct_fct * np.dot(np.dot(pi_1, self.Q_1.Q_table[i_mu_1_next]),pi_2)))

                self.Q_2.Q_table[self.Q_2.get_state_index(current_states[1])][i_alpha_1][i_alpha_2] = ((1-self.lr_Q_2) * self.Q_2.Q_table[self.Q_2.get_state_index(current_states[1])][i_alpha_1][i_alpha_2]
                    + self.lr_Q_2*(r_next_2 + self.disct_fct * np.dot(np.dot(pi_1, self.Q_2.Q_table[i_mu_2_next]),pi_2)))

                current_states = [self.Q_1.states[i_mu_1_next],self.Q_2.states[i_mu_2_next]]

                # Check covergence
                self.Q_1_diff_sup.append(np.max(np.abs(self.Q_1.Q_table - Q_1_old)))
                #
                print("***** sup|Q_new - Q_old| = {}".format(self.Q_1_diff_sup[-1]))
                self.Q_1_diff_L2.append(np.sqrt(np.sum(np.square(self.Q_1.Q_table - Q_1_old))))
                print("***** L2|Q_new - Q_old| = {}\n".format(self.Q_1_diff_L2[-1]))
                
                self.Q_2_diff_sup.append(np.max(np.abs(self.Q_2.Q_table - Q_2_old)))
                #
                print("***** sup|Q_new - Q_old| = {}".format(self.Q_2_diff_sup[-1]))
                self.Q_2_diff_L2.append(np.sqrt(np.sum(np.square(self.Q_2.Q_table - Q_2_old))))
                print("***** L2|Q_new - Q_old| = {}\n".format(self.Q_2_diff_L2[-1]))
                if (i % self.iter_save == 0):
                    np.savez("ZSMFG-NashQ/historyTables/Q_MC_ones_results_iter{}".format(i), Q_1=self.Q_1.Q_table,Q_2=self.Q_2.Q_table, n_states_x=self.Q_1.n_states_x, n_steps_state=self.Q_1.n_steps_state,
                    n_steps_ctrl=self.Q_1.n_steps_ctrl, iters=self.max_itrs, Q_1_diff_sup=self.Q_1_diff_sup, Q_1_diff_L2=self.Q_1_diff_L2,Q_2_diff_sup=self.Q_2_diff_sup, Q_2_diff_L2=self.Q_2_diff_L2,
                    Q_1_visited=self.Q_1_visited_times,Q_2_visited = self.Q_2_visited_times)



            else:
                Q_1_old = self.Q_1.Q_table.copy()
                Q_2_old = self.Q_2.Q_table.copy()
                # In No MC version, we visit each (s,a1,a2),so lr always decrease with itr times
                self.lr = self.lr_func(i)
                # Monte Carlo T/F
                for i_mu in range(self.Q_1.n_states):
                    # print("i_mu = {}\n".format(i_mu))
                    
                    mu_1 = self.Q_1.states[self.Q_1.n_states-1-i_mu] # initial predator starts from left
                    mu_2 = self.Q_2.states[i_mu] # preyer starts from the right
                    for i_alpha_1 in range(self.Q_1.n_controls):

                        for i_alpha_2 in range(self.Q_2.n_controls):

                            # print("i_alpha = {}\n".format(i_alpha))
                            alpha_1 = self.Q_1.controls[i_alpha_1]
                            alpha_2 = self.Q_2.controls[i_alpha_2]
                            next_mu_1 = self.env.get_next_mu(mu_1,alpha_1)
                            next_mu_2 = self.env.get_next_mu(mu_2,alpha_2)

                            
                            i_mu_1_next = self.Q_1.proj_W_index(next_mu_1) # find its most nearest mu
                            i_mu_2_next = self.Q_2.proj_W_index(next_mu_2) # find its most nearest mu

                            r_next_1, r_next_2 = self.env.get_population_level_reward(self.Q_1.states[i_mu_1_next], self.Q_2.states[i_mu_2_next])


                            pi_1,pi_2 = self.env.solve_stage_game(self.Q_1.Q_table[i_mu_1_next],self.Q_2.Q_table[i_mu_2_next])
                            # print(pi_2)
                            # print("mu = {},\t mu_next = {}, \t mu_next_proj = {}".format(mu_1, next_mu_1,self.Q_1.states[i_mu_1_next]))
                            
                            # print(Q_old[i_mu_1_next][i_alpha_1][i_alpha_2])
                            self.Q_1.Q_table[self.Q_1.get_state_index(mu_1)][i_alpha_1][i_alpha_2] = ((1-self.lr)*self.Q_1.Q_table[self.Q_1.get_state_index(current_states[0])][i_alpha_1][i_alpha_2]
                                + self.lr*(r_next_1 + self.disct_fct * np.dot(np.dot(pi_1, self.Q_1.Q_table[i_mu_1_next]),pi_2)))

                            self.Q_2.Q_table[self.Q_2.get_state_index(mu_2)][i_alpha_1][i_alpha_2] = ((1-self.lr)*self.Q_2.Q_table[self.Q_2.get_state_index(current_states[1])][i_alpha_1][i_alpha_2]
                                + self.lr*(r_next_2 + self.disct_fct * np.dot(np.dot(pi_1, self.Q_2.Q_table[i_mu_2_next]),pi_2)))
                            
                            

                
                
                
                self.Q_1_diff_sup.append(np.max(np.abs(self.Q_1.Q_table - Q_1_old)))
                #
                print("***** sup|Q_new - Q_old| = {}".format(self.Q_1_diff_sup[-1]))
                self.Q_1_diff_L2.append(np.sqrt(np.sum(np.square(self.Q_1.Q_table - Q_1_old))))
                print("***** L2|Q_new - Q_old| = {}\n".format(self.Q_1_diff_L2[-1]))
                
                self.Q_2_diff_sup.append(np.max(np.abs(self.Q_2.Q_table - Q_2_old)))
                #
                print("***** sup|Q_new - Q_old| = {}".format(self.Q_2_diff_sup[-1]))
                self.Q_2_diff_L2.append(np.sqrt(np.sum(np.square(self.Q_2.Q_table - Q_2_old))))
                print("***** L2|Q_new - Q_old| = {}\n".format(self.Q_2_diff_L2[-1]))
                if (i % self.iter_save == 0):
                    np.savez("ZSMFG-NashQ/historyTables/Q_noMC_results_iter{}".format(i), Q_1=self.Q_1.Q_table,Q_2=self.Q_2.Q_table, n_states_x=self.Q_1.n_states_x, n_steps_state=self.Q_1.n_steps_state,
                    n_steps_ctrl=self.Q_1.n_steps_ctrl, iters=self.max_itrs, Q_1_diff_sup=self.Q_1_diff_sup, Q_1_diff_L2=self.Q_1_diff_L2,Q_2_diff_sup=self.Q_2_diff_sup, Q_2_diff_L2=self.Q_2_diff_L2)
                
                continue
            
            
        return self.Q_1, self.Q_2

    def value_iterations(self,Q,Q_fixed,pi_fixed,env,max_steps):
        
        total_reward_1 = 0
        total_reward_2 = 0

        
        
        #current_state = [self.Q_1.states[random.randint(self.Q_1.n_states)],self.Q_2.states[random.randint(self.Q_2.n_states)]]
        current_states = [Q.states[-1],Q_fixed.states[0]]
        
        
        for i in tqdm(range(1,max_steps+1)):
            if self.MonteCarlo:
                self.epsilon = self.adjust_eps(0.9,0.05,i)
                i_alpha_2 = np.random.choice(len(pi_fixed[tuple(map(tuple,current_states))]),p = pi_fixed[tuple(map(tuple,current_states))])
                if self.decision_strategy == "random":
                    i_alpha_1 = self.table.controls[rnd.randrange(Q.n_controls)]

                if self.decision_strategy == "epsilon-greedy":
                    random_number = rnd.uniform(0,1)
                    if random_number >=self.epsilon:

                        i_alpha_1 = np.argmax(Q.Q_table[Q.get_state_index(current_states[0])][i_alpha_2])


                    else:
                        i_alpha_1 = rnd.randrange(Q_fixed.n_controls)

                if self.decision_strategy == "greedy":

                    i_alpha_1 = np.argmax(Q.Q_table[Q.get_state_index(current_states[0])][i_alpha_2])


                next_mu_1 = env.get_next_mu(current_states[0],Q.controls[i_alpha_1])
                next_mu_2 = env.get_next_mu(current_states[1],Q_fixed.controls[i_alpha_2])

                i_mu_1_next = Q.proj_W_index(next_mu_1) # find its most nearest mu
                i_mu_2_next = Q_fixed.proj_W_index(next_mu_2)

                r_next_1, r_next_2 = env.get_population_level_reward(Q.states[i_mu_1_next], Q_fixed.states[i_mu_2_next])
                
                #self.lr_Q_1 = sel
                total_reward_1 += r_next_1
                total_reward_2 += r_next_2

                Q.Q_table[Q.get_state_index(current_states[0])][i_alpha_1][i_alpha_2] = ((1-self.lr)*Q.Q_table[Q.get_state_index(current_states[0])][i_alpha_1][i_alpha_2]
                    + self.lr*(r_next_1 + self.disct_fct * Q.Q_table[i_mu_1_next][i_alpha_1][i_alpha_2]))

                Q_fixed.Q_table[Q_fixed.get_state_index(current_states[1])][i_alpha_1][i_alpha_2] = ((1-self.lr)*Q_fixed.Q_table[Q_fixed.get_state_index(current_states[1])][i_alpha_1][i_alpha_2]
                    + self.lr*(r_next_2 + self.disct_fct * Q_fixed.Q_table[i_mu_2_next][i_alpha_1][i_alpha_2]))
            else:
                for i_mu in range(Q.n_states):
                    # print("i_mu = {}\n".format(i_mu))

                    mu_1 = Q.states[Q.n_states-1-i_mu] # initial predator starts from left
                    mu_2 = Q_fixed.states[i_mu] # preyer starts from the right
                    for i_alpha_1 in range(Q.n_controls):

                        for i_alpha_2 in range(Q_fixed.n_controls):

                            # print("i_alpha = {}\n".format(i_alpha))
                            alpha_1 = Q.controls[i_alpha_1]
                            alpha_2 = Q_fixed.controls[i_alpha_2]
                            next_mu_1 = env.get_next_mu(mu_1,alpha_1)
                            next_mu_2 = env.get_next_mu(mu_2,alpha_2)

                            
                            i_mu_1_next = Q.proj_W_index(next_mu_1) # find its most nearest mu
                            i_mu_2_next = Q_fixed.proj_W_index(next_mu_2) # find its most nearest mu

                            r_next_1, r_next_2 = env.get_population_level_reward(Q.states[i_mu_1_next], Q_fixed.states[i_mu_2_next])


                            #pi_1,pi_2 = self.env.get_nash_Q_value(self.Q_1.Q_table[i_mu_1_next],self.Q_2.Q_table[i_mu_2_next])
                            # print(pi_2)
                            # print("mu = {},\t mu_next = {}, \t mu_next_proj = {}".format(mu_1, next_mu_1,self.Q_1.states[i_mu_1_next]))
                            
                            # print(Q_old[i_mu_1_next][i_alpha_1][i_alpha_2])
                            total_reward += r_next_1
                            Q.Q_table[Q.get_state_index(current_states[0])][i_alpha_1][i_alpha_2] = ((1-self.lr)*Q.Q_table[Q.get_state_index(current_states[0])][i_alpha_1][i_alpha_2]
                    + self.lr*(r_next_1 + self.disct_fct * np.max(Q.Q_table[i_mu_1_next][i_alpha_1])))
                            Q_fixed.Q_table[Q_fixed.get_state_index(current_states[1])][i_alpha_1][i_alpha_2] = ((1-self.lr)*Q_fixed.Q_2.Q_table[Q_fixed.get_state_index(current_states[1])][i_alpha_1][i_alpha_2]
                    + self.lr*(r_next_2 + self.disct_fct * np.max(Q_fixed.Q_table[i_mu_2_next][i_alpha_2])))
                
                continue
            current_states = [self.Q_1.states[i_mu_1_next],self.Q_2.states[i_mu_2_next]]
            return total_reward_1, total_reward_2

    
    def recover_equilibrium_policy(self,max_steps, Q_1, Q_2,env):
        # recover the whole stage policy given two Q-tables
        current_states = [Q_1.states[-1],Q_2.states[0]]
        r_1 = 0
        r_2 = 0

        policy_1 = dict()
        policy_2 = dict()
        # dictionary instead of list, states as keys
        for i in tqdm(range(max_steps)):
            #print(current_states)
            
            Q_1_stage_table = Q_1.Q_table[Q_1.get_state_index(current_states[0])]
            Q_2_stage_table = Q_2.Q_table[Q_2.get_state_index(current_states[1])]
            #print(Q_1_stage_table==Q_2_stage_table)
            # get_nash_Q_value
            print(current_states)
            pi_1,pi_2 = env.solve_stage_game(Q_1_stage_table,Q_2_stage_table)
            policy_1[tuple(map(tuple,current_states))] = pi_1
            policy_2[tuple(map(tuple,current_states))] = pi_2

            i_alpha_1 = np.random.choice(len(pi_1),p=pi_1)
            i_alpha_2 = np.random.choice(len(pi_2),p=pi_2)
            # print(pi_1)
            # print(pi_2)

            next_mu_1 = env.get_next_mu(current_states[0],Q_1.controls[i_alpha_1])
            next_mu_2 = env.get_next_mu(current_states[1],Q_2.controls[i_alpha_2])
            reward_1, reward_2 = env.get_population_level_reward(next_mu_1,next_mu_2)
            r_1 += reward_1
            r_2 += reward_2
            i_mu_1_next = Q_1.proj_W_index(next_mu_1) 
            i_mu_2_next = Q_2.proj_W_index(next_mu_2)
            # print(i_mu_1_next)
            # print(i_mu_2_next)
            current_states = [Q_1.states[i_mu_1_next],Q_2.states[i_mu_2_next]]
            
        
        return policy_1, policy_2, r_1, r_2
    





