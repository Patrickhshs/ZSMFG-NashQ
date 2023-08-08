from asyncio import FastChildWatcher
from random import random
import nashpy as nash
import numpy as np
import tqdm

class NashQPlayer():

    def __init__(self, env,table_type,
                learning_rate = 0.5,
                iterations = 20,
                discount_factor = 0.7,
                decision_strategy = "epsilong-greedy",
                epsilon = 0.5,
                MonteCarlo = False
                ):
        self.env=env
        self.lr = learning_rate
        self.max_itrs = iterations
        self.disct_fct = discount_factor
        self.decision_strategy = decision_strategy
        self.epsilon = epsilon
        self.Q_predator = table_type
        self.Q_preyer = table_type
        self.MonteCarlo = MonteCarlo
    
    def training(self):
        self.Q_predator.init_states() #initialize the states
        self.Q_predator.init_ctrl()
        self.Q_preyer.init_states() #initialize the states
        self.Q_preyer.init_ctrl()

        # Set terminal stop? should we?
        preyer_terminate_state = self.Q_predator.states[-1]
        predator_terminate_state = self.Q_preyer.states[0]

        
        
        #current_state = [self.Q_predator.states[random.randint(self.Q_predator.n_states)],self.Q_preyer.states[random.randint(self.Q_preyer.n_states)]]
        current_states = [self.Q_predator.states[-1],self.Q_preyer.states[0]]
        
        for i in tqdm(range(self.max_itrs)):
            if self.MonteCarlo:
                if current_states[0] == predator_terminate_state or current_states[1] ==preyer_terminate_state:
                    current_states = [self.Q_predator.states[-1],self.Q_preyer.states[0]]

                if self.decision_strategy == "random":
                    i_alpha_1 = self.table.controls[random.randrange(self.n_controls)]
                    i_alpha_2 = self.table.controls[random.randrange(self.n_controls)]

                if self.decision_strategy == "epsilon-greedy":
                    random_number = random.uniform(0,1)
                    if random_number >=self.epsilon:
                        strategies = self.env.get_nash_Q_value(self.Q_predator.Q_table[current_states[0]],
                        self.Q_preyer.Q_table[current_states[1]])

                        i_alpha_1 = np.argmax(strategies[0])
                        i_alpha_2 = np.argmax(strategies[1])

                    else:
                        i_alpha_1 = self.Q_predator.controls[random.randrange(self.n_controls)]
                        i_alpha_2 = self.Q_preyer.controls[random.randrange(self.n_controls)]

                if self.decision_strategy == "greedy":
                    strategies = self.env.get_nash_Q_value(self.Q_predator.Q_table[current_states[0]],
                    self.Q_preyer.Q_table[current_states[1]])

                    # Is it the right way to sample alpha bar?
                    i_alpha_1 = np.argmax(strategies[0])
                    i_alpha_2 = np.argmax(strategies[1])

                next_mu_1 = self.env.get_next_mu(current_states[0],self.Q_predator.controls[i_alpha_1])
                next_mu_2 = self.env.get_next_mu(current_states[1],self.Q_preyer.controls[i_alpha_2])

                i_mu_1_next = self.Q_predator.proj_W_index(next_mu_1) # find its most nearest mu
                i_mu_2_next = self.Q_preyer.proj_W_index(next_mu_2)

                r_next_1, r_next_2 = self.env.get_population_level_reward(self.Q_predator.states[i_mu_1_next], self.Q_preyer.states[i_mu_2_next])
                
                # do we need 
                pi_1,pi_2 = self.env.get_nash_Q_value(self.Q_predator.Q_table[i_mu_1_next],self.Q_preyer.Q_table[i_mu_2_next])

                self.Q_predator.Q_table[current_states[0]][i_alpha_1][i_alpha_2] = ((1-self.lr)*self.Q_predator.Q_table[current_states[0]][i_alpha_1][i_alpha_2]
                    + self.lr*(r_next_1 + self.disct_fct * np.dot(np.dot(pi_1, self.Q_predator.Q_table[i_mu_1_next]),pi_2)))

                self.Q_preyer.Q_table[current_states[0]][i_alpha_1][i_alpha_2] = ((1-self.lr)*self.Q_preyer.Q_table[current_states[0]][i_alpha_1][i_alpha_2]
                    + self.lr*(r_next_2 + self.disct_fct * np.dot(np.dot(pi_1, self.Q_preyer.Q_table[i_mu_2_next]),pi_2)))
            else:
                for i_mu in range(table.n_states):
                    # print("i_mu = {}\n".format(i_mu))

                    mu_1 = table.states[table.n_states-1-i_mu] # initial predator starts from left
                    mu_2 = table.states[i_mu] # preyer starts from the right
                    for i_alpha_1 in range(table.n_controls):

                        for i_alpha_2 in range(table.n_controls):

                            # print("i_alpha = {}\n".format(i_alpha))
                            alpha_1 = table.controls[i_alpha_1]
                            alpha_2 = table.controls[i_alpha_2]
                            next_mu_1 = env.get_next_mu(mu_1,alpha_1)
                            next_mu_2 = env.get_next_mu(mu_2,alpha_2)

                            
                            i_mu_1_next = table.proj_W_index(next_mu_1) # find its most nearest mu
                            i_mu_2_next = table.proj_W_index(next_mu_2) # find its most nearest mu

                            r_next_1, r_next_2 = env.get_population_level_reward(table.states[i_mu_1_next], table.states[i_mu_2_next])

                            # Nash Q
                            #r_matrix_1,r_matrix_2 = env.get_reward_mat(table.states[i_mu_1_next],table.states[i_mu_2_next],table)
                            # print(r_matrix_1.shape)
                            # print(Q_old[i_mu_1_next].shape)
                            # print(Q_old_anta[i_mu_2_next].shape)
                            #print(r_matrix_1)
                            #print(r_matrix_2)

                            # print(Q_old[i_mu_1_next])
                            # print(Q_old_anta[i_mu_2_next])
                            pi_1,pi_2 = env.find_nash_gambit(Q_old[i_mu_1_next],Q_old_anta[i_mu_2_next])
                            #pi_1,pi_2 = env.get_nash_Q_value(Q_old[i_mu_1_next],Q_old_anta[i_mu_2_next],table)
                            
                            #print(r_matrix_1)
                            #print(r_matrix_2)
                            #pi_1, pi_2 = env.linear_programming_duality(Q_old[i_mu_1_next],Q_old_anta[i_mu_2_next])
                            #pi_1, pi_2 = env.compute_nash_equilibrium(Q_old[i_mu_1_next],Q_old_anta[i_mu_2_next])
                            
                            #print(Q_old[i_mu_1_next]==-Q_old_anta[i_mu_2_next])
                            # print(pi_1)
                            # print(pi_2.shape)
                            # print(Q_old)
                            
                            #print(pi_1[1].shape)
                            
                            # print(Q_old[i_mu_1_next][i_alpha_1][i_alpha_2])
                            Q_nash = np.dot(np.dot(pi_1, Q_old[i_mu_1_next]),pi_2)
                            # print(np.multiply(pi_1[1].T, Q_old[i_mu_1_next]).shape)
                            Q_nash_anta = np.dot(np.dot(pi_1, Q_old_anta[i_mu_2_next]),pi_2)
                            # print(r_next_1)\
                            
                            # print("mu = {},\t mu_next = {}, \t mu_next_proj = {}".format(mu_1, next_mu_1, table.states[i_mu_1_next]))
                            
                            #update the New Q table
                            Q_new[i_mu][i_alpha_1][i_alpha_2] += lr * (r_next_1 + discount * Q_nash)
                            Q_new_anta[i_mu][i_alpha_1][i_alpha_2] += lr * (r_next_2 + discount * Q_nash_anta)
            current_states = [self.Q_predator.states[i_mu_1_next],self.Q_preyer.states[i_mu_2_next]]
            
            
        return self.Q_predator, self.Q_preyer

    
    def recover_equilibrium_policy(self,starting_states,Q_1, Q_2,env):
        # recover the whole stage policy given two Q-tables
        current_states = starting_states
        preyer_terminate_state = Q_1.states[-1]
        predator_terminate_state = Q_2.states[0]
        policy_predator = []
        policy_preyer = []
        while current_states[0] != preyer_terminate_state or current_states[1] != predator_terminate_state:
            print(current_states)
            Q_predator = Q_1[current_states[0]]
            Q_preyer = Q_2[current_states[1]]

            pi_1,pi_2 = env.get_nash_Q_value(Q_predator,Q_preyer)
            policy_predator.append(pi_1)
            policy_preyer.append(pi_2)

            i_alpha_1 = np.argmax(pi_1)
            i_alpha_2 = np.argmax(pi_2)
            next_mu_1 = env.get_next_mu(current_states[0],Q_predator.controls[i_alpha_1])
            next_mu_2 = env.get_next_mu(current_states[1],Q_preyer.controls[i_alpha_2])

            i_mu_1_next = Q_predator.proj_W_index(next_mu_1) # find its most nearest mu
            i_mu_2_next = Q_preyer.proj_W_index(next_mu_2)
            current_states = [Q_predator.states[i_mu_1_next],Q_preyer.states[i_mu_2_next]]
        
        return policy_predator, policy_preyer


