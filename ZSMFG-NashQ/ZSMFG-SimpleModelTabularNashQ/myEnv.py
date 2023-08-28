from random import random
import numpy as np
import nashpy as nash
import random as rnd
from nashpy.algorithms.lemke_howson_lex import lemke_howson_lex
from ecos_solver import NashEquilibriumECOSSolver


class my1dGridEnv(object):

    def __init__(self,size= 3):
        self.size = size # Dimension of 1D world
        self.n_states = self.size 
        self.n_actions = 3 
        self.epsilon = [0.2,0.3,0.5] # [0,-1,1]
        self.action_space = [0,-1,1]
        self.c = 1 # proportion of the density of other population
        #self.T = 0.2



    def get_index(self,m):

        return m

    # recalculate the transition_matrix because every time the agents deploys a different strategy pi
    def cal_transition_matrix(self,pi):

        # pi=[p_stay,p_left,p_right]
        trans_matrix = np.zeros((self.n_states,self.n_states))
        epsilon_matrix = np.zeros((self.n_states,self.n_states))
        self.epsilon = np.random.rand(3)
        self.epsilon /=np.sum(self.epsilon)
        #print(self.epsilon)
        for i in range(self.n_states):
            if i==0:
                trans_matrix[self.get_index(i),self.get_index(i)] = pi[0]
                epsilon_matrix[self.get_index(i),self.get_index(i)] = self.epsilon[0]
                trans_matrix[self.get_index(i),-1] = pi[1]
                epsilon_matrix[self.get_index(i),-1] = self.epsilon[1]
                trans_matrix[self.get_index(i),self.get_index(i)+1] = pi[2]
                epsilon_matrix[self.get_index(i),self.get_index(i)+1] = self.epsilon[2]
            elif i==self.n_states-1:
                trans_matrix[self.get_index(i),self.get_index(i)] = pi[0]
                epsilon_matrix[self.get_index(i),self.get_index(i)] = self.epsilon[0]
                trans_matrix[self.get_index(i),self.get_index(i)-1] = pi[1]
                epsilon_matrix[self.get_index(i),self.get_index(i)-1] = self.epsilon[1]
                trans_matrix[self.get_index(i),0] = pi[2]
                epsilon_matrix[self.get_index(i),0] = self.epsilon[2]
            else:
                trans_matrix[self.get_index(i),self.get_index(i)] = pi[0]
                epsilon_matrix[self.get_index(i),self.get_index(i)] = self.epsilon[0]
                trans_matrix[self.get_index(i),self.get_index(i)-1] = pi[1]
                epsilon_matrix[self.get_index(i),self.get_index(i)-1] = self.epsilon[1]
                trans_matrix[self.get_index(i),self.get_index(i)+1] = pi[2]
                epsilon_matrix[self.get_index(i),self.get_index(i)+1] = self.epsilon[2]
        
        return trans_matrix, epsilon_matrix





    # def get_agent_level_reward(self,state,mu_of_other_population,agent1=True):
    #     if agent1:
    #         cost = self.c*mu_of_other_population[state]

    #     return cost
            


    # visit every action pair of player 1 and player 2 to get reward matrix
    def get_population_level_reward(self,mu_1,mu_2):
        reward_1 = self.c*(np.dot(mu_1,mu_2))
        reward_2 = -self.c*(np.dot(mu_1,mu_2))
        
        
        return  reward_1, reward_2

    def get_reward_mat(self,mu_1,mu_2,table):
        reward_mat_1 = np.zeros((table.n_controls,table.n_controls))
        reward_mat_2 = np.zeros((table.n_controls,table.n_controls))


        simplex_controls = table.controls
        for i in range(len(simplex_controls)):
            for l in range(len(simplex_controls)):
                next_mu_1 = self.get_next_mu(mu_1,simplex_controls[i])
                next_mu_2 = self.get_next_mu(mu_2,simplex_controls[l])
                i_next_1 = table.proj_W_index(next_mu_1)
                i_next_2 = table.proj_W_index(next_mu_2)

                reward_1, reward_2 = self.get_population_level_reward(table.states[i_next_1], table.states[i_next_2])
                reward_mat_1[i][l] = reward_1
                reward_mat_2[i][l] = reward_2


                 
        return  reward_mat_1, reward_mat_2




    def get_next_mu(self,mu,strategy):
        transi_mat,epi_mat = self.cal_transition_matrix(strategy)
        P = transi_mat@epi_mat

        return P@mu.T




        


    # def solve_stage_game(self,payoff_mat_1,payoff_mat_2):
    #         # Zero sum case solver to get stage nash eq by lemke-Howson

    #         dim = payoff_mat_1.shape[0]
    #         final_eq = None
    #         # To handle the problem that sometimes Lemke-Howson implementation will give 
    #         # wrong returned NE shapes or NAN in value, use different initial_dropped_label value 
    #         # to find a valid one. 
    #         eq_enumeration = []
    #         for l in range(0, sum(payoff_mat_1.shape) - 1):
    #             # Lemke Howson can not solve degenerate matrix.
    #             # eq = rps.lemke_howson(initial_dropped_label=l) # The initial_dropped_label is an integer between 0 and sum(A.shape) - 1

    #             # Lexicographic Lemke Howson can solve degenerate matrix: https://github.com/newaijj/Nashpy/blob/ffea3522706ad51f712d42023d41683c8fa740e6/tests/unit/test_lemke_howson_lex.py#L9 
                
    #             try:
    #                 eq = lemke_howson_lex(payoff_mat_1, payoff_mat_2, initial_dropped_label=l)
    #             except ValueError as e:
    #                 if str(e) == "could not find dropped label":
    #                     # Skip to next label
    #                     print("Skipping label, no dropped label found")
    #                     continue
    #                 elif "pivot" in str(e):
    #                     continue
    #             else:   
    #                 if eq[0].shape[0] ==  dim and eq[1].shape[0] == dim and not np.isnan(eq[0]).any() and not np.isnan(eq[1]).any():
    #                     # valid shape and valid value (not nan)
    #                     return eq
    #                     #eq_enumeration.append(eq)
    #         final_eq = eq_enumeration[rnd.randrange(len(eq_enumeration))]
    #         #print(final_eq)
    #         if final_eq is None:
    #             raise ValueError('No valid Nash equilibrium is found!')
    #         return final_eq

    def solve_stage_game(self,payoff_mat_1,payoff_mat_2):
        pi_1 = NashEquilibriumECOSSolver(payoff_mat_1)[0]
        pi_2 = NashEquilibriumECOSSolver(payoff_mat_2)[0]
        return pi_1,pi_2
            
            # game = nash.Game(payoff_mat_1, payoff_mat_2)

            # equilibriums = list(game.support_enumeration())
            # #equilibriums = list(game.lemke_howson_enumeration())
            # # #equilibria = game.vertex_enumeration()
            # greedy_equilibrium = equilibriums[random.randrange(len(equilibriums))]
            # ##print(greedy_equilibrium)
            # # if np.where(greedy_equilibrium[0]==1)[0] == 0:
            # #     pi1 = random.randrange(table.n_controls)
            # #     pi2 = random.randrange(table.n_controls)

            # # else:
            # #     pi1 = greedy_equilibrium[0]
            # #     pi2 = greedy_equilibrium[1]
            # # print(np.where(greedy_equilibrium[0]==1))
            # # print(pi1,pi2)
            # pi = None
            # for _pi in equilibriums:
            #     if _pi[0].shape == (self.n_actions, ) and _pi[1].shape == (
            #             self.n_actions, ):
            #         if any(
            #             np.isnan(
            #                 _pi[0])) is False and any(
            #             np.isnan(
            #                 _pi[1])) is False:
            #             pi = _pi
            #             break
            # #print(pi)
            # if pi is None:
            #     # pi1 = np.repeat(
            #     #     1.0 / table.n_controls, table.n_controls)
            #     # pi2 = np.repeat(
            #     #     1.0 / table.n_controls, table.n_controls)
            #     strategy = np.random.uniform(0, 1, payoff_mat_1.shape[0])
            #     pi1 = strategy / np.sum(strategy)
            #     strategy = np.random.uniform(0, 1, payoff_mat_1.shape[0])
            #     pi2 = strategy / np.sum(strategy)

            #     pi = (pi1, pi2)

            # #return pi1,pi2  
            # # pi = game.lemke_howson(initial_dropped_label=0)
            
            # return pi[0],pi[1]