import numpy as np

from myEnv import my1dGridEnv

from myTable import myQTable

from tqdm import tqdm



#npzfile = np.load("Tabular Q-learning/test10f7a3_paramsBook_from-v3d5_10Pts_gamma0p5_envT1_cont20Pts_contT0p1_cont30pts_contT0p2/test10f7_results_iter10.npz")
#Q_prev =      npzfile['Q']


lr = 0.5

env = my1dGridEnv()
table = myQTable()
table.init_states() #initialize the states
table.init_ctrl() #initialize the controls

antagonist_table = myQTable()
antagonist_table.init_states()
antagonist_table.init_ctrl()
#Q_old should be table.Q_old if there is no previous Q-table

Q_old_anta = antagonist_table.Q_table
Q_old = table.Q_table





iter_save = 20

discount_gamma = 0.5 # for one unit of time
discount_beta = - np.log(discount_gamma)
discount = np.exp(-discount_beta * env.T)
print("discount = {}".format(discount))
print(table.controls)        



N_episodes = 20



iters = []
Q_diff_sup = []
Q_diff_L2 = []
if __name__ == '__main__':
    for i in tqdm(range(1, 1+N_episodes)):
        print('\n\n======================================== Episode {}\n\n'.format(i))
        Q_new = (1-lr) * Q_old.copy()#First part of Q_new
        Q_new_anta = (1-lr) * Q_old_anta.copy()
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
                    pi_1,pi_2 = env.get_nash_Q_value(Q_old[i_mu_1_next],Q_old_anta[i_mu_2_next])
                     
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
                    

        # print("np.abs(Q_new - Q_old) = ", np.abs(Q_new - Q_old))

        # Calculate the Q_diff_sup and Q_diff_L2 to see if converges
        
        Q_diff_sup.append(np.max(np.abs(Q_new - Q_old)))
        #
        print("***** sup|Q_new - Q_old| = {}".format(Q_diff_sup[-1]))
        Q_diff_L2.append(np.sqrt(np.sum(np.square(Q_new - Q_old))))
        print("***** L2|Q_new - Q_old| = {}\n".format(Q_diff_L2[-1]))
        # print("Q_new = {}".format(Q_new))
        #opt_ctrls = table.get_opt_ctrl(Q_new)
        # print("***** opt_ctrls = {}".format(opt_ctrls))
        Q_old = Q_new.copy()
        Q_old_anta = Q_new_anta.copy()
        if (i % iter_save == 0):
            np.savez("historyTables/results_iter{}".format(i), Q=Q_new, n_states_x=table.n_states_x, n_steps_state=table.n_steps_state, n_steps_ctrl=table.n_steps_ctrl, iters=iters, Q_diff_sup=Q_diff_sup, Q_diff_L2=Q_diff_L2)