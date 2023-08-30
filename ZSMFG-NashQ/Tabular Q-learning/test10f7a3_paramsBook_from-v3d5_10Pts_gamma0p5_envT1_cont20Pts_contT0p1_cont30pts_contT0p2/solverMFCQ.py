import numpy as np
import time
import itertools
import scipy
import scipy.stats as stats

import my_env

from tableMFCQ import MyTable




#### LOAD PREVIOUS
npzfile = np.load("ZSMFG-NashQ/Tabular Q-learning/test10f7a3_paramsBook_from-v3d5_10Pts_gamma0p5_envT1_cont20Pts_contT0p1_cont30pts_contT0p2/test10f_results_iter100.npz")
Q_prev =             npzfile['Q']
# n_states_x_prev =    npzfile['n_states_x']
# n_steps_state_prev = npzfile['n_steps_state']
# n_steps_ctrl_prev =  npzfile['n_steps_ctrl']

# NEW



lr = 1.0


# INITIALIZE WITH PREVIOUS MATRIX
# for i_mu in range(n_states):
#     mu = states[i_mu]
#     i_mu_proj = np.argmin(map(lambda mu2 : np.sum(np.abs(mu - mu2)), states_prev)) # proj on previous set of states
#     Q_old[i_mu] = Q_prev[i_mu_proj]
env = my_env.MyEnvKFPCyberSecurity()
table = MyTable(n_states_x = 4,n_steps_state = 30,environment=env)
table.init_states()#initialize the states
table.init_ctrl()#initialize the controls

#Q_old should be table.Q_old if there is no previous Q-table

Q_old = Q_prev.copy()
print(np.shape(Q_old))



iter_save = 50

discount_gamma = 0.5 # for one unit of time
discount_beta = - np.log(discount_gamma)
discount = np.exp(-discount_beta * env.T)
print("discount = {}".format(discount))



N_episodes = 1000



iters = []
Q_diff_sup = []
Q_diff_L2 = []
if __name__ == '__main__':
    for i in range(1, 1+N_episodes):
        print('\n\n======================================== Episode {}\n\n'.format(i))
        Q_new = (1-lr) * Q_old.copy()#First part of Q_new
        for i_mu in range(table.n_states):
            # print("i_mu = {}\n".format(i_mu))

            mu = table.states[i_mu]
            for i_alpha in range(table.n_controls):
                # print("i_alpha = {}\n".format(i_alpha))
                alpha = table.controls[i_alpha]
                mu_next, r_next = env.get_mu_and_reward(mu, alpha)#get the new mu
                i_mu_next = table.proj_W_index(mu_next)#find its most nearest mu

                # print("mu = {},\t mu_next = {}, \t mu_next_proj = {}".format(mu, mu_next, states[i_mu_next]))

                Q_opt = np.max(Q_old[i_mu_next]) #select the largest Q value
                
                #update the New Q table
                Q_new[i_mu, i_alpha] += lr * (r_next + discount * Q_opt)

        # print("np.abs(Q_new - Q_old) = ", np.abs(Q_new - Q_old))

        #Calculate the Q_diff_sup and Q_diff_L2 to see if converges
        iters.append(i)
        Q_diff_sup.append(np.max(np.abs(Q_new - Q_old)))
        print("***** sup|Q_new - Q_old| = {}".format(Q_diff_sup[-1]))
        Q_diff_L2.append(np.sqrt(np.sum(np.square(Q_new - Q_old))))
        print("***** L2|Q_new - Q_old| = {}\n".format(Q_diff_L2[-1]))
        # print("Q_new = {}".format(Q_new))
        opt_ctrls = table.get_opt_ctrl(Q_new)
        # print("***** opt_ctrls = {}".format(opt_ctrls))
        Q_old = Q_new.copy()
        if (i % iter_save == 0):
            np.savez("results_iter{}".format(i), Q=Q_new, n_states_x=table.n_states_x, n_steps_state=table.n_steps_state, n_steps_ctrl=table.n_steps_ctrl, iters=iters, Q_diff_sup=Q_diff_sup, Q_diff_L2=Q_diff_L2)
