import numpy as np
import time
import itertools
import scipy.stats as stats
import matplotlib.pyplot as plt

import my_env





iiter = 20
npzfile = np.load("results_iter{}.npz".format(iiter))
Q =             npzfile['Q']
n_states_x =    npzfile['n_states_x']
n_steps_state = npzfile['n_steps_state']
n_steps_ctrl =  npzfile['n_steps_ctrl']
iters =         npzfile['iters']
Q_diff_sup =    npzfile['Q_diff_sup']
Q_diff_L2 =    npzfile['Q_diff_L2']

combi_mu = itertools.product(np.linspace(0,n_steps_state,n_steps_state+1,dtype=int), repeat=n_states_x) # all possible distributions in the discretization of the simplex
distributions_unnorm = np.asarray([el for el in combi_mu])
states_tmp = distributions_unnorm.copy()
states = states_tmp[np.where(np.sum(states_tmp, axis=1)==n_steps_state)] / float(n_steps_state)
n_states = np.shape(states)[0]
combi_ctrl = itertools.product(np.linspace(0,1,n_steps_ctrl+1), repeat=n_states_x)#n_states_x) # all possible controls as functions of state_x
controls = np.asarray([el for el in combi_ctrl]) #np.linspace(0,1,n_steps_ctrl+1)
# print("controls = {}".format(controls))
n_controls = np.shape(controls)[0]
print('MDP: n states = {}\nn controls = {}'.format(n_states, n_controls))
env = my_env.MyEnvKFPCyberSecurity()
T_tot = 20.0
env.T = 0.1
env.Nt = 1
env.Dt = env.T / env.Nt

discount_gamma = 0.5 # for one unit of time
discount_beta = - np.log(discount_gamma)

def proj_W_index(mu):
    # print("W mu = {}".format(mu))
    #print("W map = {}".format(map(lambda mu2 : stats.wasserstein_distance(mu,mu2), states)))
    # return np.argmin(map(lambda mu2 : stats.wasserstein_distance(mu,mu2), states))
    return np.argmin(map(lambda mu2 : np.sum(np.abs(mu - mu2)), states))



def get_opt_ctrl(Q_table):
    return [np.argmax(Q_table[i_mu]) for i_mu in range(n_states)]

if __name__ == '__main__':
    for test_i, mu_old in zip([1,2,3], [np.asarray([0.25, 0.25, 0.25, 0.25]), np.asarray([1.0, 0.0, 0.0, 0.0]), np.asarray([0.0, 0.0, 0.0, 1.0])]):
        print("\n\ntest_i = {}".format(test_i))
        # SOLUTION FROM Q-LEARNING
        mus = [mu_old.copy()]
        # print("Q[proj_W_index(mu_old)] = ", Q[proj_W_index(mu_old)])
        us = [np.max(Q[proj_W_index(mu_old)])]
        times = [0]
        n_epi = int(T_tot / env.T)
        Q_0 = us[0]
        accu_Q = 0
        for n in range(1,1+n_epi):
            if (n % (n_epi/10) == 0):
                print("n = ", n)
            i_mu_old = proj_W_index(mu_old)
            alpha_opt = controls[np.argmax(Q[i_mu_old])]
            mu_next, r_next = env.get_mu_and_reward(mu_old, alpha_opt)
            print("mu_next = {}, \t proj = {}, \t Q = {}".format(mu_next, states[proj_W_index(mu_next)], Q[proj_W_index(mu_next)]))
            u_next = np.max(Q[proj_W_index(mu_next)])
            mus.append(mu_next)
            us.append(u_next)
            times.append(times[-1]+env.T)
            mu_old = mu_next.copy()
            accu_Q += np.exp(-discount_beta*(n-1)*env.T) * r_next
            print("np.exp(-discount_beta*(n-1)*env.T) = {}, \t r_next = {}, \t accu_Q = {}".format(np.exp(-discount_beta*(n-1)*env.T), r_next, accu_Q))
        print("Q_0 = {}, \t accu_Q = {}".format(Q_0, accu_Q))
        mus_arr = np.asarray(mus)
        us_arr = -np.asarray(us)
        # PLOTTING DISTRIBUTION
        plt.clf()
        # BENCHMARK FROM PDE
        # npzfile = np.load('BENCHMARK-MFG/test{}_data_and_solution.npz'.format(test_i))
        # mu_pde = npzfile['mu']
        # T_pde = npzfile['T']
        # Nt_pde = npzfile['Nt']
        # Dt_pde = T_pde / Nt_pde
        # plt.plot(np.linspace(0,Nt_pde,Nt_pde+1)*Dt_pde, mu_pde[:, 0], label="mu_pde_MFG[0]", color='grey', linewidth=4)
        # plt.plot(np.linspace(0,Nt_pde,Nt_pde+1)*Dt_pde, mu_pde[:, 1], label="mu_pde_MFG[1]", color='magenta', linestyle='--', linewidth=4)
        # plt.plot(np.linspace(0,Nt_pde,Nt_pde+1)*Dt_pde, mu_pde[:, 2], label="mu_pde_MFG[2]", color='xkcd:lightgreen', linestyle=':', linewidth=4)
        # plt.plot(np.linspace(0,Nt_pde,Nt_pde+1)*Dt_pde, mu_pde[:, 3], label="mu_pde_MFG[3]", color='cyan', linestyle='-.', linewidth=4)
        # BENCHMARK FROM PDE
        npzfile = np.load('BENCHMARK-MFC/test{}_data_and_solution.npz'.format(test_i))
        mu_pde = npzfile['mu']
        T_pde = npzfile['T']
        Nt_pde = npzfile['Nt']
        Dt_pde = T_pde / Nt_pde
        plt.plot(np.linspace(0,Nt_pde,Nt_pde+1)*Dt_pde, mu_pde[:, 0], label="mu_pde_MFC[0]", color='xkcd:steel', linewidth=2)
        plt.plot(np.linspace(0,Nt_pde,Nt_pde+1)*Dt_pde, mu_pde[:, 1], label="mu_pde_MFC[1]", color='xkcd:bright magenta', linestyle='--', linewidth=2)
        plt.plot(np.linspace(0,Nt_pde,Nt_pde+1)*Dt_pde, mu_pde[:, 2], label="mu_pde_MFC[2]", color='xkcd:bright green', linestyle=':', linewidth=2)
        plt.plot(np.linspace(0,Nt_pde,Nt_pde+1)*Dt_pde, mu_pde[:, 3], label="mu_pde_MFC[3]", color='xkcd:royal blue', linestyle='-.', linewidth=2)

        plt.plot(times, mus_arr[:,0], label='mu[0]', color='black')
        plt.plot(times, mus_arr[:,1], label='mu[1]', color='red', linestyle='--')
        plt.plot(times, mus_arr[:,2], label='mu[2]', color='green', linestyle=':', linewidth=5)
        plt.plot(times, mus_arr[:,3], label='mu[3]', color='blue', linestyle='-.')
        plt.xlabel("time")
        plt.legend()
        # plt.show()
        plt.savefig("test_{}_mu.pdf".format(test_i))
        #
        # PLOTTING VALUE FUNCTION
        plt.clf()
        # BENCHMARK FROM PDE
        # npzfile = np.load('BENCHMARK-MFG/test{}_data_and_solution.npz'.format(test_i))
        # mu_pde = npzfile['mu']
        # u_pde = npzfile['u']
        # t_c_pde = npzfile['t_c']
        # T_pde = npzfile['T']
        # Nt_pde = npzfile['Nt']
        # Dt_pde = T_pde / Nt_pde
        # # plt.plot(np.linspace(0,Nt_pde,Nt_pde+1)*Dt_pde, np.sum(mu_pde*u_pde, axis=1), label="mu*u MFG", color='xkcd:orange', linewidth=4, ls="--")
        # plt.plot(np.linspace(0,Nt_pde,Nt_pde+1)*Dt_pde, t_c_pde, label="V_MFG", color='xkcd:orange', linewidth=4)
        # BENCHMARK FROM PDE
        npzfile = np.load('BENCHMARK-MFC/test{}_data_and_solution.npz'.format(test_i))
        mu_pde = npzfile['mu']
        u_pde = npzfile['u']
        t_c_pde = npzfile['t_c']
        T_pde = npzfile['T']
        Nt_pde = npzfile['Nt']
        Dt_pde = T_pde / Nt_pde
        # plt.plot(np.linspace(0,Nt_pde,Nt_pde+1)*Dt_pde, np.sum(mu_pde*u_pde, axis=1), label="mu*u MFC", color='xkcd:purple', linewidth=2, ls="--")
        plt.plot(np.linspace(0,Nt_pde,Nt_pde+1)*Dt_pde, t_c_pde, label="V_MFC", color='xkcd:purple', linewidth=2)

        plt.plot(times, us_arr, label='V_Qlearning', color='red')
        plt.xlabel("time")
        plt.legend()
        # plt.show()
        plt.savefig("test_{}_V.pdf".format(test_i))
    plt.clf()
    plt.semilogy(iters, Q_diff_sup, label='Q_diff_sup')
    plt.semilogy(iters, Q_diff_L2, label='Q_diff_L2')
    plt.xlabel('iterations')
    plt.legend()
    plt.savefig("convergenceQ_iter{}.pdf".format(iiter))
