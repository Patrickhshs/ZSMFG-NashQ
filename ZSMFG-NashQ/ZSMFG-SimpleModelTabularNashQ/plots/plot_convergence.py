import numpy as np
import matplotlib.pyplot as plt





#npzfile = np.load("ZSMFG-NashQ/historyTables/Q_1_and_Q_2_results_iter1000.npz")
npzfile = np.load("ZSMFG-NashQ/historyTables/Q_MC_zeros_ecos_results_iter50000.npz")
#Q =             npzfile['Q_1_diff_sup']
n_states_x =    npzfile['n_states_x']
n_steps_state = npzfile['n_steps_state']
n_steps_ctrl =  npzfile['n_steps_ctrl']
iters =         [i for i in range(1,50001)]
# Q_1_diff_sup  =    np.log(npzfile['Q_1_diff_sup'])
# Q_1_diff_L2  =    np.log(npzfile['Q_1_diff_L2'])
# Q_2_diff_sup  =    np.log(npzfile['Q_2_diff_sup'])
# Q_2_diff_L2  =    np.log(npzfile['Q_2_diff_L2'])

Q_1_diff_sup  =    npzfile['Q_1_diff_sup']
Q_1_diff_L2  =    npzfile['Q_1_diff_L2']
Q_2_diff_sup  =    npzfile['Q_2_diff_sup']
Q_2_diff_L2  =    npzfile['Q_2_diff_L2']

if __name__ == "__main__":
    plt.clf()
    #plt.plot(iters, Q_1_diff_sup, label='Q_1_diff_sup')
    plt.plot(iters, Q_1_diff_L2, label='Q_1_diff_L2',color="orange")
    #plt.plot(iters, Q_2_diff_sup, label='Q_2_diff_sup',linestyle="--")
    # plt.plot(iters, Q_2_diff_L2, label='Q_2_diff_L2')
    plt.xlabel('iterations')
    plt.legend()
    plt.savefig("ZSMFG-NashQ/ZSMFG-SimpleModelTabularNashQ/plots/convergenceQ1_zeros_withMC_50000iters_L2.pdf")