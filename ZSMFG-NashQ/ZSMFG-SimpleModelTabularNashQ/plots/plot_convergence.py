import numpy as np
import matplotlib.pyplot as plt





#npzfile = np.load("ZSMFG-NashQ/historyTables/Q_1_and_Q_2_results_iter1000.npz")
npzfile = np.load("ZSMFG-NashQ/historyTables/corrected/Q_MC_zeros_ecos_2action_results_iter5000.npz")
#Q =             npzfile['Q_1_diff_sup']
n_states_x =    npzfile['n_states_x']
n_steps_state = npzfile['n_steps_state']
n_steps_ctrl =  npzfile['n_steps_ctrl']
iters =         [i for i in range(1,5001)]
Q_1_diff_sup  =    np.log(npzfile['Q_1_diff_sup'])
Q_1_diff_L2  =    np.log(npzfile['Q_1_diff_L2'])
Q_2_diff_sup  =    np.log(npzfile['Q_2_diff_sup'])
Q_2_diff_L2  =    np.log(npzfile['Q_2_diff_L2'])

# Q_1_diff_sup  =    npzfile['Q_1_diff_sup']
# Q_1_diff_L2  =    npzfile['Q_1_diff_L2']
# Q_2_diff_sup  =    npzfile['Q_2_diff_sup']
# Q_2_diff_L2  =    npzfile['Q_2_diff_L2']
# Q_1 = npzfile["Q_1"]

if __name__ == "__main__":
    #print(Q_1.shape)
    plt.clf()
    # plt.plot(iters, Q_1_diff_sup, label='Q_1_diff_sup',marker="s",linestyle="--")
    # plt.plot(iters, Q_1_diff_L2, label='Q_1_diff_L2',marker="o")
    plt.plot(iters, Q_2_diff_sup, label='Q_2_diff_sup',marker="d",linestyle="--")
    plt.plot(iters, Q_2_diff_L2, label='Q_2_diff_L2',marker="x")
    plt.xlabel('iterations')
    plt.ylabel('log-scale')
    plt.legend()
    plt.title("plot of player2's convergence result")
    plt.savefig("ZSMFG-NashQ/ZSMFG-SimpleModelTabularNashQ/plots/convergenceQ2_2actions_zeros_MC_5000iters_L2.png")