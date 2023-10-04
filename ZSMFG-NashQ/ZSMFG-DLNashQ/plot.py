import numpy as np
import matplotlib.pyplot as plt





#npzfile = np.load("ZSMFG-NashQ/historyTables/Q_1_and_Q_2_results_iter1000.npz")
npzfile = np.load("ZSMFG-NashQ/ZSMFG-DLNashQ/nashDQNmodels/training_loss.npz")
#Q =             npzfile['Q_1_diff_sup']
loss_1 =    np.log(npzfile['player_1_loss'])
loss_2 =    np.log(npzfile['player_2_loss'])


iters =         [i for i in range(len(loss_1))]


# Q_1_diff_sup  =    npzfile['Q_1_diff_sup']
# Q_1_diff_L2  =    npzfile['Q_1_diff_L2']
# Q_2_diff_sup  =    npzfile['Q_2_diff_sup']
# Q_2_diff_L2  =    npzfile['Q_2_diff_L2']
# Q_1 = npzfile["Q_1"]

if __name__ == "__main__":
    #print(Q_1.shape)

    plt.clf()
    #plt.plot(iters, loss_1, label='Q_1_diff_sup',marker="s",linestyle="--")
    plt.plot(iters, loss_2, label='Player_2_training_loss')#,marker="o")
    # plt.plot(iters, Q_2_diff_sup, label='Q_2_diff_sup',marker="d",linestyle="--")
    # plt.plot(iters, Q_2_diff_L2, label='Q_2_diff_L2',marker="x")
    plt.xlabel('iterations')
    plt.ylabel('log-scale')
    plt.legend()
    plt.title("plot of player 2's convergence result")
    plt.savefig("ZSMFG-NashQ/ZSMFG-DLNashQ/plots/loss_2.pdf")