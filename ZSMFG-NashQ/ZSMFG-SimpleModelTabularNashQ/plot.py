import numpy as np
import matplotlib.pyplot as plt





npzfile = np.load("ZSMFG-SimpleModelTabularNashQ/results_iter100.npz")
Q =             npzfile['Q']
n_states_x =    npzfile['n_states_x']
n_steps_state = npzfile['n_steps_state']
n_steps_ctrl =  npzfile['n_steps_ctrl']
iters =         npzfile['iters']
Q_diff_sup =    npzfile['Q_diff_sup']
Q_diff_L2 =    npzfile['Q_diff_L2']

if __name__ =="__main__":
    plt.clf()
    plt.plot(iters, Q_diff_sup, label='Q_diff_sup')
    plt.plot(iters, Q_diff_L2, label='Q_diff_L2')
    plt.xlabel('iterations')
    plt.legend()
    plt.savefig("convergenceQ_iter100.pdf")