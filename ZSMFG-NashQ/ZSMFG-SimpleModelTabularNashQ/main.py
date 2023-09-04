from NashQLearner import NashQPlayer
from myEnv import my1dGridEnv
from myTable import myQTable
from computeExploitability import compute_exploitability
import numpy as np

env = my1dGridEnv()
file = np.load("ZSMFG-NashQ/historyTables/Q_noMC_zeros_ecos_results_iter100.npz")
Q_1 = file["Q_1"]
Q_2 = file["Q_2"]



if __name__ == '__main__':
    
    Players = NashQPlayer(env,Q_1_table=myQTable(),Q_2_table=myQTable(),MonteCarlo=True,iterations=5000,max_episode_steps=10,
                        iter_save=5000,discount_factor=0.98)
    #print(Players.Q_1.controls[0])
    Q_predator, Q_preyer = Players.training()
    
    # Fictisiou play
    # approximate/learn 
    #print(Players.recover_equilibrium_policy(100,Q_predator,Q_preyer,env))
    #policy_1,policy_2,r_1,r_2 = Players.recover_equilibrium_policy(500,Q_predator,Q_preyer,env)
    #r_1_exploitability,r_2_exploitability = compute_exploitability(500,Players,Q_predator,Q_preyer,policy_1,policy_2,env)
    

