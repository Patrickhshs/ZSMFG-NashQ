from NashQLearner import NashQPlayer
from myEnv import my1dGridEnv
from myTable import myQTable
from computeExploitability import compute_exploitability

env = my1dGridEnv()
table = myQTable()


if __name__ == '__main__':
    
    Players = NashQPlayer(env,Q_1_table=table,Q_2_table=table,MonteCarlo=True,iterations=100,
                        iter_save=25000,discount_factor=0.9)
    #print(Players.Q_1.controls[0])
    Q_predator, Q_preyer = Players.training()
    
    # Fictisiou play
    # approximate/learn te8 
    #print(Players.recover_equilibrium_policy(100,Q_predator,Q_preyer,env))
    #policy_1,policy_2,r_1,r_2 = Players.recover_equilibrium_policy(500,Q_predator,Q_preyer,env)
    #r_1_exploitability,r_2_exploitability = compute_exploitability(500,Players,Q_predator,Q_preyer,policy_1,policy_2,env)
    

