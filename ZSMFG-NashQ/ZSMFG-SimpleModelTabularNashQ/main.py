from NashQLearner import NashQPlayer
from myEnv import my1dGridEnv
from myTable import myQTable

env = my1dGridEnv()
table = myQTable()


if __name__ == '__main__':
    Players = NashQPlayer(env,table,MonteCarlo=True,iterations=50)
    Q_predator, Q_preyer = Players.training()
    # Fictisiou play
    # approximate/learn te8 
    #print(Players.recover_equilibrium_policy(100,Q_predator,Q_preyer,env))
    policy_1,policy_2 = Players.recover_equilibrium_policy(100,Q_predator,Q_preyer,env)
    

