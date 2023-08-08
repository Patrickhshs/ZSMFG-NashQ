from NashQLearner import NashQPlayer
from myEnv import my1dGridEnv
from myTable import myQTable

env = my1dGridEnv()
table = myQTable()


if __name__ == '__main__':
    Players = NashQPlayer(env,table,MonteCarlo=False,iterations=100)
    Q_predator, Q_preyer = Players.training()
    print(Q_predator.Q_table.shape)
    print(Q_preyer.Q_table.shape)
