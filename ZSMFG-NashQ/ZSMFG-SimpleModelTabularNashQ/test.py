import nashpy as nash
import numpy as np

A = np.array([[2.97, 2.97, 2.97, 3.06, 3.06, 3.06],
 [2.97, 2.97, 2.97,  3.06, 3.06, 3.06],
 [2.97, 2.97, 2.97 ,3.06 ,3.06, 3.06]])
# A = np.array([[0,0,0],[0,0,0],[0,0,0]])
B = -A
game = nash.Game(A)
game_2 = nash.Game(B.T)
equi = game.lemke_howson_enumeration()
for i in equi:
    print(i[0])
equi_2 = game_2.lemke_howson_enumeration()
for i in equi_2:
    print(i[0])

