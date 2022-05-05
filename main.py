from minimax import minimaxPlayer
from qtable import playGames
from qtable import playRandomMove
from qtable import (qtables, trainingAgentsX,
                    trainingAgentsO, QTablePlayer)



########
#
playRandomMove_Minimax = minimaxPlayer(True)
playMove_Minimax = minimaxPlayer(False)

print("Playing random vs random:")
print("################################")
playGames(2000, playRandomMove, playRandomMove)
print("")

print("minimax random vs minimax random:")
print("################################")
playGames(2000, playRandomMove_Minimax, playRandomMove_Minimax)
print("")

print("minimax random vs random")
print("################################")

playGames(2000, playRandomMove_Minimax, playRandomMove)
print("")
print("random vs minimax random:")
print("################################")
playGames(2000, playRandomMove, playRandomMove_Minimax)
print("")
#
print("Training -  qtable(X) vs. random")
trainingAgentsX(qTables=qtables,
                o_strategies=[playRandomMove])
print("Training - qtable(O) vs. random")
trainingAgentsO(qTables=qtables,
                x_strategies=[playRandomMove])
print(end="")

play_q_table_move = QTablePlayer(qtables)
print("Game -  qtable vs random")
print("################################")
playGames(1000, play_q_table_move, playRandomMove)
print("")

print("Game - random vs qtable")
print("###################################")
playGames(1000, playRandomMove, play_q_table_move)
print("")

print("Game - qtable vs qtable")
print("#####################################")
playGames(1000, play_q_table_move, play_q_table_move)
