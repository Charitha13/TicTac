# TicTacToe
CSCI 6660 - Final Project
#Run - 
Run main.py and it will  run various scenarios of minimax and QLearning implemenattion of TicTacToe Game
minimax.py has implementation of tictactoe using minimax algorithm.
qtable.py has implementation of tictactoe using qlearning algorithm.
It uses double Q tables to store the Q values.
epsilon value starts from 0.9 and it keeps decreasing for every 10% values and trains the agents.
It basically trains starting with exploration method to exploitation method using the Q tables.
