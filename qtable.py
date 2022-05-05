import statistics as stats
import operator
from collections import deque
import random
import itertools
import numpy as np


class Transform:
    def __init__(self, *operations):
        self.operations = operations

    def transform(self, target):
        for op in self.operations:
            target = op.transform(target)
        return target

    def reverse(self, target):
        for op in reverse(self.operations):
            target = op.reverse(target)
        return target


class Identity:
    @staticmethod
    def transform(matrix2d):
        return matrix2d

    @staticmethod
    def reverse(matrix2d):
        return matrix2d


class Rotate:
    def __init__(self, number_of_rotations):
        self.number_of_rotations = number_of_rotations
        self.op = np.rot90

    def transform(self, matrix2d):
        return self.op(matrix2d, self.number_of_rotations)

    def reverse(self, transformed_matrix2d):
        return self.op(transformed_matrix2d, -self.number_of_rotations)


class Flip:
    def __init__(self, op):
        self.op = op

    def transform(self, matrix2d):
        return self.op(matrix2d)

    def reverse(self, transformed_matrix2d):
        return self.transform(transformed_matrix2d)


def reverse(items):
    return items[::-1]



TRANSFORMATIONS = [Identity(), Rotate(1), Rotate(2), Rotate(3),
                   Flip(np.flipud), Flip(np.fliplr),
                   Transform(Rotate(1), Flip(np.flipud)),
                   Transform(Rotate(1), Flip(np.fliplr))]

BOARD_SIZE = 3
BOARD_DIMENSIONS = (BOARD_SIZE, BOARD_SIZE)

CELL_X = 1
CELL_O = -1
CELL_EMPTY = 0

RESULT_X_WINS = 1
RESULT_O_WINS = -1
RESULT_DRAW = 0
RESULT_NOT_OVER = 2

new_board = np.array([CELL_EMPTY] * BOARD_SIZE ** 2)


def play_game(x_strategy, o_strategy):
    board = Board()
    player_strategies = itertools.cycle([x_strategy, o_strategy])

    while not board.is_gameover():
        play = next(player_strategies)
        board = play(board)

    return board


def playGames(total_games, x_strategy, o_strategy, play_single_game=play_game):
    results = {
        RESULT_X_WINS: 0,
        RESULT_O_WINS: 0,
        RESULT_DRAW: 0
    }

    for g in range(total_games):
        end_of_game = (play_single_game(x_strategy, o_strategy))
        result = end_of_game.gameResult()
        results[result] += 1

    x_wins_percent = results[RESULT_X_WINS] / total_games * 100
    o_wins_percent = results[RESULT_O_WINS] / total_games * 100
    draw_percent = results[RESULT_DRAW] / total_games * 100

    print(f"x wins: {x_wins_percent:.3f}%")
    print(f"o wins: {o_wins_percent:.3f}%")
    print(f"game draw  : {draw_percent:.3f}%")


def playRandomMove(board):
    move = board.getRandomValidMoves()
    return board.play_move(move)


def is_even(value):
    return value % 2 == 0


def is_empty(values):
    return values is None or len(values) == 0



WIN_VALUE = 1.0
DRAW_VALUE = 1.0
LOSS_VALUE = 0.0

INITIAL_Q_VALUES_FOR_X = 0.0
INITIAL_Q_VALUES_FOR_O = 0.0


class Board:
    def __init__(self, board=None, illegal_move=None):
        if board is None:
            self.board = np.copy(new_board)
        else:
            self.board = board

        self.illegal_move = illegal_move

        self.board_2d = self.board.reshape(BOARD_DIMENSIONS)

    def gameResult(self):
        if self.illegal_move is not None:
            return RESULT_O_WINS if self.get_turn() == CELL_X else RESULT_X_WINS

        rows_cols_and_diagonals = getRowColDiagonals(self.board_2d)

        sums = list(map(sum, rows_cols_and_diagonals))
        max_value = max(sums)
        min_value = min(sums)

        if max_value == BOARD_SIZE:
            return RESULT_X_WINS

        if min_value == -BOARD_SIZE:
            return RESULT_O_WINS

        if CELL_EMPTY not in self.board_2d:
            return RESULT_DRAW

        return RESULT_NOT_OVER

    def is_gameover(self):
        return self.gameResult() != RESULT_NOT_OVER

    def is_in_illegal_state(self):
        return self.illegal_move is not None

    def play_move(self, move_index):
        board_copy = np.copy(self.board)

        if move_index not in self.get_valid_move_indexes():
            return Board(board_copy, illegal_move=move_index)

        board_copy[move_index] = self.get_turn()
        return Board(board_copy)

    def get_turn(self):
        non_zero = np.count_nonzero(self.board)
        return CELL_X if is_even(non_zero) else CELL_O

    def get_valid_move_indexes(self):
        return ([i for i in range(self.board.size)
                 if self.board[i] == CELL_EMPTY])

    def getNotValidIndexesMove(self):
        return ([i for i in range(self.board.size)
                if self.board[i] != CELL_EMPTY])

    def getRandomValidMoves(self):
        return random.choice(self.get_valid_move_indexes())



class BoardCache:
    def __init__(self):
        self.cache = {}

    def set_for_position(self, board, o):
        self.cache[board.board_2d.tobytes()] = o

    def get_for_position(self, board):
        board_2d = board.board_2d

        orientations = getBoardOrientations(board_2d)

        for b, t in orientations:
            result = self.cache.get(b.tobytes())
            if result is not None:
                return (result, t), True

        return None, False

    def reset(self):
        self.cache = {}


def getBoardOrientations(board_2d):
    return [(t.transform(board_2d), t) for t in TRANSFORMATIONS]


def getRowColDiagonals(board_2d):
    rows_and_diagonal = getRowsAndDiagonal(board_2d)
    cols_and_antidiagonal = getRowsAndDiagonal(np.rot90(board_2d))
    return rows_and_diagonal + cols_and_antidiagonal


def getRowsAndDiagonal(board_2d):
    num_rows = board_2d.shape[0]
    return ([row for row in board_2d[range(num_rows), :]]
            + [board_2d.diagonal()])


def get_symbol(cell):
    if cell == CELL_X:
        return 'X'
    if cell == CELL_O:
        return 'O'
    return '-'


def isDraw(board):
    return board.gameResult() == RESULT_DRAW


class QTable:
    def __init__(self):
        self.qtable = BoardCache()

    def getQValues(self, board):
        move_indexes = board.get_valid_move_indexes()
        qvalues = [self.getQValue(board, mi) for mi
                   in board.get_valid_move_indexes()]

        return dict(zip(move_indexes, qvalues))

    def getQValue(self, board, move_index):
        new_position = board.play_move(move_index)
        result, found = self.qtable.get_for_position(new_position)
        if found is True:
            qvalue, h = result
            return qvalue

        return getInitialQvalue(new_position)

    def updateQVal(self, board, move_index, qvalue):
        new_position = board.play_move(move_index)

        result, found = self.qtable.get_for_position(new_position)
        if found is True:
            h, t = result
            new_position_transformed = Board(
                t.transform(new_position.board_2d).flatten())
            self.qtable.set_for_position(new_position_transformed, qvalue)
            return

        self.qtable.set_for_position(new_position, qvalue)

    def getIndexAndQVals(self, board):
        q_values = self.getQValues(board)
        return max(q_values.items(), key=operator.itemgetter(1))



def getInitialQvalue(board):
    return (INITIAL_Q_VALUES_FOR_X if board.get_turn() == CELL_O
            else INITIAL_Q_VALUES_FOR_O)


qtables = [QTable()]

double_qtables = [QTable(), QTable()]


def QTablePlayer(q_tables):
    def play(board):
        return getNextMove(board, q_tables)

    return play


def getNextMove(board, q_tables=None):
    if q_tables is None:
        q_tables = qtables

    move_index = choodeNextIndex(q_tables, board, 0)
    return board.play_move(move_index)


def choodeNextIndex(q_tables, board, epsilon):
    if epsilon > 0:
        random_value_from_0_to_1 = np.random.uniform()
        if random_value_from_0_to_1 < epsilon:
            return board.getRandomValidMoves()

    moveQVal = getAvgOfQVals(q_tables, board)

    return max(moveQVal, key=lambda pair: pair[1])[0]


def getAvgOfQVals(q_tables, board):
    moveIndexes = sorted(q_tables[0].getQValues(board).keys())

    avgQVal = [stats.mean(getQValsToMove(q_tables, board, mi))
               for mi in moveIndexes]

    return list(zip(moveIndexes, avgQVal))


def getQValsToMove(qTables, board, move_index):
    return [q_table.getQValue(board, move_index) for q_table in qTables]


def trainingAgentsX(total_games=9000, qTables=None,
                    learning_rate=0.5, discount_factor=1.0, epsilon=0.9,
                    o_strategies=None):
    if qTables is None:
        qTables = qtables
    if o_strategies is None:
        o_strategies = [playRandomMove]

    trainingAgents(total_games, qTables, CELL_X, learning_rate,
                   discount_factor, epsilon, None, o_strategies)


def trainingAgentsO(total_games=9000, qTables=None,
                    learning_rate=0.5, discount_factor=1.0, epsilon=0.9,
                    x_strategies=None):
    if qTables is None:
        qTables = qtables
    if x_strategies is None:
        x_strategies = [playRandomMove]

    trainingAgents(total_games, qTables, CELL_O, learning_rate,
                   discount_factor, epsilon, x_strategies, None)


def trainingAgents(total_games, q_tables, q_table_player, learning_rate,
                   discount_factor, epsilon, x_strategies, o_strategies):
    if x_strategies:
        x_strategies_to_use = itertools.cycle(x_strategies)

    if o_strategies:
        o_strategies_to_use = itertools.cycle(o_strategies)

    for game in range(total_games):
        move_history = deque()

        if not x_strategies:
            x = [markPlayer(q_tables, move_history, epsilon)]
            x_strategies_to_use = itertools.cycle(x)

        if not o_strategies:
            o = [markPlayer(q_tables, move_history, epsilon)]
            o_strategies_to_use = itertools.cycle(o)

        x_strategy_to_use = next(x_strategies_to_use)
        o_strategy_to_use = next(o_strategies_to_use)

        trainGameAgents(q_tables, move_history, q_table_player,
                        x_strategy_to_use, o_strategy_to_use, learning_rate,
                        discount_factor)

        if (game+1) % (total_games / 10) == 0:
            epsilon = max(0, epsilon - 0.1)
            print(f"{game+1}/{total_games} games, using epsilon={epsilon}")


def trainGameAgents(q_tables, move_history, q_table_player, x_strategy,
                    o_strategy, learning_rate, discount_factor):
    board = play_game(x_strategy, o_strategy)

    updateGameover(q_tables, move_history, q_table_player, board,
                   learning_rate, discount_factor)


def updateGameover(qTables, move_history, q_table_player, board,
                   learning_rate, discount_factor):
    game_result_reward = findResult(q_table_player, board)

    # move history is in reverse-chronological order - last to first
    next_position, move_index = move_history[0]
    for qTable in qTables:
        current_q_value = qTable.getQValue(next_position, move_index)
        new_q_value = newQValueCalc(current_q_value, game_result_reward,
                                    0.0, learning_rate, discount_factor)

        qTable.updateQVal(next_position, move_index, new_q_value)

    for (position, move_index) in list(move_history)[1:]:
        current_q_table, next_q_table = getRandomQTables(qTables)

        max_next_move_index, _ = current_q_table.getIndexAndQVals(
            next_position)

        max_next_q_value = next_q_table.getQValue(next_position,
                                                    max_next_move_index)

        current_q_value = current_q_table.getQValue(position, move_index)
        new_q_value = newQValueCalc(current_q_value, 0.0,
                                    max_next_q_value, learning_rate,
                                    discount_factor)
        current_q_table.updateQVal(position, move_index, new_q_value)

        next_position = position


def newQValueCalc(current_q_value, reward, max_next_q_value,
                  learning_rate, discount_factor):
    weighted_prior_values = (1 - learning_rate) * current_q_value
    weighted_new_value = (learning_rate
                          * (reward + discount_factor * max_next_q_value))
    return weighted_prior_values + weighted_new_value


def getRandomQTables(q_tables):
    q_tables_copy = q_tables.copy()
    random.shuffle(q_tables_copy)
    q_tables_cycle = itertools.cycle(q_tables_copy)

    current_q_table = next(q_tables_cycle)
    next_q_table = next(q_tables_cycle)

    return current_q_table, next_q_table


def markPlayer(q_tables, move_history, epsilon):
    def play(board):
        move_index = choodeNextIndex(q_tables, board, epsilon)
        move_history.appendleft((board, move_index))
        return board.play_move(move_index)

    return play


def findResult(player, board):
    if hasWon(player, board):
        return WIN_VALUE
    if hasLost(player, board):
        return LOSS_VALUE
    if isDraw(board):
        return DRAW_VALUE


def hasWon(player, board):
    result = board.gameResult()
    return ((player == CELL_O and result == RESULT_O_WINS)
            or (player == CELL_X and result == RESULT_X_WINS))


def hasLost(player, board):
    result = board.gameResult()
    return ((player == CELL_O and result == RESULT_X_WINS)
            or (player == CELL_X and result == RESULT_O_WINS))
