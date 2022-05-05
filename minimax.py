import random
import numpy as np
import itertools


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


def play_games(total_games, x_strategy, o_strategy, play_single_game=play_game):
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

    print(f"x wins: {x_wins_percent:.2f}%")
    print(f"o wins: {o_wins_percent:.2f}%")
    print(f"draw  : {draw_percent:.2f}%")


def play_random_move(board):
    move = board.getRandomValidMoves()
    return board.play_move(move)


def is_even(value):
    return value % 2 == 0


def is_empty(values):
    return values is None or len(values) == 0



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

    def setPosition(self, board, o):
        self.cache[board.board_2d.tobytes()] = o

    def getPosition(self, board):
        board_2d = board.board_2d

        boardOrientations = getBoardOrientations(board_2d)

        for b, t in boardOrientations:
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

cache = BoardCache()


def minimaxPlayer(randomize):
    def play(board):
        return nextMove(board, randomize)

    return play

def is_empty(values):
    return values is None or len(values) == 0


def nextMove(board, randomize=False):
    move_value_pairs = getvalues(board)
    move = getBestMove(board, move_value_pairs, randomize)

    return board.play_move(move)


def getvalues(board):
    validMoves = board.get_valid_move_indexes()

    assert not is_empty(validMoves), "end position"

    moveIndexVals = [(m, positionVal(board.play_move(m)))
                        for m in validMoves]

    return moveIndexVals


def positionVal(board):
    result, found = cache.getPosition(board)
    if found:
        return result[0]

    positionValue = calcPositionValue(board)

    cache.setPosition(board, positionValue)

    return positionValue


def calcPositionValue(board):
    if board.is_gameover():
        return board.gameResult()

    valid_move_indexes = board.get_valid_move_indexes()

    values = [positionVal(board.play_move(m))
              for m in valid_move_indexes]

    minMax = chooseMinOrMax(board)
    position_value = minMax(values)

    return position_value


def getBestMove(board, move_value_pairs, randomize):
    min_or_max = chooseMinOrMax(board)
    move, value = min_or_max(move_value_pairs, key=lambda mvp: mvp[1])
    if not randomize:
        return move

    bestmoveIndexVals = [mvp for mvp in move_value_pairs
                             if mvp[1] == value]
    chosenMove, v = random.choice(bestmoveIndexVals)
    return chosenMove


def chooseMinOrMax(board):
    turn = board.get_turn()
    return min if turn == CELL_O else max
