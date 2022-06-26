import sys
import random
import timeit
import math
import argparse
from collections import Counter
from copy import deepcopy


class GO:
    def __init__(self, n):
        """
        Go game.

        :param n: size of the board n*n
        """
        self.size = n
        # self.previous_board = None # Store the previous board
        self.X_move = True  # X chess plays first
        self.died_pieces = []  # Intialize died pieces to be empty
        self.n_move = 0  # Trace the number of moves
        self.max_move = n * n - 1  # The max movement of a Go game
        self.komi = n / 2  # Komi rule
        self.verbose = False  # Verbose only when there is a manual player

    def init_board(self, n):
        '''
        Initialize a board with size n*n.

        :param n: width and height of the board.
        :return: None.
        '''
        board = [[0 for x in range(n)] for y in range(n)]  # Empty space marked as 0
        # 'X' pieces marked as 1
        # 'O' pieces marked as 2
        self.board = board
        self.previous_board = deepcopy(board)

    def set_board(self, piece_type, previous_board, board):
        '''
        Initialize board status.
        :param previous_board: previous board state.
        :param board: current board state.
        :return: None.
        '''

        # 'X' pieces marked as 1
        # 'O' pieces marked as 2

        for i in range(self.size):
            for j in range(self.size):
                if previous_board[i][j] == piece_type and board[i][j] != piece_type:
                    self.died_pieces.append((i, j))

        # self.piece_type = piece_type
        self.previous_board = previous_board
        self.board = board

    def compare_board(self, board1, board2):
        for i in range(self.size):
            for j in range(self.size):
                if board1[i][j] != board2[i][j]:
                    return False
        return True

    def copy_board(self):
        '''
        Copy the current board for potential testing.

        :param: None.
        :return: the copied board instance.
        '''
        return deepcopy(self)

    def detect_neighbor(self, i, j):
        '''
        Detect all the neighbors of a given stone.

        :param i: row number of the board.
        :param j: column number of the board.
        :return: a list containing the neighbors row and column (row, column) of position (i, j).
        '''
        board = self.board
        neighbors = []
        # Detect borders and add neighbor coordinates
        if i > 0: neighbors.append((i - 1, j))
        if i < len(board) - 1: neighbors.append((i + 1, j))
        if j > 0: neighbors.append((i, j - 1))
        if j < len(board) - 1: neighbors.append((i, j + 1))
        return neighbors

    def detect_neighbor_ally(self, i, j):
        '''
        Detect the neighbor allies of a given stone.

        :param i: row number of the board.
        :param j: column number of the board.
        :return: a list containing the neighbored allies row and column (row, column) of position (i, j).
        '''
        board = self.board
        neighbors = self.detect_neighbor(i, j)  # Detect neighbors
        group_allies = []
        # Iterate through neighbors
        for piece in neighbors:
            # Add to allies list if having the same color
            if board[piece[0]][piece[1]] == board[i][j]:
                group_allies.append(piece)
        return group_allies

    def ally_dfs(self, i, j):
        '''
        Using DFS to search for all allies of a given stone.

        :param i: row number of the board.
        :param j: column number of the board.
        :return: a list containing the all allies row and column (row, column) of position (i, j).
        '''
        stack = [(i, j)]  # stack for DFS search
        ally_members = []  # record allies positions during the search
        while stack:
            piece = stack.pop()
            ally_members.append(piece)
            neighbor_allies = self.detect_neighbor_ally(piece[0], piece[1])
            for ally in neighbor_allies:
                if ally not in stack and ally not in ally_members:
                    stack.append(ally)
        return ally_members

    def find_liberty(self, i, j):
        '''
        Find liberty of a given stone. If a group of allied stones has no liberty, they all die.

        :param i: row number of the board.
        :param j: column number of the board.
        :return: boolean indicating whether the given stone still has liberty.
        '''
        total_liberty = 0
        board = self.board
        ally_members = self.ally_dfs(i, j)
        for member in ally_members:
            neighbors = self.detect_neighbor(member[0], member[1])
            for piece in neighbors:
                # If there is empty space around a piece, it has liberty
                if board[piece[0]][piece[1]] == 0:
                    total_liberty += 1
        # If none of the pieces in a allied group has an empty space, it has no liberty
        return total_liberty

    def heuristic(self, piece_type):
        board = self.board
        my_pieces, opponent_pieces, my_heur, opp_heur = 0, 0, 0, 0
        for i in range(len(board)):
            for j in range(len(board)):
                if board[i][j] == piece_type:
                    my_pieces += 1
                    my_heur += (my_pieces + self.find_liberty(i, j))
                elif board[i][j] == 3 - piece_type:
                    opponent_pieces += 1
                    opp_heur += (opponent_pieces + self.find_liberty(i, j))

        return my_heur - opp_heur

    def find_died_pieces(self, piece_type):
        '''
        Find the died stones that has no liberty in the board for a given piece type.

        :param piece_type: 1('X') or 2('O').
        :return: a list containing the dead pieces row and column(row, column).
        '''
        board = self.board
        died_pieces = []

        for i in range(len(board)):
            for j in range(len(board)):
                # Check if there is a piece at this position:
                if board[i][j] == piece_type:
                    # The piece die if it has no liberty
                    if self.find_liberty(i, j) == 0:
                        died_pieces.append((i, j))
        return died_pieces

    def remove_died_pieces(self, piece_type):
        '''
        Remove the dead stones in the board.

        :param piece_type: 1('X') or 2('O').
        :return: locations of dead pieces.
        '''

        died_pieces = self.find_died_pieces(piece_type)
        if not died_pieces: return []
        self.remove_certain_pieces(died_pieces)
        return died_pieces

    def remove_certain_pieces(self, positions):
        '''
        Remove the stones of certain locations.

        :param positions: a list containing the pieces to be removed row and column(row, column)
        :return: None.
        '''
        board = self.board
        for piece in positions:
            board[piece[0]][piece[1]] = 0
        self.update_board(board)

    def valid_place_check(self, i, j, piece_type):
        '''
        Check whether a placement is valid.

        :param i: row number of the board.
        :param j: column number of the board.
        :param piece_type: 1(white piece) or 2(black piece).
        :return: boolean indicating whether the placement is valid.
        '''
        board = self.board

        # Check if the place is in the board range
        if not (i >= 0 and i < len(board)):
            return False
        if not (j >= 0 and j < len(board)):
            return False

        # Check if the place already has a piece
        if board[i][j] != 0:
            return False

        # Copy the board for testing
        test_go = self.copy_board()
        test_board = test_go.board

        # Check if the place has liberty
        test_board[i][j] = piece_type
        test_go.update_board(test_board)

        # Check if move is unnecessary, that is surrounded by friendly_pieces
        surround = 0
        if i > 0:
            if test_go.board[i - 1][j] == piece_type:
                surround += 1
        else:
            surround += 1
        if i < len(board) - 1:
            if test_go.board[i + 1][j] == piece_type:
                surround += 1
        else:
            surround += 1

        if j > 0:
            if test_go.board[i][j - 1] == piece_type:
                surround += 1
        else:
            surround += 1
        if j < len(board) - 1:
            if test_go.board[i][j + 1] == piece_type:
                surround += 1
        else:
            surround += 1

        if surround == 4:
            return False

        if test_go.find_liberty(i, j) > 0:
            return True

        # If not, remove the died pieces of opponent and check again
        test_go.remove_died_pieces(3 - piece_type)
        if test_go.find_liberty(i, j) == 0:
            return False
        # Check special case: repeat placement causing the repeat board state (KO rule)
        else:
            if self.died_pieces and self.compare_board(self.previous_board, test_go.board):
                return False

        return True

    def all_valid_moves(self, piece_type):
        board = self.board
        valid_moves = list()
        for i in range(len(board)):
            for j in range(len(board)):
                if self.valid_place_check(i, j, piece_type):
                    valid_moves.append((i, j))

        return valid_moves

    def findEmptyPlaces(self):
        board = self.board
        empty = []
        for i in range(len(board)):
            for j in range(len(board)):
                if board[i][j] == 0:
                    empty.append((i, j))

        return empty

    def findAggressiveMove(self):
        board = self.board
        empty_spaces = self.findEmptyPlaces()
        move_to_num_died_pieces = dict()
        for move in empty_spaces:
            board[move[0]][move[1]] = piece_type
            died_pieces = self.find_died_pieces(3 - piece_type)
            board[move[0]][move[1]] = 0
            if len(died_pieces) >= 1:
                move_to_num_died_pieces[move] = len(died_pieces)

        sorted_by_num_died_pieces = sorted(move_to_num_died_pieces, key=move_to_num_died_pieces.get, reverse=True)

        for move in sorted_by_num_died_pieces:
            test_go = self.copy_board()
            test_board = test_go.board
            test_board[move[0]][move[1]] = piece_type
            test_go.update_board(test_board)
            test_go.remove_died_pieces(3 - piece_type)
            if move is not None and not self.compare_board(self.previous_board, test_go.board):
                return move

    def filterUsefulMoves(self, possible_moves):
        board = self.board
        harmful_moves = []
        for move in possible_moves:
            board[move[0]][move[1]] = piece_type
            opponent_moves = self.all_valid_moves(3 - piece_type)
            for opp_move in opponent_moves:
                board[opp_move[0]][opp_move[1]] = 3 - piece_type
                died_pieces = self.find_died_pieces(piece_type)
                board[opp_move[0]][opp_move[1]] = 0
                if move in died_pieces:
                    harmful_moves.append(move)
            board[move[0]][move[1]] = 0

        for hm in harmful_moves:
            if hm in possible_moves:
                possible_moves.remove(hm)

        return possible_moves

    def findBestMove(self, piece_type):
        # Aggressive Phase
        aggressive_move = self.findAggressiveMove()

        if aggressive_move:
            return aggressive_move

        # If can't be aggressive, prepare for proper defense
        possible_moves = self.all_valid_moves(piece_type)
        filtered_possible_moves = self.filterUsefulMoves(possible_moves)

        if not filtered_possible_moves:
            return ['PASS']

        # Defensive Move


        best_alpha_beta_moves, alpha_beta_max_value = self.alphaBeta(2, piece_type)
        if best_alpha_beta_moves and alpha_beta_max_value >= 1000:
            return random.choice(best_alpha_beta_moves)

        if len(possible_moves) >= 15:
            if (2, 2) in possible_moves:
                return 2, 2
            if (1, 1) in possible_moves:
                return 1, 1
            if (1, 3) in possible_moves:
                return 1, 3
            if (3, 1) in possible_moves:
                return 3, 1
            if (3, 3) in possible_moves:
                return 3, 3
            if (2, 0) in possible_moves:
                return 2, 0
            if (2, 4) in possible_moves:
                return 2, 4
            if (0, 2) in possible_moves:
                return 0, 2
            if (4, 2) in possible_moves:
                return 4, 2

        return random.choice(best_alpha_beta_moves)

    def alphaBeta(self, max_depth, piece_type):
        alpha = -math.inf
        beta = math.inf
        moves = list()
        max_value = 0

        for move in self.all_valid_moves(piece_type):
            test_go = self.copy_board()
            test_go.previous_board = deepcopy(test_go.board)
            test_board = test_go.board
            test_board[move[0]][move[1]] = piece_type
            test_go.update_board(test_board)
            removed_pieces = len(test_go.remove_died_pieces(3 - piece_type))

            heuristic = test_go.heuristic(3 - piece_type)
            evaluation = test_go.alphaBetaPruning(max_depth, alpha, beta, heuristic, 3 - piece_type, True)

            evaluation = -1 * evaluation + (1000 * removed_pieces)

            if evaluation > max_value or not moves:
                max_value = evaluation
                alpha = max_value
                moves = [move]
            elif evaluation == max_value:
                moves.append(move)

        return moves, max_value

    def alphaBetaPruning(self, max_depth, alpha, beta, heuristic, piece_type, is_min_move):
        if max_depth == 0:
            return heuristic

        max_value = heuristic
        for move in self.all_valid_moves(piece_type):
            test_go = self.copy_board()
            test_go.previous_board = deepcopy(test_go.board)
            test_board = test_go.board
            test_board[move[0]][move[1]] = piece_type
            test_go.update_board(test_board)
            removed_pieces = len(test_go.remove_died_pieces(3 - piece_type))

            heuristic = test_go.heuristic(3 - piece_type)
            evaluation = test_go.alphaBetaPruning(max_depth - 1, alpha, beta, heuristic, 3 - piece_type,
                                                  not is_min_move)

            evaluation = -1 * evaluation + (1000 * removed_pieces)

            if evaluation > max_value:
                max_value = evaluation
            new_evaluation = -1 * max_value

            if is_min_move:
                if new_evaluation < alpha:
                    return max_value
                if max_value > beta:
                    beta = max_value
            else:
                if new_evaluation < beta:
                    return max_value
                if max_value > alpha:
                    alpha = max_value

        return max_value

    def update_board(self, new_board):
        '''
        Update the board with new_board

        :param new_board: new board.
        :return: None.
        '''
        self.board = new_board


def readInput(n, path="input.txt"):
    with open(path, 'r') as f:
        lines = f.readlines()

        piece_type = int(lines[0])

        previous_board = [[int(x) for x in line.rstrip('\n')] for line in lines[1:n + 1]]
        board = [[int(x) for x in line.rstrip('\n')] for line in lines[n + 1: 2 * n + 1]]

        return piece_type, previous_board, board


def writeOutput(result, path="output.txt"):
    res = ""
    if result == "PASS":
        res = "PASS"
    else:
        res += str(result[0]) + ',' + str(result[1])

    with open(path, 'w') as f:
        f.write(res)


if __name__ == "__main__":
    N = 5
    piece_type, previous_board, board = readInput(N)
    go = GO(N)
    go.set_board(piece_type, previous_board, board)

    my_move = go.findBestMove(piece_type)
    print(my_move)

    writeOutput(my_move)
