from random import choice
import random
import sys
import time
import math
from copy import deepcopy
import operator
import timeit
import argparse
from collections import Counter


class OPGO:
    def __init__(self, n):
        self.size = n
        self.X_move = True
        self.died_pieces = []
        self.n_move = 0
        self.max_move = n * n - 1
        self.komi = n / 2
        self.verbose = False

    def set_board(self, piece_type, previous_board, board):
        # setting the board, previous board with initial piecetype
        for i in range(self.size):
            for j in range(self.size):
                if previous_board[i][j] == piece_type and board[i][j] != piece_type:
                    self.died_pieces.append((i, j))

        self.previous_board = previous_board
        self.board = board

    def compare_board(self, b1, b2):
        # comparing if 2 boards are the same or not. return false if not equal to each other
        for i in range(self.size):
            for j in range(self.size):
                if b1[i][j] != b2[i][j]:
                    return False
        return True

    def copy_board(self):
        # creating a copy of the board to test without altering the go.board
        return deepcopy(self)

    def detect_neighbor(self, i, j):

        board = self.board
        neighbors = []
        # Detect borders and add neighbor coordinates
        if i > 0: neighbors.append((i - 1, j))
        if i < len(board) - 1: neighbors.append((i + 1, j))
        if j > 0: neighbors.append((i, j - 1))
        if j < len(board) - 1: neighbors.append((i, j + 1))
        return neighbors

    def detect_neighbor_ally(self, i, j):
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
        stack = [(i, j)]  # stack for DFS serach
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
        board = self.board
        ally_members = self.ally_dfs(i, j)
        for member in ally_members:
            neighbors = self.detect_neighbor(member[0], member[1])
            for piece in neighbors:
                # If there is empty space around a piece, it has liberty
                if board[piece[0]][piece[1]] == 0:
                    return True
        # If none of the pieces in a allied group has an empty space, it has no liberty
        return False

    def find_died_pieces(self, piece_type):
        board = self.board
        died_pieces = []

        for i in range(len(board)):
            for j in range(len(board)):
                # Check if there is a piece at this position:
                if board[i][j] == piece_type:
                    # The piece die if it has no liberty
                    if not self.find_liberty(i, j):
                        died_pieces.append((i, j))
        return died_pieces

    def remove_died_pieces(self, piece_type):
        died_pieces = self.find_died_pieces(piece_type)
        if not died_pieces: return []
        self.remove_certain_pieces(died_pieces)
        return died_pieces

    def remove_certain_pieces(self, positions):
        board = self.board
        for piece in positions:
            board[piece[0]][piece[1]] = 0
        self.update_board(board)

    def opp_valid_place_check(self, i, j, piece_type, test_check=False):
        board = self.board
        verbose = self.verbose
        if test_check:
            verbose = False

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
        if test_go.find_liberty(i, j):
            return True

        # If not, remove the died pieces of opponent and check again
        test_go.remove_died_pieces(3 - piece_type)
        if not test_go.find_liberty(i, j):
            return False

        # Check special case: repeat placement causing the repeat board state (KO rule)
        else:
            if self.died_pieces and self.compare_board(self.previous_board, test_go.board):
                return False
        return True

    def update_board(self, new_board):
        self.board = new_board

    def score(self, piece_type):
        # finding the number of stones on the board for a particular piecetype
        board = self.board
        cnt = 0
        for i in range(self.size):
            for j in range(self.size):
                if board[i][j] == piece_type:
                    cnt += 1
        return cnt

    #######################################################################################################################################################


class GO:
    def __init__(self, n):
        self.size = n
        self.X_move = True
        self.died_pieces = []
        self.n_move = 0
        self.max_move = n * n - 1
        self.komi = n / 2
        self.verbose = False

    def set_board(self, piece_type, previous_board, board):
        # setting the board and previous board for a particular piecetype
        for i in range(self.size):
            for j in range(self.size):
                if previous_board[i][j] == piece_type and board[i][j] != piece_type:
                    self.died_pieces.append((i, j))

        self.previous_board = previous_board
        self.board = board

    def compare_board(self, b1, b2):
        # compare if two boards are similar
        for i in range(self.size):
            for j in range(self.size):
                if b1[i][j] != b2[i][j]:
                    return False
        return True

    def copy_board(self):
        return deepcopy(self)

    def detect_neighbor(self, i, j):
        board = self.board
        neighbors = []
        # on the go.board, return neighbours of a position within the board
        if i > 0: neighbors.append((i - 1, j))
        if i < len(board) - 1: neighbors.append((i + 1, j))
        if j > 0: neighbors.append((i, j - 1))
        if j < len(board) - 1: neighbors.append((i, j + 1))
        return neighbors

    def detect_neighbor_moves(self, board, i, j):
        neighbors = []
        # on the board that is passed as a parameter, return neighbours of a position within the board
        if i > 0: neighbors.append((i - 1, j))
        if i < len(board) - 1: neighbors.append((i + 1, j))
        if j > 0: neighbors.append((i, j - 1))
        if j < len(board) - 1: neighbors.append((i, j + 1))
        return neighbors

    def detect_neighbor_ally(self, i, j):
        # on the go.board, check the allies in neighbouring stones and return those positions for a particular stone
        board = self.board
        neighbors = self.detect_neighbor(i, j)
        allies = []
        # for each position in neighbour
        for x in neighbors:
            # if nrighbour has tha same color as the stone, then add it to allies
            if board[x[0]][x[1]] == board[i][j]:
                allies.append(x)
        return allies

    def detect_neighbor_ally_moves(self, board, i, j):
        # on the board that is passed as the parameter, check the allies in neighbouring stones and return those positions for a particular stone
        neighbors = self.detect_neighbor_moves(board, i, j)
        allies = []
        # for each position in neighbour
        for piece in neighbors:
            # if nrighbour has tha same color as the stone, then add it to allies
            if board[piece[0]][piece[1]] == board[i][j]:
                allies.append(piece)
        return allies

    def ally_dfs(self, i, j):
        # searching for ally stones using DFS
        stack = [(i, j)]
        allies = []
        while stack:
            piece = stack.pop()
            allies.append(piece)
            neighbor_allies = self.detect_neighbor_ally(piece[0], piece[1])
            for ally in neighbor_allies:
                if ally not in stack and ally not in allies:
                    stack.append(ally)
        return allies

    def ally_dfs_moves(self, board, i, j):
        # searching for ally stones using DFS
        stack = [(i, j)]
        allies = []
        while stack:
            piece = stack.pop()
            allies.append(piece)
            neighbor_allies = self.detect_neighbor_ally_moves(board, piece[0], piece[1])
            for ally in neighbor_allies:
                if ally not in stack and ally not in allies:
                    stack.append(ally)
        return allies

    def find_liberty(self, i, j):
        # check if a particular stone has liberty left
        board = self.board
        allies = self.ally_dfs(i, j)
        for x in allies:
            neighbors = self.detect_neighbor(x[0], x[1])
            for y in neighbors:
                if board[y[0]][y[1]] == 0:
                    return True
        return False

    def find_liberty_moves(self, board, i, j):
        # check if a particular stone has liberty left
        allies = self.ally_dfs_moves(board, i, j)
        for x in allies:
            neighbors = self.detect_neighbor_moves(board, x[0], x[1])
            for y in neighbors:
                if board[y[0]][y[1]] == 0:
                    return True
        return False

    def find_died_pieces(self, piece_type):
        # find died pieces on the go.board
        board = self.board
        died_pieces = []
        for i in range(len(board)):
            for j in range(len(board)):
                # Check if there is a piece at this position:
                if board[i][j] == piece_type:
                    # The piece die if it has no liberty
                    if not self.find_liberty(i, j):
                        died_pieces.append((i, j))
        return died_pieces

    def find_died_pieces_moves(self, board, piece_type):
        # find died pieces on the board passed as a parameter
        died_pieces = []
        for i in range(len(board)):
            for j in range(len(board)):
                # Check if there is a piece at this position:
                if board[i][j] == piece_type:
                    # The piece die if it has no liberty
                    if not self.find_liberty_moves(board, i, j):
                        died_pieces.append((i, j))
        return died_pieces

    def remove_died_pieces(self, piece_type):
        died_pieces = self.find_died_pieces(piece_type)
        if not died_pieces: return []
        self.remove_certain_pieces(died_pieces)
        return died_pieces

    def remove_certain_pieces(self, positions):
        board = self.board
        for piece in positions:
            board[piece[0]][piece[1]] = 0
        self.update_board(board)

    def valid_place_check(self, i, j, piece_type, test_check=False):
        board = self.board
        verbose = self.verbose
        if test_check:
            verbose = False

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
        if test_go.find_liberty(i, j):
            return True

        # If not, remove the died pieces of opponent and check again
        test_go.remove_died_pieces(3 - piece_type)
        if not test_go.find_liberty(i, j):
            return False

        # Check special case: repeat placement causing the repeat board state (KO rule)
        else:
            if self.died_pieces and self.compare_board(self.previous_board, test_go.board):
                return False
        return True

    def valid_place_check_moves(self, board, i, j, piece_type, test_check=False):
        verbose = self.verbose
        if test_check:
            verbose = False

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
        if test_go.find_liberty(i, j):
            return True

        # If not, remove the died pieces of opponent and check again
        test_go.remove_died_pieces(3 - piece_type)
        if not test_go.find_liberty(i, j):
            return False

        # Check special case: repeat placement causing the repeat board state (KO rule)
        else:
            if self.died_pieces and self.compare_board(self.previous_board, test_go.board):
                return False
        return True

    def update_board(self, new_board):
        self.board = new_board

    def score(self, piece_type):
        board = self.board
        cnt = 0
        for i in range(self.size):
            for j in range(self.size):
                if board[i][j] == piece_type:
                    cnt += 1
        return cnt


#######################################################################################################################################################

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


def find_our_location(board, piece_type):
    # find positions of stones on the board
    placement = []
    for i in range(go.size):
        for j in range(go.size):
            if board[i][j] == piece_type:
                placement.append((i, j))
    return placement


def evaluate(board, color):
    # evaluation function: difference of no. of our stones and no. of opponent's stones
    scoreo, scorex = wins()

    if color == 1:
        score = scorex - scoreo
    if color == 2:
        score = scoreo - scorex
    # print('this is the score for this board')
    # print(score)
    return score


def wins():
    # return count
    cnt_2 = go.score(2)
    cnt_1 = go.score(1)
    return cnt_2 + go.komi, cnt_1


def empty_cells(board, player):
    # check valid moves for a player(go.board)
    possible_placements = []
    for i in range(go.size):
        for j in range(go.size):
            if go.valid_place_check(i, j, player, test_check=True):
                possible_placements.append((i, j))
    random.shuffle(possible_placements)
    return possible_placements


def empty_cells_moves(board, player):
    # check valid moves for a player(board we pass as parameter)
    possible_placements = []
    for i in range(go.size):
        for j in range(go.size):
            if go.valid_place_check_moves(board, i, j, player, test_check=True):
                possible_placements.append((i, j))
    random.shuffle(possible_placements)
    return possible_placements


def opp_empty_cells(opgo, board, player):
    # check valid moves for a player(for opponent)
    possible_placements = []
    for i in range(opgo.size):
        for j in range(opgo.size):
            if opgo.opp_valid_place_check(i, j, player, test_check=True):
                possible_placements.append((i, j))
    random.shuffle(possible_placements)
    return possible_placements


def valid_move(x, y, player):
    # check if move is valid
    if (x, y) in empty_cells(board, player):
        return True
    else:
        return False


def set_move(board, x, y, player):
    # set the move on the board
    if valid_move(x, y, player):
        go.previous_board = deepcopy(board)
        board[x][y] = player
        go.board = board
        return board
    else:
        return board


def opp_valid_move(opgo, x, y, player):
    # check if move is valid(for opponent)
    if (x, y) in opp_empty_cells(opgo, board, player):
        return True
    else:
        return False


def opp_set_move(opgo, board, x, y, player):
    # set the move on the board(for opponent board)
    if opp_valid_move(opgo, x, y, player):
        opgo.previous_board = deepcopy(board)
        board[x][y] = player
        opgo.board = board
        return board
    else:
        return board


def minimax_min_node(board, color, depth, alpha, beta, start_time):
    new_board = deepcopy(board)
    # print('in minimax min')
    cur_min = math.inf
    moves = empty_cells(new_board, color)

    end = time.time()
    if len(moves) == 0 or depth == 0 or end - start_time > 8.5:
        return (-1, -1), evaluate(new_board, color)
    else:
        for i in moves:
            board_to_pass_each_time = deepcopy(board)
            new_board = set_move(board_to_pass_each_time, i[0], i[1], color)
            go.remove_died_pieces(3 - color)
            if color == 1:
                next_player = 2
            else:
                next_player = 1
            new_move, new_score = minimax_max_node(new_board, next_player, depth - 1, alpha, beta, start_time)
            if new_score < cur_min:
                cur_min = new_score
                best_move = i
            beta = min(new_score, beta)
            if beta <= alpha:
                break
        return best_move, cur_min


def minimax_max_node(board, color, depth, alpha, beta, start_time):
    end = time.time()
    new_board = deepcopy(board)
    cur_max = -math.inf
    # get our valid moves
    moves = empty_cells(new_board, color)
    # remove moves from valid moves which would be killed by opponent in the next round if they are placed
    stonestoremove = []
    for i in moves:
        go.board[i[0]][i[1]] = color
        opmoves = empty_cells(go.board, 3 - color)
        for j in opmoves:
            go.board[j[0]][j[1]] = 3 - color
            deadstones = go.find_died_pieces(color)
            go.board[j[0]][j[1]] = 0
            if i in deadstones:
                if i not in stonestoremove:
                    stonestoremove.append(i)
        go.board[i[0]][i[1]] = 0

    for x in stonestoremove:
        if x in moves:
            moves.remove(x)

    if len(moves) == 0 or depth == 0 or end - start_time > 8.5:
        return (-1, -1), evaluate(new_board, color)
    else:
        for i in moves:
            board_to_pass_each_time = deepcopy(board)
            new_board = set_move(board_to_pass_each_time, i[0], i[1], color)
            go.remove_died_pieces(3 - color)
            if color == 1:
                next_player = 2
            else:
                next_player = 1
            new_move, new_score = minimax_min_node(new_board, next_player, depth - 1, alpha, beta, start_time)
            if new_score > cur_max:
                cur_max = new_score
                best_move = i
            alpha = max(new_score, alpha)
            if beta <= alpha:
                break
        return best_move, cur_max


def select_move_minimax(board, color):
    start = time.time()
    best_move, score = minimax_max_node(board, color, max_depth, -math.inf, math.inf, start)
    i, j = best_move[0], best_move[1]

    return i, j, score


def opp_minimax_min_node(opgo, board, color, depth, alpha, beta, start_time):
    new_board = deepcopy(board)

    cur_min = math.inf
    moves = opp_empty_cells(opgo, new_board, color)

    end = time.time()
    if len(moves) == 0 or depth == 0 or end - start_time > 8.5:
        return (-1, -1), evaluate(new_board, color)
    else:
        for i in moves:

            board_to_pass_each_time = deepcopy(board)
            new_board = opp_set_move(opgo, board_to_pass_each_time, i[0], i[1], color)
            opgo.remove_died_pieces(3 - color)
            if color == 1:
                next_player = 2
            else:
                next_player = 1
            new_move, new_score = opp_minimax_max_node(opgo, new_board, next_player, depth - 1, alpha, beta, start_time)
            if new_score < cur_min:
                cur_min = new_score
                best_move = i
            beta = min(new_score, beta)
            if beta <= alpha:
                break
        return best_move, cur_min


def opp_minimax_max_node(opgo, board, color, depth, alpha, beta, start_time):
    end = time.time()
    new_board = deepcopy(board)
    # board_to_pass_each_time = deepcopy(board)
    cur_max = -math.inf
    moves = opp_empty_cells(opgo, new_board, color)

    if len(moves) == 0 or depth == 0 or end - start_time > 8.5:
        return (-1, -1), evaluate(new_board, color)
    else:
        for i in moves:

            board_to_pass_each_time = deepcopy(board)
            new_board = opp_set_move(opgo, board_to_pass_each_time, i[0], i[1], color)
            opgo.remove_died_pieces(3 - color)
            if color == 1:
                next_player = 2
            else:
                next_player = 1
            new_move, new_score = opp_minimax_min_node(opgo, new_board, next_player, depth - 1, alpha, beta, start_time)
            if new_score > cur_max:
                cur_max = new_score
                best_move = i
            alpha = max(new_score, alpha)
            if beta <= alpha:
                break
        return best_move, cur_max


def opp_select_move_minimax(opgo, board, color):
    start = time.time()
    best_move, score = opp_minimax_max_node(opgo, board, color, max_depth_opponent, -math.inf, math.inf, start)
    i, j = best_move[0], best_move[1]

    return i, j, score


def get_input(go, piece_type):
    ####################################################################################################################################
    # KILLER MOVE
    ####################################################################################################################################

    empty_spaces = []
    for i in range(go.size):
        for j in range(go.size):
            if go.board[i][j] == 0:
                empty_spaces.append((i, j))

    killcount = dict()
    for i in empty_spaces:
        go.board[i[0]][i[1]] = piece_type
        died_pieces = go.find_died_pieces(3 - piece_type)
        go.board[i[0]][i[1]] = 0
        if len(died_pieces) >= 1:
            # print('taking out more than 1 opponents')
            killcount[i] = len(died_pieces)
    # print(killcount)

    sorted_killcount = sorted(killcount, key=killcount.get, reverse=True)
    # print(sorted_killcount)

    for i in sorted_killcount:
        # lets start here
        testboard = deepcopy(go.board)
        testboard[i[0]][i[1]] = piece_type
        died_stone = go.find_died_pieces_moves(testboard, 3 - piece_type)
        for x in died_stone:
            testboard[x[0]][x[1]] = 0
        # print(go.previous_board)
        # print(testboard)
        if i != None and go.previous_board != testboard:
            return i

    ####################################################################################################################################
    # MOVES TO REMOVE
    ####################################################################################################################################
    moves = empty_cells(go.board, piece_type)
    # print('moves')
    # print(moves)
    # print(len(moves))

    moves_to_remove = []
    # moves = empty_cells_moves(go.board, piece_type)
    # print('these are the moves')
    # print(moves)
    for i in moves:
        go.board[i[0]][i[1]] = piece_type
        oppmove = empty_cells_moves(go.board, 3 - piece_type)
        for j in oppmove:
            go.board[j[0]][j[1]] = 3 - piece_type
            died_pieces = go.find_died_pieces(piece_type)
            go.board[j[0]][j[1]] = 0
            if i in died_pieces:
                moves_to_remove.append(i)
        go.board[i[0]][i[1]] = 0
    # print('these are the moves to remove')
    # print(moves_to_remove)

    for x in moves_to_remove:
        if x in moves:
            moves.remove(x)
    print('these are the final moves')
    print(moves)

    if len(moves) == 0:
        return "PASS"
    ####################################################################################################################################
    # DEFENSE
    ####################################################################################################################################
    # saving our own ass

    # print('this is the board right now')
    # print(go.board)

    save_moves = dict()

    # opponent_moves = empty_cells_moves(go.board, 3 - piece_type)
    opponent_moves = []
    for i in range(go.size):
        for j in range(go.size):
            if go.board[i][j] == 0:
                opponent_moves.append((i, j))

    for i in opponent_moves:
        go.board[i[0]][i[1]] = 3 - piece_type
        our_dead_stones = go.find_died_pieces(piece_type)
        go.board[i[0]][i[1]] = 0
        if len(our_dead_stones) >= 1:
            # print('opponent is killing our guy')
            save_moves[i] = len(our_dead_stones)

    sorted_save_moves = sorted(save_moves, key=save_moves.get, reverse=True)

    for i in sorted_save_moves:
        if i != None and i in moves:
            # print('playing this move to save my ass')
            return i

    ####################################################################################################################################

    # now we occupy they two liberties without fucking up
    # print('this is the board right now')

    # print(go.board)

    # now, we calculate the liberties of the board
    # so, we call the so called function get positions or whatever

    position_of_opponent = find_our_location(go.board, 3 - piece_type)
    # now we have the positions of the opponent.
    # so, we check all the neighbours
    empty_x = []
    # empty_y = []
    neighbours = []

    for i in position_of_opponent:
        neighbors = [(i[0] + board[0], i[1] + board[1]) for board in
                     [(-1, 0), (1, 0), (0, -1), (0, 1)]
                     if ((0 <= i[0] + board[0] < go.size) and (0 <= i[1] + board[1] < go.size))]
        for x in neighbors:
            neighbours.append(x)

    # now we check if the neighbours are empty and add them in a list
    for i in neighbours:
        if go.board[i[0]][i[1]] == 0:
            empty_x.append(i)
    # print('these are the positions of liberties of opponent')
    # print(empty_x)

    # now we place our moves, and check for change in the opponent's liberties
    for x in moves:
        # place our move
        testboard = deepcopy(go.board)
        testboard[x[0]][x[1]] = piece_type
        # remove died pieces if any
        died_stone = go.find_died_pieces_moves(testboard, 3 - piece_type)
        for m in died_stone:
            testboard[m[0]][m[1]] = 0
        # now check the liberty of the opponent
        position_of_opponent = find_our_location(testboard, 3 - piece_type)
        empty_y = []
        neighbours = []

        for i in position_of_opponent:
            neighbors = [(i[0] + board[0], i[1] + board[1]) for board in
                         [(-1, 0), (1, 0), (0, -1), (0, 1)]
                         if ((0 <= i[0] + board[0] < go.size) and (0 <= i[1] + board[1] < go.size))]
            for n in neighbors:
                neighbours.append(n)

        for z in neighbours:
            if go.board[z[0]][z[1]] == 0:
                empty_y.append(z)

        if len(empty_x) - len(empty_y) >= 1:
            # print('we are taking two of their liberties')
            # print(empty_y)
            return x

    ####################################################################################################################################
    # INITIAL MOVES
    ####################################################################################################################################

    if len(moves) >= 15:
        if (2, 2) in moves:
            x = 2
            y = 2
            return (x, y)
        if (1, 1) in moves:
            x = 1
            y = 1
            return (x, y)
        if (1, 3) in moves:
            x = 1
            y = 3
            return (x, y)
        if (3, 1) in moves:
            x = 3
            y = 1
            return (x, y)
        if (3, 3) in moves:
            x = 3
            y = 3
            return (x, y)
        if (2, 0) in moves:
            x = 2
            y = 0
            return (x, y)
        if (2, 4) in moves:
            x = 2
            y = 4
            return (x, y)
        if (0, 2) in moves:
            x = 0
            y = 2
            return (x, y)
        if (4, 2) in moves:
            x = 4
            y = 2
            return (x, y)

    ####################################################################################################################################
    # OPPONENT'S MOVE USING MINMAX AND PLAY ACCORDINGLY
    ####################################################################################################################################

    # finding opponents move
    # print('this is the board before playing out the opponents move')
    # print(go.board)
    # opponent's best move
    opp_board = deepcopy(go.board)
    opp_previous_board = deepcopy(go.previous_board)

    opgo = OPGO(5)
    opgo.set_board(3 - piece_type, opp_previous_board, opp_board)
    # print('this is the opgo board')
    # print(opgo.board)

    movei, movej, score = opp_select_move_minimax(opgo, opp_board, 3 - piece_type)
    x, y = movei, movej
    # print('this is the opponents best move now.')
    # print((x,y))
    # set the board as opponent move
    go.board[x][y] = 3 - piece_type
    # print('this is the board after placing the opponents move')
    # print(go.board)

    empty_spaces = []
    for i in range(go.size):
        for j in range(go.size):
            if go.board[i][j] == 0:
                empty_spaces.append((i, j))

    killcount = dict()
    for i in empty_spaces:
        go.board[i[0]][i[1]] = piece_type
        died_pieces = go.find_died_pieces(3 - piece_type)
        go.board[i[0]][i[1]] = 0
        if len(died_pieces) >= 1:
            print('taking out more than 1 opponents')
            killcount[i] = len(died_pieces)

    killcount_remove = []
    sorted_killcount = sorted(killcount, key=killcount.get, reverse=True)
    go.board[x][y] = 0

    if len(sorted_killcount) != 0:
        for i in sorted_killcount:
            go.board[i[0]][i[1]] == piece_type
            opmoves = empty_cells_moves(go.board, 3 - piece_type)
            for j in opmoves:
                go.board[j[0]][j[1]] = 3 - piece_type
                died_pieces = go.find_died_pieces_moves(go.board, piece_type)
                go.board[j[0]][j[1]] = 0
                if i in died_pieces:
                    killcount_remove.append(i)
            go.board[i[0]][i[1]] = 0

        for x in killcount_remove:
            if x in sorted_killcount:
                sorted_killcount.remove(x)

        for i in sorted_killcount:
            if i in moves:
                return i

    ####################################################################################################################################
    # MINMAX
    ####################################################################################################################################

    movei, movej, score = select_move_minimax(go.board, piece_type)
    x, y = movei, movej
    return (x, y)


N = 5
max_depth = 4
max_depth_opponent = 1
piece_type, previous_board, board = readInput(N)
go = GO(N)
go.set_board(piece_type, previous_board, board)
action = get_input(go, piece_type)
if action == None:
    action = "PASS"

print(action)
writeOutput(action)
