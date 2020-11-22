import numpy as np
import copy
from random import choice
from math import sqrt, log
import time

class ConnectFour():

    def __init__(self):
        self.playerBoards = np.zeros(2, dtype=np.uint64)
        self.heights = np.array([0, 7, 14, 21, 28, 35, 42], dtype=np.uint64)
        self.activePlayer = 0
        self.winner = None
        self.moveCount = 0
    
    def play_move(self, col, player=None):
        if player == None:
            player = self.activePlayer

        self.playerBoards[player] ^= np.left_shift(np.uint64(1),self.heights[col], dtype=np.uint64)
        self.heights[col] += 1
        is_terminal = self.is_win(self.playerBoards[player])
        if is_terminal:
            self.winner = player
        
        self.moveCount += 1
        self.activePlayer = 1 - player

    
    def is_win(self, board):
        directions = np.array([1,6,7,8])
        
        shift1 = np.right_shift(board, directions)
        board = np.bitwise_and(board, shift1)
        shift2 = np.right_shift(board, 2 * directions)
        res = np.bitwise_and(board, shift2)
        return np.max(res) > 0

    def list_moves(self):
        top = np.uint64(int('1000000_1000000_1000000_1000000_1000000_1000000_1000000', 2))
        height_shifts = np.left_shift(1, self.heights)
        moves = [i for i,h in enumerate(height_shifts) if not (top & h)]
        return moves
    

    def is_terminal(self):
        return self.moveCount >= 42 or self.winner != None

    def create_from_array(input_board):
        c4 = ConnectFour()

        b = np.array(input_board)
        b = np.reshape(b, (6,7))

        boards = np.zeros(2, dtype=np.uint64)
        moveCount = 0
        heights = np.array([0, 7, 14, 21, 28, 35, 42], dtype=np.uint64)

        for j in range(6):
            for i in range(7):
                index = i * 7 + j
                m = np.uint64(1 << index)
                board_index = None
                if b[5-j,i] == 1:
                    board_index = 0
                elif b[5-j,i] == 2:
                    board_index = 1

                if board_index != None:
                    boards[board_index] |= m
                    moveCount += 1
                    heights[i] = index+1

        activePlayer = moveCount & 1

        c4.playerBoards = boards
        c4.moveCount = moveCount
        c4.activePlayer = activePlayer
        c4.heights = heights
        winner = [i for i in range(2) if c4.is_win(c4.playerBoards[i])]
        if len(winner) > 0:
            c4.winner = winner[0]
        return c4

    def random_playout(self):
        while not self.is_terminal():
            move = choice(self.list_moves())
            self.play_move(move, self.activePlayer)

        return self.winner

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))
        return result

class Node():

    def __init__(self, state, action, parent):
        self.Q = 0
        self.N = 0
        self.action = action
        self.children = dict()
        self.state = state
        self.parent = parent
        self.player = 1-self.state.activePlayer
        self.is_expanded = False
        self.is_terminal = self.state.is_terminal()
    
    def expand(self):
        free_moves = list(set(self.state.list_moves()) - set(self.children))
        move = choice([m for m in free_moves if not self.children.get(m)])
        next_state = copy.deepcopy(self.state)
        next_state.play_move(move)
            
        child = Node(next_state, move, self)
        if next_state.is_terminal():
            child.is_terminal = True
            child.is_expanded = True
        
        self.children[move] = child

        if len(free_moves) == 1:
            self.is_expanded = True

        return child

    def best_child(self, c=0.707):
        def UCB(v, v1, c):
            Q_v1 = v1.Q
            N_v1 = v1.N + 1
            N_v = v.N
            explo = c * sqrt( (2*log(N_v)) / N_v1 )
            base = (Q_v1/N_v1) 
            val = base + explo
            return val

        children = sorted(self.children.items())
        scores = [UCB(self, child, c) for i,child in children]
        best_index = np.argmax(scores)

        best = children[best_index][1]
        return best


class MCTS():
    def __init__(self, state):
        self.root = Node(state, -1, None)

    def TreePolicy(self):
        v = self.root
        while not v.is_terminal:
            if not v.is_expanded:
                return v.expand()
            else:
                c = 1
                v = v.best_child(c)
        return v
    
    def DefaultPolicy(self, v):
        sim = copy.deepcopy(v.state)
        winner = sim.random_playout()
        if winner == v.player:
            return 1
        elif winner == None:
            return 0.5
        else: 
            return 0

    def Backup(self, v, reward):
        while v:
            v.N += 1
            v.Q += reward
            reward = 1-reward
            v = v.parent

def mcts_agent(observation, configuration):
    game = ConnectFour.create_from_array(observation.board)
    tree_search = MCTS(game)
    
    timeout = configuration.timeout * 0.95
    start = time.time()
    while (time.time() - start) < timeout:
        best_child = tree_search.TreePolicy()
        reward = tree_search.DefaultPolicy(best_child)
        tree_search.Backup(best_child, reward)
    
    return tree_search.root.best_child(0).action