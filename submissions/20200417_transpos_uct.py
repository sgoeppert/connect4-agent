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
        self.moveCount = 0
        self.winner = -1
    
    def play_move(self, col):
        player = self.activePlayer
        self.playerBoards[player] ^= np.left_shift(np.uint64(1),self.heights[col])
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
        # top = '1000000_1000000_1000000_1000000_1000000_1000000_1000000'
        top = np.uint64(283691315109952)
        height_shifts = np.left_shift(1, self.heights)
        moves = [i for i,h in enumerate(height_shifts) if not (top & h)]
        return moves
    
    def is_terminal(self):
        return self.moveCount >= 42 or self.winner != -1

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
            move = np.random.choice(self.list_moves())
            self.play_move(move)

        return self.winner

    def copy(self):
        inst = ConnectFour()
        inst.heights = self.heights.copy()
        inst.playerBoards = self.playerBoards.copy()
        inst.winner = self.winner
        inst.activePlayer = self.activePlayer
        inst.moveCount = self.moveCount
        return inst
    
    def hash(self):
        mask = self.playerBoards[0] | self.playerBoards[1]
        # bottom = '0000001_0000001_0000001_0000001_0000001_0000001_0000001'
        bottom = np.uint64(4432676798593)
        return int(self.playerBoards[self.activePlayer] + mask + bottom)

    def __hash__(self):
        return self.hash()


class Node():
    def __init__(self, state):
        self.state = state
        self._in = []
        self._out = []

        self.playout_score = 0
        self.playout_count = 0
    
    def add_child_node(self, node, action):
        edge = Edge(action)
        edge.end = node
        edge.start = self
        self._out.append(edge)
        node._in.append(edge)
        return node

    def add_child_action(self, action):
        new_state = self.state.copy()
        new_state.play_move(action)
        child = Node(new_state)

        edge = Edge(action)
        edge.end = child
        edge.start = self
        self._out.append(edge)
        child._in.append(edge)
        return child

    @property
    def children(self):
        return self._out
    
    @property
    def untried_moves(self):
        used_actions = [e.action for e in self._out]
        return (set(self.state.list_moves()) - set(used_actions))

    @property
    def is_expanded(self):
        return len(self.untried_moves) == 0
    
    @property
    def is_terminal(self):
        return self.state.is_terminal()
    

class Edge():

    def __init__(self, action):
        self.action = action
        self.start = None
        self.end = None

        self.total_reward = 0
        self.visit_count = 0

    @property
    def siblings(self):
        if self.start:
            return self.start._out
        else:
            return []

    @property
    def children(self):
        if self.end:
            return self.end._out
        else:
            return []
    
class MCTS():
    def __init__(self, state, exploration_weight=0.25):
        self.root = Node(state)

        self.c = exploration_weight

        self.nodes = dict()
        self.d1 = 1
        self.d2 = 0
        self.d3 = 0

    def select(self, node):
        path = [node]
        while not node.is_terminal:
            if not node.is_expanded:
                path.append(self.expand(node))
                return path
            else:
                edge = self.best_child(node, c=self.c)
                node = edge.end
                path.append(node)
        return path

    def expand(self, node):
        action = choice(tuple(node.untried_moves))
        new_state = node.state.copy()
        new_state.play_move(action)

        child = self.nodes.get(new_state.hash())
        if not child:
            child = Node(new_state)
            self.nodes[new_state.hash()] = child
    
        return node.add_child_node(child, action)

    def simulate(self, node):
        player = 1 - node.state.activePlayer
        sim = node.state.copy()
        winner = sim.random_playout()

        if winner == player:
            return 1
        elif winner == -1:
            return 0.5
        else: 
            return 0

    def backpropagate(self, path, reward):
        leaf = path[-1]
        leaf.playout_count += 1
        leaf.playout_score = reward

        updated = set()

        def update_descent(node, path, reward):
            for e in node._in:
                if e in updated or not e.start in path:
                    continue
                
                updated.add(e)
                e.total_reward += reward
                e.visit_count += 1
                if e.start:
                    update_descent(e.start, path, 1 - reward)

        update_descent(leaf, path, reward)

    def best_child(self, node, c=1):

        ln_node_visits = np.log(sum([e.visit_count for e in node.children]))
        def uct(n):
            return (n.total_reward / n.visit_count) + 2*c * np.sqrt(
                2*ln_node_visits / n.visit_count
            ) + np.random.uniform(0, 1/10e5)

        return max(node.children, key=uct)


def mcts_agent(observation, configuration):
    game = ConnectFour.create_from_array(observation.board)
    tree = MCTS(game)
    
    timeout = configuration.timeout * 0.90
    start = time.time()
    while (time.time() - start) < timeout:
        path = tree.select(tree.root)
        n = path[-1]
        reward = tree.simulate(n)
        tree.backpropagate(path, reward)
    
    return tree.best_child(tree.root, c=0).action
