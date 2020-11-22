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
    
    def add_child(self, action):
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
    def player(self):
        return self.state.activePlayer

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
    
    def __str__(self):
        edges = [str(e) for e in self.children]
        return f"Node Edges: {edges}"

    def __repr__(self):
        return self.__str__()

class Edge():

    def __init__(self, action):
        self.action = action
        self.start = None
        self.end = None

        self.total_reward = 0
        self.visit_count = 0

        self.pess = 0
        self.opti = 1

    @property
    def solved(self):
        return self.pess == self.opti

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
    
    @property
    def player(self):
        return self.start.state.activePlayer
    
    def __str__(self):
        return f"A{self.action} Q {self.total_reward} N {self.visit_count} V {round(self.total_reward/self.visit_count,4)} Pess {self.pess} Opti {self.opti}"
    
class MCTS():
    def __init__(self, state):
        self.root = Node(state)
        self.max_player = self.root.state.activePlayer

    def select(self, node):
        path = [node]
        while not node.is_terminal:
            if not node.is_expanded:
                path.append(self.expand(node))
                return path
            else:
                edge = self.best_child(node)
                node = edge.end
                path.append(node)
        return path
    
    def expand(self, node):
        action = choice(tuple(node.untried_moves))
        return node.add_child(action)

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



        def prop_pess(node):
            for s in node._in:
                for n in s.start._in:
                    # s = all edges leading to this node
                    # n = parent of s
                    old_pess = n.pess
                    if old_pess < s.pess:
                        # n is a max edge
                        if n.player == self.max_player:
                            n.pess = s.pess
                            prop_pess(n.end)
                        else:
                            if not n.end.is_expanded:
                                n.pess = 0
                            else:
                                n.pess = min([c.pess for c in n.children])
                            if old_pess > n.pess:
                                prop_pess(n.end)

        def prop_opti(node):
            for s in node._in:
                for n in s.start._in:
                    old_opti = n.opti
                    if old_opti > s.opti:
                        # n is a max edge
                        if n.player == self.max_player:
                            if not n.end.is_expanded:
                                n.opti = 1
                            else:
                                n.opti = max([c.opti for c in n.children])
                            if old_opti > n.opti:
                                prop_opti(n.end)
                        else:
                            n.opti = s.opti
                            prop_opti(n.end)

        updated = set()
        def update_all(node, path, reward):
            for e in node._in:
                if e in updated:
                    continue

                updated.add(e)
                e.total_reward += reward
                e.visit_count += 1
                if e.start:
                    update_all(e.start, path, 1 - reward)

        def update_descent(node, path, reward):
            for e in node._in:
                if e in updated or not e.start in path:
                    continue
                
                updated.add(e)
                e.total_reward += reward
                e.visit_count += 1
                if e.start:
                    update_descent(e.start, path, 1 - reward)

        
        # update_all(leaf, path, reward)
        update_descent(leaf, path, reward)

        
        if leaf.is_terminal:
            for e in leaf._in:
                score = leaf.playout_score
                if e.player != self.max_player:
                    score = 1 - score
                e.pess = score
                e.opti = score
            prop_pess(leaf)
            prop_opti(leaf)
        

    def best_child(self, node, c=0.2):

        ln_node_visits = np.log(sum([e.visit_count for e in node.children]))
        gamma = 0
        delta = -0.1
        def Q(n):
            if node.player == self.max_player:
                bias = gamma * n.pess + delta * n.opti
            else:
                bias = delta * (1-n.pess) + gamma * (1-n.opti)

            return n.total_reward / n.visit_count + bias

        def uct(n):
            q = Q(n)
            return q + 2*c * np.sqrt(
                2*ln_node_visits / n.visit_count
            ) + np.random.uniform(0, 1/10e5)

        
        is_max_node = node.player == self.max_player
        options = []
        if node._in:
            parent = node._in[0]

            for ch in node.children:
                if is_max_node:
                    if ch.opti <= parent.pess:
                        pass
                    else:
                        options.append(ch)
                else:
                    if ch.pess >= parent.opti:
                        pass
                    else:
                        options.append(ch)
        else:
            options = node.children

        if len(options) == 0:
            # everything has been pruned, just pick at random
            options = node.children

        return max(options, key=uct)

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
