import numpy as np
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

    def display(self):
        output = np.zeros((6,7))
        for j in range(6):
            for i in range(7):
                index = i * 7 + j
                m = np.uint64(1 << index)
                if self.playerBoards[0] & m:
                    output[5-j,i] = 1
                elif self.playerBoards[1] & m:
                    output[5-j,i] = 2
                
        return output

    def __str__(self):
        output = np.zeros((6,7))
        for j in range(6):
            for i in range(7):
                index = i * 7 + j
                m = np.uint64(1 << index)
                if self.playerBoards[0] & m:
                    output[5-j,i] = 1
                elif self.playerBoards[1] & m:
                    output[5-j,i] = 2
                
        return output.__str__()

    
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
    
    def __init__(self, state, depth=0):
        self._rewards = 0
        self._visits = 0

        self._children = dict()
        self._parents = []

        self._state = state
        self.depth = depth
        self.action = None
        
    @property
    def active_player(self):
        return self._state.activePlayer

    @property
    def player(self):
        return 1 - self._state.activePlayer
    
    def is_terminal(self):
        return self._state.is_terminal()
    
    def simulate(self):
        sim = self._state.copy()
        winner = sim.random_playout()

        return winner

    def is_expanded(self):
        all_moves = set(self._state.list_moves())
        tried_moves = set(self._children)
        return len(list(all_moves - tried_moves)) == 0

    def expand(self):
        all_moves = set(self._state.list_moves())
        tried_moves = set(self._children)
        action = np.random.choice(list(all_moves - tried_moves))
        
        next_state = self._state.copy()
        next_state.play_move(action)
        
        child = Node(next_state, depth=self.depth+1)
        child.action = action
        self._children[action] = child

        child._parents.append(self)
        return child

    def best_child(self, c_p=0.2):

        log_n_visits = np.log(self._visits)
        def uct(child):
            a, n = child
            score = n._rewards / n._visits + c_p * np.sqrt(2 * log_n_visits/n._visits)
            return score

        return max(self._children.items(), key=uct)

    def __repr__(self):
        child_str = ",".join([f"(A {a}, Q {c._rewards/max(1,c._visits):.4} N {c._visits})" for a,c in self._children.items()])
        return f"Node {self.action} Q {self._rewards/max(1,self._visits):.4} N {self._visits} Children: [{child_str}]"

class MCTS():
    def __init__(self, state, exploration=1.41):
        self.root = Node(state, depth=-1)
        self._c_p = exploration
        self._max_player = self.root.active_player

    def select(self):
        n = self.root
        path = [n]

        while not n.is_terminal():
            if not n.is_expanded():
                next_n = n.expand()
                path.append(next_n)
                return path
            else:
                a, n = n.best_child(c_p=self._c_p)
                path.append(n)
        return path
    
    def rollout(self, node):
        res = node.simulate()
        player = node.player
        if res == player:
            return 1
        elif res == -1:
            return 0
        else:
            return -1
    
    def backup(self, path, reward):
        for node in reversed(path):
            node._visits += 1
            node._rewards += reward
            reward = -reward

def player(observation, configuration):
    start = time.time()
    game = ConnectFour.create_from_array(observation.board)
    mcts = MCTS(game, exploration=0.75)

    timeout = configuration.timeout * 0.95
    while (time.time() - start) < timeout:
        path = mcts.select()
        leaf = path[-1]
        reward = mcts.rollout(leaf)
        mcts.backup(path, reward)
    
    a,_ = mcts.root.best_child(c_p=0)
    return int(a)