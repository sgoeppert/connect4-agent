import random
import numpy as np
import time


class ConnectFour:

    def __init__(self):
        """The game board is stored in two 64-bit unsigned integers
        each player's set of played positions is store in one uint64
        these bitboards can be interpreted as followed:

        6 13 20 27 34 41 48
        -------------------
        5 12 19 26 33 40 47
        4 11 18 25 32 39 46
        3 10 17 24 31 38 45
        2 9  16 23 30 37 44
        1 8  15 22 29 36 43
        0 7  14 21 28 35 42

        So 0b000..000001 is the bottom left position (index 0).

        There is an additional 7th row at the top which is a necessary buffer for
        later bitwise operations used to calculate winning states
        """
        self._player_boards = np.zeros(2, dtype=np.uint64)
        """We also store the lowest position for each column that is empty. This allows 
        us to simply grab the index into the bitboard when playing a stone in a column
        """
        self._heights = np.array([0, 7, 14, 21, 28, 35, 42], dtype=np.uint64)
        """We also keep track of the currently active player, the number of moves 
        played and the winner
        """
        self._current_player = 0
        self._move_count = 0
        self._winner = -1

    @property
    def current_player(self) -> int:
        return self._current_player

    @classmethod
    def num_players(cls) -> int:
        return 2

    def get_reward(self, player: int) -> float:
        if self._winner == -1:
            return 0
        elif self._winner == player:
            return 1
        else:
            return -1

    def get_reward_normalized(self, player: int) -> float:
        if self._winner == -1:
            return 0.5
        else:
            return float(self._winner == player)

    def list_moves(self):
        """To determine the possible moves we mask the bits where the next moves
        would be played and check if they are in the spillover-row of the bitboard.
        If they are then the column is full.
        """
        # top = '1000000_1000000_1000000_1000000_1000000_1000000_1000000'
        top = np.uint64(283691315109952)
        height_shifts = np.left_shift(1, self._heights)
        moves = np.array([i for i, h in enumerate(height_shifts) if not (top & h)])
        return moves.tolist()

    def is_terminal(self):
        """The game is over if 42 moves have been played or the winner has been found"""
        return self._move_count >= 42 or self._winner != -1

    def play_move(self, move: int):

        if move not in self.list_moves():
            raise ValueError("Move {} not possible. Possible moves {}".format(move, self.list_moves()))

        """Play a stone in the given column and check for a win"""
        player = self._current_player

        """First grab the bitboard of the currently active player. Then create a 
        bitmask with a single bit in the spot we want to play and set the bit in the 
        bitboard
        """
        self._player_boards[player] ^= (np.uint64(1) << self._heights[move])
        """Increase the height index"""
        self._heights[move] += 1

        is_terminal = self._is_win(self._player_boards[player])
        if is_terminal:
            self._winner = player

        self._move_count += 1
        self._current_player = 1 - player

    def _is_win(self, board: np.uint64):
        """Check for a win in all 4 directions at once. This is done by shifting the
        bitboard and combining it (using the binary AND operation) with the original
        board.
        Let's look at two individual columns, one with win and one without
            0                       0
            0                       1
            1                       1
            1                       0
            1                       1
            1                       1
        As a bitboard this looks like 001111 (011011). After >> 1 we get 000111 (001101)
        and combined with the original board we're left with

          001111                011011
        & 000111              & 001101
        --------              --------
          000111                001001

        Shifting the original board one place to the right essentially collapses pairs of
        adjacent bits into 1 bit per pair. 11 => 1, 111 => 11, 1111 => 111
        Now shifting the new board over twice results in 000001 (000010) and after combining

          000111                001001
        & 000001              & 000010
        --------              --------
          000001                000000

        Shifting the bitboard 7 places is equivalent to shifting columns - we store
        7 bits per column although the board is only 6 rows high, to prevent these shifts
        from "spilling over" into the next column - and 6(8) is shifting along the
        diagonals
        """
        directions = np.array([1, 6, 7, 8])
        # shift the board in all 4 directions. returns an array with the board shifted down, right, diag_up and
        # diag_down
        shift1 = board >> directions
        board &= shift1
        shift2 = board >> (2 * directions)
        res = board & shift2
        return np.max(res) > 0

    def copy(self):
        inst = ConnectFour()
        inst._heights = self._heights.copy()
        inst._player_boards = self._player_boards.copy()
        inst._winner = self._winner
        inst._current_player = self._current_player
        inst._move_count = self._move_count
        return inst

    def __hash__(self):
        mask = self._player_boards[0] | self._player_boards[1]
        # bottom = '0000001_0000001_0000001_0000001_0000001_0000001_0000001'
        bottom = np.uint64(4432676798593)
        return int(self._player_boards[self._current_player] + mask + bottom)

    def __repr__(self):
        output = np.zeros((6, 7))
        for j in range(6):
            for i in range(7):
                index = i * 7 + j
                m = np.uint64(1 << index)
                if self._player_boards[0] & m:
                    output[5 - j, i] = 1
                elif self._player_boards[1] & m:
                    output[5 - j, i] = 2

        output = output.__str__()
        output += "\nActive Player {} Moves made {}".format(self._current_player, self._move_count)
        return output

    @staticmethod
    def create_from_array(input_board):
        c4 = ConnectFour()

        b = np.array(input_board)
        b = np.reshape(b, (6, 7))

        boards = np.zeros(2, dtype=np.uint64)
        move_count = 0
        heights = np.array([0, 7, 14, 21, 28, 35, 42], dtype=np.uint64)

        for j in range(6):
            for i in range(7):
                index = i * 7 + j
                m = np.uint64(1 << index)
                board_index = None
                if b[5 - j, i] == 1:
                    board_index = 0
                elif b[5 - j, i] == 2:
                    board_index = 1

                if board_index != None:
                    boards[board_index] |= m
                    move_count += 1
                    heights[i] = index + 1

        active_player = move_count & 1

        c4._player_boards = boards
        c4._move_count = move_count
        c4._current_player = active_player
        c4._heights = heights
        winner = [i for i in range(2) if c4._is_win(c4._player_boards[i])]
        if len(winner) > 0:
            c4._winner = winner[0]
        return c4


class Node:
    def __init__(self, state):
        self.state = state

        self.reward = 0
        self.visits = 0

        self.children = {}
        self.available_actions = state.list_moves()

    def is_expanded(self):
        return len(self.available_actions) == 0

    def get_random_move(self):
        m = random.choice(self.available_actions)
        self.available_actions.remove(m)
        return m

    @property
    def value(self):
        if self.visits:
            return self.reward / self.visits
        else:
            return 0

    def __repr__(self):
        return f"Node({self.reward} {self.visits} {list(self.children.keys())})"


class MCTS:
    def __init__(self, game):
        self.root = Node(game)

    def best_child(self, node: Node, c_p=1.0):
        p_visits = node.visits

        nodes = node.children.items()
        max_val = -9999999
        max_node = None
        all_visits = np.log(p_visits)

        best_action = None
        for a, n in nodes:
            v = n.value + c_p * np.sqrt(2 * all_visits / n.visits)
            if v > max_val:
                max_val = v
                max_node = n
                best_action = a
        return max_node, best_action

    def descend_tree(self, c_p=0.8):
        n = self.root
        path = [n]

        while not n.state.is_terminal():
            if not n.is_expanded():
                # expand child
                next_move = n.get_random_move()
                next_state = n.state.copy()
                next_state.play_move(next_move)
                next_node = Node(next_state)

                n.children[next_move] = next_node
                path.append(next_node)
                return path

            else:
                next_node, next_move = self.best_child(n, c_p=c_p)
                n = next_node
                path.append(n)
        return path

    def evaluate(self, node):
        game_state = node.state.copy()
        player = 1 - game_state.current_player

        while not game_state.is_terminal():
            m = random.choice(game_state.list_moves())
            game_state.play_move(m)

        rew = game_state.get_reward(player)

        return rew

    def backup(self, path, reward):

        for n in reversed(path):
            n.visits += 1
            n.reward += reward
            reward = -reward

    def search_tree(self, c_p):
        path = self.descend_tree(c_p)
        reward = self.evaluate(path[-1])
        self.backup(path, reward)

    @staticmethod
    def run_search(game, steps=200, c_p=0.8):
        mcts = MCTS(game)

        for _ in range(steps):
            mcts.search_tree(c_p)

        most_visits = -999
        action = None

        for a, n in mcts.root.children.items():
            if n.visits > most_visits:
                most_visits = n.visits
                action = a

        return action


def player(observation, configuration):
    start = time.time()
    game = ConnectFour.create_from_array(observation.board)
    mcts = MCTS(game)

    available_time = configuration.timeout * 0.95
    while (time.time() - start) < available_time:
        mcts.search_tree(c_p=0.8)

    most_visits = -999
    action = None

    for a, n in mcts.root.children.items():
        if n.visits > most_visits:
            most_visits = n.visits
            action = a

    return action