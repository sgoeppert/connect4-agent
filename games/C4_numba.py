from numba.experimental import jitclass
from numba import jit
from numba import int64
import numpy as np

spec = [
    ('_player_boards', int64[:]),
    ('_heights', int64[:]),
    ('_move_count', int64),
    ('_current_player', int64),
    ('_winner', int64),
]


@jitclass(spec)
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

    def get_current_player(self):
        return self._current_player

    def get_other_player(self, player):
        return 1 - player

    def num_players(self) -> int:
        return 2

    def get_action_space(self) -> int:
        return 7

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
            return 1.0 if (self._winner == player) else 0.0

    def list_moves(self):
        """To determine the possible moves we mask the bits where the next moves
        would be played and check if they are in the spillover-row of the bitboard.
        If they are then the column is full.
        """
        # top = '1000000_1000000_1000000_1000000_1000000_1000000_1000000'
        top = np.uint64(283691315109952)
        height_shifts = np.left_shift(1, self._heights)
        moves = np.array([i for i, h in enumerate(height_shifts) if not (top & h)])
        return list(moves)

    def is_terminal(self):
        """The game is over if 42 moves have been played or the winner has been found"""
        return self._move_count >= 42 or self._winner != -1

    def play_move(self, move: int):

        """Play a stone in the given column and check for a win"""
        player = self._current_player

        if move not in self.list_moves():
            raise ValueError

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
        board = board & shift1
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


    def encoded_pytorch(self) -> np.ndarray:
        """
        Encode the game state with channels first
        :return:
        """
        output = np.zeros((3, 6, 7))
        for row in range(6):
            for col in range(7):
                index = col * 7 + row
                m = np.uint64(1 << index)
                if self._player_boards[0] & m:
                    output[0, row, col] = 1
                elif self._player_boards[1] & m:
                    output[1, row, col] = 1

        if self._current_player == 1:
            output[2, :, :] = 1

        return output

    def mirror(self):
        def mirror_board(board):
            # print("Original", np.binary_repr(board, width=64))
            newboard = np.uint64(0)
            mask = np.uint64(2 ** 7 - 1)
            for j in range(7):
                shift = np.uint64(7)
                # print(type(board))
                shifted = np.left_shift(newboard, shift)
                masked = np.uint64(((board >> np.uint64(shift * j)) & mask))
                newboard = shifted | masked
                # print(np.binary_repr(newboard, width=64), "\n")
            return newboard

        new_boards = np.array([mirror_board(board) for board in self._player_boards], dtype=np.uint64)

        new_heights = self._heights[::-1] % np.uint64(7)
        for i in range(len(new_heights)):
            new_heights[i] += i * 7

        game = self.copy()
        game._player_boards = new_boards
        game._heights = new_heights.astype(np.uint64)

        return game

    def hash(self):
        return hash(self)

    def __hash__(self):
        mask = self._player_boards[0] | self._player_boards[1]
        # bottom = '0000001_0000001_0000001_0000001_0000001_0000001_0000001'
        bottom = np.uint64(4432676798593)
        return int(self._player_boards[self._current_player] + mask + bottom)

    def display(self):
        output = np.zeros((6, 7))
        for j in range(6):
            for i in range(7):
                index = i * 7 + j
                m = np.uint64(1 << index)
                if self._player_boards[0] & m:
                    output[5 - j, i] = 1
                elif self._player_boards[1] & m:
                    output[5 - j, i] = 2

        return output


    def __repr__(self):
        output = self.display().__str__()
        output += "\nActive Player {} Moves made {}".format(self._current_player, self._move_count)
        return output


@jit(nopython=True)
def create_from_array(input_board):
    c4 = ConnectFour()

    # b = np.array(input_board).astype(np.uint64)
    b = np.reshape(input_board, (6, 7))

    boards = np.zeros(2, dtype=np.uint64)
    move_count = 0
    heights = np.array([0, 7, 14, 21, 28, 35, 42], dtype=np.uint64)

    for j in range(6):
        for i in range(7):
            index = i * 7 + j
            m = np.uint64(1 << index)
            board_index = -1
            if b[5 - j, i] == 1:
                board_index = 0
            elif b[5 - j, i] == 2:
                board_index = 1

            if board_index != -1:
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


# call some functions to force numba to compile the code
_g = ConnectFour()
_g.play_move(0)
_g.list_moves()
_g.is_terminal()

if __name__ == "__main__":
    pass
    # from lib.utils import timer
    # import random
    #
    # n_games = 10000
    # with timer("{} games".format(n_games)):
    #     for i in range(n_games):
    #         game = ConnectFour()
    #         while not game.is_terminal():
    #             m = random.choice(game.list_moves())
    #             game.play_move(m)

    # g = ConnectFour()
    # g.list_moves()
    # m = random.choice(g.list_moves())
    # print(m)
    # g.play_move(m)
    # mir = g.mirror()
    # print(mir.as_array())
    # print(hash(g))
    # print(g.hash())
    # print(g.encoded_pytorch())
    # print(g.copy())
    # print(g.is_terminal())
    # print(g.get_reward(0))
    # print(g.get_reward_normalized(0))
    # print(g.current_player)
    # print(g.get_action_space())
    # print(g.num_players())