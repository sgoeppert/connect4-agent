
EMPTY = 0

class ConnectFour:
    def __init__(self, columns=7, rows=6, inarow=4, mark=1, board=None):
        if board is not None:
            self.board = board
        else:
            self.board = [0] * rows * columns
        self.cols = columns
        self.rows = rows
        self.mark = mark
        self.inarow = inarow
        self.finished = False
        self.winner = None

    def is_terminal(self):
        return self.finished

    def list_moves(self):
        return [c for c in range(self.cols) if self.board[c] == 0]

    def get_action_space(self):
        return self.cols

    def get_current_player(self):
        return self.mark

    def get_other_player(self, player):
        return 3 - player

    def num_players(self):
        return 2

    def play_move(self, col):
        try:
            row = max([r for r in range(self.rows) if self.board[col + (r * self.cols)] == EMPTY])
            self.board[col + (row * self.cols)] = self.mark
        except ValueError:
            raise RuntimeError("Invalid play")

        if self.is_win(col, self.mark, has_played=True):
            self.finished = True
            self.winner = self.mark
        elif self.is_tie():
            self.finished = True
            self.winner = None

        self.mark = 3 - self.mark

    def is_win(self, column, mark, has_played=False):
        columns = self.cols
        rows = self.rows
        inarow = self.inarow - 1
        row = (
            min([r for r in range(rows) if self.board[column + (r * columns)] == mark])
            if has_played
            else max([r for r in range(rows) if self.board[column + (r * columns)] == EMPTY])
        )

        def count(offset_row, offset_column):
            for i in range(1, inarow + 1):
                r = row + offset_row * i
                c = column + offset_column * i
                if (
                        r < 0
                        or r >= rows
                        or c < 0
                        or c >= columns
                        or self.board[c + (r * columns)] != mark
                ):
                    return i - 1
            return inarow

        return (
                count(1, 0) >= inarow  # vertical.
                or (count(0, 1) + count(0, -1)) >= inarow  # horizontal.
                or (count(-1, -1) + count(1, 1)) >= inarow  # top left diagonal.
                or (count(-1, 1) + count(1, -1)) >= inarow  # top right diagonal.
        )

    def is_tie(self):
        return all(mark != 0 for mark in self.board)

    def get_reward(self, mark):
        if not self.finished:
            raise RuntimeError("get_score called on non terminal game")

        if mark == self.winner:
            return 1
        elif self.winner is None:
            return 0
        else:
            return -1

    def copy(self):
        cp = ConnectFour(board=self.board.copy(),
                         columns=self.cols,
                         rows=self.rows,
                         mark=self.mark,
                         inarow=self.inarow)
        cp.finished = self.finished
        cp.winner = self.winner
        return cp

    def hash(self):
        return hash(str(self.board))
