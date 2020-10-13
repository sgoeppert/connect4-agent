from dataclasses import dataclass
from typing import List

@dataclass
class Configuration:
    rows: int = 6
    columns: int = 7
    inarow: int = 4
    timeout: int = 0
    actTimeout: int = 0
    agentTimeout: int = 0


@dataclass
class Observation:
    board: List
    mark: int = 1


class ConnectFour:
    def __init__(
            self,
            columns: int = 7,
            rows: int = 6,
            inarow: int = 4,
            mark: int = 1,
            board: List[int] = None
    ):
        if board is not None:
            self.board = board
        else:
            self.board = [0] * rows * columns
        self.cols = columns
        self.rows = rows
        self.mark = mark
        self.inarow = inarow
        self.stones_per_column = [0] * columns
        self.finished = False
        self.winner = None

    def is_terminal(self) -> bool:
        return self.finished

    def list_moves(self) -> List[int]:
        return [c for c in range(self.cols) if self.board[c] == 0]

    def get_current_player(self) -> int:
        return self.mark

    def get_other_player(self, player: int) -> int:
        return 3 - player

    def play_move(self, col: int) -> "ConnectFour":
        try:
            row = max([r for r in range(self.rows) if self.board[col + (r * self.cols)] == 0])
            self.board[col + (row * self.cols)] = self.mark
            self.stones_per_column[col] += 1
        except ValueError:
            raise RuntimeError("Invalid play")

        if self.is_win(col, self.mark, has_played=True):
            self.finished = True
            self.winner = self.mark
        elif self.is_tie():
            self.finished = True
            self.winner = None

        self.mark = 3 - self.mark
        return self

    def is_win(self, column: int, mark: int, has_played: bool = False) -> bool:
        columns = self.cols
        rows = self.rows
        inarow = self.inarow - 1
        row = (
            min([r for r in range(rows) if self.board[column + (r * columns)] == mark])
            if has_played
            else max([r for r in range(rows) if self.board[column + (r * columns)] == 0])
        )

        def count(offset_row: int, offset_column: int) -> int:
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

    def is_tie(self) -> bool:
        return all(mark != 0 for mark in self.board)

    def get_move_name(self, column: int, played: bool = False) -> int:
        stones_in_col = self.stones_per_column[column]
        player = self.get_current_player()
        if played:
            stones_in_col -= 1
            player = self.get_other_player(player)
        return 10 * (stones_in_col * self.cols + column) + player

    def get_reward(self, mark: int) -> int:
        if not self.finished:
            raise RuntimeError("get_reward called on non terminal game")

        if mark == self.winner:
            return 1
        elif self.winner is None:
            return 0
        else:
            return -1

    def copy(self) -> "ConnectFour":
        cp = ConnectFour(board=self.board[:],
                         columns=self.cols,
                         rows=self.rows,
                         mark=self.mark,
                         inarow=self.inarow)
        cp.finished = self.finished
        cp.winner = self.winner
        cp.stones_per_column = self.stones_per_column[:]
        return cp

    def hash(self) -> int:
        return hash(tuple(self.board))

    def __hash__(self):
        return self.hash()
