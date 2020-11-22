import json
import os
from datetime import datetime
from typing import  Optional, Type
import numpy as np
from pathlib import Path
from contextlib import contextmanager
import time
import numbers

from bachelorarbeit.selfplay import MoveEvaluation
from bachelorarbeit.players.base_players import Player
import config


def run_move_evaluation_experiment(
        title: str,
        player: Type[Player],
        player_config: Optional[dict] = None,
        num_processes: int = config.NUM_PROCESSES,
        max_tasks: Optional[int] = None,
        repeats: int = 1,
        show_progress_bar: bool = False
):
    """
    Bewertet die Gena
    :param title:
    :param player:
    :param player_config:
    :param num_processes:
    :param max_tasks:
    :param repeats:
    :param show_progress_bar:
    :return:
    """
    dataset_file = str(Path(config.ROOT_DIR) / "auswertungen" / "data" / "refmoves1k_kaggle")

    good, perfect, total = 0, 0, 0
    g = []
    p = []
    for it in range(repeats):
        evaluator = MoveEvaluation(
            player=player,
            player_config=player_config,
            dataset_file=dataset_file,
            num_processes=num_processes,
            max_tasks=max_tasks
        )
        _good, _perfect, _total = evaluator.score_player(show_progress_bar)
        g.append(_good / _total)
        p.append(_perfect / _total)
        good += _good
        perfect += _perfect
        total += _total

    p_std = np.std(p)
    g_std = np.std(g)

    return {
        "title": title,
        "player": player.name,
        "configuration": player_config,
        "repeats": repeats,
        "n_positions": total // repeats,
        "perfect_pct": perfect / total,
        "good_pct": good / total,
        "perfect_std": p_std,
        "good_std": g_std,
        "raw_results": {
            "total": total,
            "perfect": p,
            "good": g,
        }
    }


def get_range(center, num_vals=5, step=0.1):
    vals_left = num_vals // 2
    vals_right = num_vals - vals_left - 1

    values = [center - (i * step) for i in range(vals_left, 0, -1)]
    values += [center + (i * step) for i in range(vals_right + 1)]
    return list(map(lambda v: round(v, 4), values))


def dump_json(filename, data):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_file = filename.format(timestamp)
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    with open(out_file, "w+") as f:
        json.dump(data, f)
    print("Wrote results to file ", out_file)


"""
Verschiedene Methoden für die Transformation des Spielfelds in eine Repräsentation für das neuronale Netz:
"""


def transform_board_large(board):
    b = np.asarray(board)
    if b.ndim == 1:
        b.shape = (1,) + b.shape
    return np.append(b == 1, b == 2, axis=1).astype(int).tolist()


def transform_board(board):
    b = np.asarray(board)
    if b.ndim == 1:
        b.shape = (1,) + b.shape
    return (b / 2).tolist()


def transform_board_nega(board):
    b = np.asarray(board)
    if b.ndim == 1:
        b.shape = (1,) + b.shape
    b[b == 2] = -1
    return b.tolist()


def transform_board_cnn(board):
    b = np.asarray(board)
    if b.ndim == 1:
        b.shape = (1,) + b.shape
    owner = 1 - (np.count_nonzero(b, axis=-1) % 2)
    new_board = np.concatenate((b == 1, b == 2, [[x] * b.shape[1] for x in owner]), axis=1).reshape((-1,3,6,7))
    return np.moveaxis(new_board, -3, -1).tolist()


def normalize(val):
    return (val + 1) / 2


def denormalize(val):
    return (2 * val) - 1


def flip_board(board, rows=6, cols=7):
    return np.array(board).reshape((rows, cols))[:, ::-1].reshape(-1).tolist()


@contextmanager
def timer(name="Timer"):
    tick = time.time()
    yield
    tock = time.time()
    print(f"{name} took {tock - tick}s.")


class Table:
    """
    Eine Klasse für die Erstellung von Tabellen und die Ausgabe in einem für Latex geeigneten Format
    """
    def __init__(self):
        self.rows = []
        self.row_header = []
        self.col_header = []
        self.top_left = None
        self.row_length = 0
        self.label = None
        self.caption = None

    def _check_row_length(self, row, padding=None, padding_value=None):
        if len(row) != self.row_length:
            if padding is None:
                raise ValueError("The table row {} does not have the same length as the table. "
                                 "Got length {} expected length {}.".format(row, len(row), self.row_length))
            elif padding == "right":
                row = row + [padding_value] * (self.row_length - len(row))
            elif padding == "left":
                row = [padding_value] * (self.row_length - len(row)) + row

        return row

    def set_row_label(self, row: int, val: any):
        if len(self.row_header) == 0:
            self.row_header = [None] * (row + 1)

        if row > len(self.row_header) - 1:
            self.row_header += [None] * (row - len(self.row_header) + 1)
        self.row_header[row] = val

    def set_col_head(self, col: int, val: any):
        self.col_header[col] = val

    def set_full_col_header(self, row: list):
        if len(self.rows) != 0 and len(row) != self.row_length:
            raise ValueError("Length of the column header {} (length {}) does not "
                             "match the number of columns {}".format(row, len(row), self.row_length))

        self.col_header = row

    def add_row(self, row: list, label=None, padding=None, padding_value=None):
        # store the length if this is the first row. this defines the number of columns in the table
        if len(self.rows) == 0:
            self.row_length = len(row)
            if len(self.col_header) == 0:
                self.col_header = [None] * self.row_length

        else:
            row = self._check_row_length(row, padding, padding_value)

        self.rows.append(row)
        r_index = len(self.rows) - 1
        self.set_row_label(r_index, label)
        # self.row_header.append(label)

    def insert_row(self, pos: int, row: list, label=None, padding=None, padding_value=None):
        row = self._check_row_length(row, padding, padding_value)
        self.rows.insert(pos, row)
        self.row_header.insert(pos, label)

    def add_column(self, col: list, head=None):
        # add padding to column
        if len(col) < len(self.rows):
            col = col + [None] * (len(self.rows) - len(col))

        for i, c_val in enumerate(col):
            # if the column has more elements than the table has rows, add a new row and row_header
            if i >= len(self.rows):
                self.rows.append([None] * self.row_length)
                self.row_header.append(None)

            # add the column value to the row
            self.rows[i].append(c_val)

        self.row_length += 1
        self.col_header.append(head)

    def insert_column(self, pos: int, col: list, head=None):
        if len(col) < len(self.rows):
            col = col + [None] * (len(self.rows) - len(col))

        for i, c_val in enumerate(col):
            if i < len(self.rows):
                self.rows[i].insert(pos, c_val)
            else:
                new_row = [None] * self.row_length
                new_row.insert(pos, c_val)
                self.rows.append(new_row)
                self.row_header.append(None)

        self.row_length += 1
        self.col_header.insert(pos, head)

    @staticmethod
    def format_cell(val, replacement=" "):
        if isinstance(val, numbers.Real):
            val = round(val, 3)

        return str(val) if val is not None else str(replacement)

    def print_latex(self):
        lines = []
        lines.append("\\begin{table}[h!]")
        lines.append("\\centering")

        prepend_column = self.top_left is not None or any(self.row_header)
        prepend_row = self.top_left is not None or any(self.col_header)

        cell_format = ["c"] * self.row_length
        if prepend_column:
            cell_format.insert(0, "c|")
        cell_format_string = "|" + ("|".join(cell_format)) + "|"
        lines.append("\\begin{tabular}{" + cell_format_string + "}")
        lines.append("\\hline")
        if prepend_row:
            header_list = [self.top_left] + self.col_header
            if any(header_list):
                header_line = " & ".join(map(Table.format_cell, header_list)) + " \\\\"
                lines.append(header_line)
            lines.append("\\hline")

        for i, row in enumerate(self.rows):
            if prepend_column:
                row = [self.row_header[i]] + row
            row_line = " & ".join(map(Table.format_cell, row)) + " \\\\"
            lines.append(row_line)
            lines.append("\\hline")

        lines.append("\\end{tabular}")
        if self.caption is not None:
            lines.append("\\caption{" + self.caption + "}")
        if self.label is not None:
            lines.append("\\label{tab:" + self.label + "}")
        lines.append("\\end{table}")

        return "\n".join(lines) + "\n"

    def print(self):
        lines = []
        lines.append("Table:")

        prepend_column = self.top_left is not None or any(self.row_header)
        prepend_row = self.top_left is not None or any(self.col_header)

        def subst(char):
            return Table.format_cell(char).center(8)

        if prepend_row:
            header_list = [self.top_left] + self.col_header
            if any(header_list):
                header_line = "| " + (" | ".join(map(subst, header_list))) + " |"
                lines.append(header_line)
                lines.append("=" * len(header_line))

        for i, row in enumerate(self.rows):
            if prepend_column:
                row = [self.row_header[i]] + row
            row_line = "| " + (" | ".join(map(subst, row))) + " |"
            lines.append(row_line)
            lines.append("-" * len(row_line))

        if self.caption is not None:
            lines.append("Caption: " + self.caption)
        if self.label is not None:
            lines.append("Label: " + self.label)

        return "\n".join(lines) + "\n"

    def write_to_file(self, file, ending=".txt", out_format="latex"):
        fname = Path(config.ROOT_DIR) / "tables" / (file + ending)
        os.makedirs(os.path.dirname(fname), exist_ok=True)

        if out_format == "latex":
            out_string = self.print_latex()
        else:
            out_string = self.print()

        with open(fname, "w+") as f:
            f.write(out_string)
