from old.games.connectfour import ConnectFour
from contextlib import contextmanager
from dataclasses import dataclass
from typing import List
import time
import logging
import os
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

@dataclass
class Configuration:
    rows: int = 6
    columns: int = 7
    inarow: int = 4
    timeout: int = 1

@dataclass
class Observation:
    board: List
    mark: int = 1


@contextmanager
def timer(label):
    start = time.time()
    yield
    print(f"{label} executed in {time.time() - start}s")


def play_game(p1, p2):
    conf = Configuration()
    g = ConnectFour(columns=conf.columns, rows=conf.rows, inarow=conf.inarow, mark=1)
    s = 0
    while not g.is_terminal():
        obs = Observation(board=g.board.copy(), mark=g.mark)
        if s % 2 == 0:
            m = p1(obs, conf)
        else:
            m = p2(obs, conf)
        g.play_move(m)
        s += 1
    return [g.get_reward(1), g.get_reward(2)]


def setup_logger(name, log_file, level=logging.INFO):

    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger