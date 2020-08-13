from typing import List
import math
import numpy as np

from bachelorarbeit.games import Observation, Configuration, ConnectFour
from bachelorarbeit.mcts import Node, MCTSPlayer


class RaveNode(Node):
    def __init__(self, *args, **kwargs):
        super(RaveNode, self).__init__(*args, **kwargs)
        self.rave_count = 0
        self.rave_score = 0

    def __repr__(self):
        return f"RaveNode(n: {self.number_visits}, v: {self.total_value}, rave_n: {self.rave_count}, rave_v: {self.rave_score})"


class RavePlayer(MCTSPlayer):
    name = "RavePlayer"

    def __init__(self, *args, **kwargs):
        super(RavePlayer, self).__init__(*args, **kwargs)

    def reset(self):
        super(RavePlayer, self).reset()

    def get_move(self, observation: Observation, conf: Configuration) -> int:
        self.reset()
        root_game = ConnectFour(
            columns=conf.columns,
            rows=conf.rows,
            inarow=conf.inarow,
            mark=observation.mark,
            board=observation.board
        )
        root = RaveNode(game_state=root_game)

        while self.has_resources():
            leaf = self.find_leaf(root)
            child = self.expand(leaf)
            reward = self.evaluate_game_state(child.game_state)
            self.backup(child, reward)

        best = self.best_move(root)
        return best

if __name__ == "__main__":
    from bachelorarbeit.games import Observation, Configuration, ConnectFour
    from bachelorarbeit.tools import timer

    steps = 2000
    p = RavePlayer(max_steps=steps, exploration_constant=1.2)
    conf = Configuration()
    game = ConnectFour(columns=conf.columns, rows=conf.rows, inarow=conf.inarow, mark=1)
    obs = Observation(board=game.board.copy(), mark=game.mark)

    with timer(f"{steps} steps"):
        p.get_move(obs, conf)