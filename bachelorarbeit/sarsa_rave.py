from typing import List, Dict, Union
import math
import numpy as np
import random

from bachelorarbeit.games import Observation, Configuration, ConnectFour
from bachelorarbeit.rave import RaveNode, RavePlayer


class SarsaNode(RaveNode):
    def __init__(self, global_stats: Union[Dict[str, float], None] = None, *args, **kwargs):
        super(SarsaNode, self).__init__(*args, **kwargs)
        self.v = 0
        self.max_v = 0.01
        self.min_v = -0.01

        if global_stats is None:
            self.global_stats = self.parent.global_stats
        else:
            self.global_stats = global_stats

    def normalizeQ(self):
        # max_v = min(self.parent.max_v, self.global_stats["max"])
        # min_v = max(self.parent.min_v, self.global_stats["min"])
        max_v = self.parent.max_v
        min_v = self.parent.min_v
        #
        # if max_v == min_v:
        #     max_v = self.global_stats["max"]
        #     min_v = self.global_stats["min"]

        # if max_v <= -1:
        #     max_v = self.global_stats["max"]
        # if min_v >= 1:
        #     min_v = self.global_stats["min"]

        return (self.v - min_v) / (max_v - min_v)

    def Q(self):
        return self.normalizeQ()

    def __repr__(self):
        return f"SarsaNode(n: {self.number_visits}, v: {self.v}, rave_n: {self.rave_count}, rave_v: {self.rave_score})"


class SarsaPlayer(RavePlayer):
    name = "SarsaPlayer"

    def __init__(self, gamma: float = 1.0, lamda: float = 1.0, beta: float = 0.5, *args, **kwargs):
        super(SarsaPlayer, self).__init__(beta=beta, *args, **kwargs)
        self.gamma = gamma
        self.lamda = lamda
        self.global_stats = {"max": 0, "min": 0}

    def backup_sarsa(self, node: SarsaNode, reward: float, moves: List[int], sim_steps: int):
        move_set = set(moves)

        delta_sum = math.pow(self.lamda * self.gamma, sim_steps - 1) * reward

        # print(f"Reward {reward} discounted {delta_sum}")

        current = node
        v_next = 0
        last_v = 0
        while current is not None:
            v_current = current.v
            # print("v_next", v_next, "v_current", v_current)
            delta = self.gamma * v_next - v_current
            delta_sum = self.lamda * self.gamma * delta_sum + delta

            # print("delta", delta)
            # print("delta_sum", delta_sum)

            current.number_visits += 1
            current.v += delta_sum / current.number_visits

            if last_v > current.max_v:
                current.max_v = last_v
                if last_v > self.global_stats["max"]:
                    self.global_stats["max"] = last_v

            if last_v < current.min_v:
                current.min_v = last_v
                if last_v < self.global_stats["min"]:
                    self.global_stats["min"] = last_v

            last_v = current.v
            # print(current)

            delta_sum = -delta_sum
            v_next = -v_current
            for mov, child in current.rave_children.items():
                if mov in move_set:
                    child.rave_count += 1
                    child.rave_score += -reward
            reward = -reward
            current = current.parent

    def reset(self):
        super(SarsaPlayer, self).reset()

        if not self.keep_tree:
            self.global_stats = {"max": 0, "min": 0}

    def get_move(self, observation: Observation, conf: Configuration) -> int:
        self.reset()

        root = None
        # if we're keeping the tree and have a stored node, try to determine the opponents move and apply that move
        # the resulting node is out starting root-node
        if self.keep_tree and self.root is not None:
            root = self.root
            opp_move = self.determine_opponent_move(observation.board, self.root.game_state.board, conf.columns)
            if opp_move in root.children:
                root = root.children[opp_move]
                root.parent = None
            else:
                root = None

        # if no root could be determined, create a new tree from scratch
        if root is None:
            root_game = ConnectFour(
                columns=conf.columns,
                rows=conf.rows,
                inarow=conf.inarow,
                mark=observation.mark,
                board=observation.board
            )
            root = SarsaNode(game_state=root_game, global_stats=self.global_stats)

        while self.has_resources():
            moves = []
            move_counts = SarsaPlayer.init_move_counts(root.game_state)

            leaf = self.tree_policy_rave(root, moves, move_counts)
            n_steps = len(moves)
            reward = self.evaluate_game_state_rave(leaf.game_state, moves, move_counts)
            sim_steps = len(moves) - n_steps

            self.backup_sarsa(leaf, reward, moves, sim_steps)

        # print(root.children)
        best = self.best_move(root)
        if self.keep_tree:
            self.root = root.children[best]
            self.root.parent = None

        return best


if __name__ == "__main__":
    from bachelorarbeit.games import Observation, Configuration, ConnectFour
    from bachelorarbeit.tools import timer

    steps = 2000
    pl = SarsaPlayer(max_steps=steps, exploration_constant=0.9, beta=0.5, gamma=1.0, lamda=0.98)
    conf = Configuration()
    game = ConnectFour(columns=conf.columns, rows=conf.rows, inarow=conf.inarow, mark=1)
    obs = Observation(board=game.board.copy(), mark=game.mark)

    with timer(f"{steps} steps"):
        m = pl.get_move(obs, conf)
        print(m)

    pl.max_steps = 200
    for i in range(15):
        game = ConnectFour(columns=conf.columns, rows=conf.rows, inarow=conf.inarow, mark=1)
        print("Testing game play")
        while not game.is_terminal():
            obs = Observation(board=game.board.copy(), mark=game.mark)
            m = pl.get_move(obs, conf)
            game.play_move(m)