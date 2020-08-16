from typing import List, Dict, Union
import math
import numpy as np
import random

from bachelorarbeit.games import Observation, Configuration, ConnectFour
from bachelorarbeit.rave import RaveNode, RavePlayer


class SarsaNode(RaveNode):
    def __init__(self, is_max: bool = None, *args, **kwargs):
        super(SarsaNode, self).__init__(*args, **kwargs)
        self.v = 0.5
        self.max_v = 0.501
        self.min_v = 0.499
        if is_max is None:
            self.is_max = not self.parent.is_max
        else:
            self.is_max = is_max

    def normalizeQ(self):
        max_v = self.parent.max_v
        min_v = self.parent.min_v

        return (self.v - min_v) / (max_v - min_v)

    def Q(self):
        return self.normalizeQ()

    def best_child(self, exploration_constant: float = 1.0, beta: float = 0.9) -> "RaveNode":
        n_p = math.log(self.number_visits)

        if self.is_max:
            flip = 1
        else:
            flip = -1

        _, c = max(self.children.items(),
                   key=lambda c: flip * ((1 - beta) * c[1].Q() + beta * c[1].QRave())
                                 + exploration_constant * c[1].exploration(n_p))
        return c

    def __repr__(self):
        return f"SarsaNode(n: {self.number_visits}, v: {self.v}, rave_n: {self.rave_count}, rave_v: {self.rave_score})"


class SarsaPlayer(RavePlayer):
    name = "SarsaPlayer"

    def __init__(self, gamma: float = 1.0, lamda: float = 1.0, beta: float = 0.5, *args, **kwargs):
        super(SarsaPlayer, self).__init__(beta=beta, *args, **kwargs)
        self.gamma = gamma
        self.lamda = lamda
        self.mark = 1

    def backup_sarsa(self, node: SarsaNode, reward: float, moves: List[int], sim_steps: int):
        move_set = set(moves)

        # delta_sum = math.pow(self.lamda * self.gamma, sim_steps - 1) * reward


        delta_sum = 0
        v_next = 0
        v_playout = 0.5
        R = reward
        for s in range(sim_steps):
            v_current = v_playout
            delta = R + self.gamma * v_next - v_current
            delta_sum = self.lamda * self.gamma * delta_sum + delta
            v_next = v_current
            R = 0

        # print(f"Reward {reward} discounted {delta_sum}")
        current = node
        # v_next = 0
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

            if last_v < current.min_v:
                current.min_v = last_v

            last_v = current.v
            # print(current)

            v_next = v_current
            for mov, child in current.rave_children.items():
                if mov in move_set:
                    child.rave_count += 1
                    child.rave_score += reward
            current = current.parent

    def reset(self):
        super(SarsaPlayer, self).reset()

        if not self.keep_tree:
            self.global_stats = {"max": 0, "min": 0}

    def evaluate_game_state_rave(self, game_state: ConnectFour, moves: List[int], move_counts: List[int]) -> float:
        game = game_state.copy()
        while not game.is_terminal():
            m = random.choice(game.list_moves())
            move_name = 10 * (move_counts[m] * len(move_counts) + m) + game.get_current_player()
            move_counts[m] += 1
            moves.append(move_name)
            game.play_move(m)

        return (game.get_reward(self.mark) + 1) / 2

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
            root = SarsaNode(game_state=root_game, is_max=True)

        self.mark = observation.mark

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

    steps = 20
    pl = SarsaPlayer(max_steps=steps, exploration_constant=0.1, beta=0.5, gamma=1.0, lamda=0.98)
    conf = Configuration()
    game = ConnectFour(columns=conf.columns, rows=conf.rows, inarow=conf.inarow, mark=1)
    obs = Observation(board=game.board.copy(), mark=game.mark)

    with timer(f"{steps} steps"):
        m = pl.get_move(obs, conf)
        print(m)
    exit()
    pl.max_steps = 200
    for i in range(15):
        game = ConnectFour(columns=conf.columns, rows=conf.rows, inarow=conf.inarow, mark=1)
        print("Testing game play")
        while not game.is_terminal():
            obs = Observation(board=game.board.copy(), mark=game.mark)
            m = pl.get_move(obs, conf)
            game.play_move(m)