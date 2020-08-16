from typing import List, Dict
import math
import numpy as np
import random

from bachelorarbeit.games import Observation, Configuration, ConnectFour
from bachelorarbeit.mcts import Node, MCTSPlayer


class RaveNode(Node):
    def __init__(self, *args, **kwargs):
        super(RaveNode, self).__init__(*args, **kwargs)
        self.rave_count = 0
        self.rave_score = 0
        self.rave_children: Dict[int, "RaveNode"] = {}

    def __repr__(self):
        return f"RaveNode(n: {self.number_visits}, v: {self.total_value}, rave_n: {self.rave_count}, rave_v: {self.rave_score})"

    def QRave(self) -> float:
        return self.rave_score / max(1, self.rave_count)

    def Q(self) -> float:
        return self.total_value / max(1, self.number_visits)

    def exploration(self, parent_visits):
        if self.number_visits == 0:
            return 10
        return math.sqrt(parent_visits / self.number_visits)

    def best_child(self, exploration_constant: float = 1.0, beta: float = 0.9) -> "RaveNode":
        n_p = math.log(self.number_visits)
        _, c = max(self.children.items(),
                   key=lambda c: (1 - beta) * c[1].Q() + beta * c[1].QRave()
                                 + exploration_constant * c[1].exploration(n_p))
        return c

    def expand_one_child_rave(self, move_counts: List[int], moves: List[int]) -> "RaveNode":
        myclass = type(self)
        move = random.choice(self.possible_moves)
        child_state = self.game_state.copy()
        child_state.play_move(move)

        move_name = 10 * (move_counts[move] * len(move_counts) + move) + self.game_state.get_current_player()
        move_counts[move] += 1
        moves.append(move_name)
        self.children[move] = myclass(game_state=child_state, parent=self, move=move)
        self.rave_children[move_name] = self.children[move]
        self.possible_moves.remove(move)
        if len(self.possible_moves) == 0:
            self.expanded = True

        return self.children[move]

    def expand_all_children_rave(self, move_counts: List[int]):
        myclass = type(self)
        for move in self.possible_moves:
            child_state = self.game_state.copy()
            child_state.play_move(move)
            move_name = 10 * (move_counts[move] * len(move_counts) + move) + self.game_state.get_current_player()
            self.children[move] = myclass(game_state=child_state, parent=self, move=move)
            self.rave_children[move_name] = self.children[move]
        self.possible_moves = []
        self.expanded = True


class RavePlayer(MCTSPlayer):
    name = "RavePlayer"

    def __init__(self, beta: float = 0.9, expand_all=False, *args, **kwargs):
        super(RavePlayer, self).__init__(*args, **kwargs)
        self.beta = beta
        self.expand_all = expand_all

    def reset(self):
        super(RavePlayer, self).reset()

    @staticmethod
    def init_move_counts(state):
        board = state.board
        cols = state.cols
        move_counts = [0] * cols
        for i, v in enumerate(board):
            if v != 0:
                move_counts[i % cols] += 1

        return move_counts

    def tree_policy_rave(self, root: "RaveNode", moves, move_counts) -> "RaveNode":
        current = root
        action_space = current.game_state.get_action_space()

        while not current.game_state.is_terminal():
            if current.is_expanded():
                p = current.game_state.get_current_player()
                current = current.best_child(self.exploration_constant, beta=self.beta)
                m = current.move
                move_name = 10 * (move_counts[m] * action_space + m) + p
                move_counts[m] += 1
                moves.append(move_name)
            else:
                if self.expand_all:
                    current.expand_all_children_rave(move_counts)
                    return current
                else:
                    return current.expand_one_child_rave(move_counts, moves)

        return current

    def evaluate_game_state_rave(self, game_state: ConnectFour, moves: List[int], move_counts: List[int]) -> float:
        game = game_state.copy()
        scoring = game.get_other_player(game.get_current_player())
        while not game.is_terminal():
            m = random.choice(game.list_moves())
            move_name = 10 * (move_counts[m] * len(move_counts) + m) + game.get_current_player()
            move_counts[m] += 1
            moves.append(move_name)
            game.play_move(m)

        return game.get_reward(scoring)

    def backup_rave(self, node: RaveNode, reward: float, moves: List[int]):
        move_set = set(moves)
        current = node
        while current is not None:
            current.number_visits += 1
            current.total_value += reward

            for mov, child in current.rave_children.items():
                if mov in move_set:
                    child.rave_count += 1
                    child.rave_score += -reward

            reward = -reward
            current = current.parent

    def best_move(self, node: RaveNode) -> int:
        move, n = max(node.children.items(), key=lambda c: (1 - self.beta) * c[1].Q() + self.beta * c[1].QRave())
        return move

    def get_move(self, observation: Observation, conf: Configuration) -> int:
        self.reset()

        root = self._restore_root(observation, conf)

        # if no root could be determined, create a new tree from scratch
        if root is None:
            root_game = ConnectFour(
                columns=conf.columns,
                rows=conf.rows,
                inarow=conf.inarow,
                mark=observation.mark,
                board=observation.board
            )
            root = RaveNode(game_state=root_game)

        while self.has_resources():
            moves = []
            move_counts = RavePlayer.init_move_counts(root.game_state)

            leaf = self.tree_policy_rave(root, moves, move_counts)
            reward = self.evaluate_game_state_rave(leaf.game_state, moves, move_counts)
            self.backup_rave(leaf, reward, moves)

        # print(root.children)
        best = self.best_move(root)

        self._store_root(root.children[best])

        return best


if __name__ == "__main__":
    from bachelorarbeit.games import Observation, Configuration, ConnectFour
    from bachelorarbeit.tools import timer

    steps = 20
    pl = RavePlayer(max_steps=steps, exploration_constant=0.9,expand_all=True,beta=0.9)
    conf = Configuration()
    game = ConnectFour(columns=conf.columns, rows=conf.rows, inarow=conf.inarow, mark=1)
    obs = Observation(board=game.board.copy(), mark=game.mark)

    with timer(f"{steps} steps"):
        m = pl.get_move(obs, conf)
        print(m)
