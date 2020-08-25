from typing import List, Dict, Tuple
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

    def beta(self, b: float = 0.0) -> float:
        return self.rave_count / (self.rave_count + self.number_visits + self.rave_count * self.number_visits * b * b)

    def QRave(self) -> float:
        return self.rave_score

    def exploration(self, parent_visits):
        return math.sqrt(parent_visits / self.number_visits)

    def best_child(self, C_p: float = 1.0, b: float = 0.0) -> Tuple["RaveNode", int]:
        n_p = math.log(self.number_visits)

        def UCT_Rave(child: RaveNode):
            beta = child.beta(b)
            return (1 - beta) * child.Q() + beta * child.QRave() \
                   + C_p * math.sqrt(n_p / child.number_visits)

        m, c = max(self.children.items(), key=lambda c: UCT_Rave(c[1]))
        return c, m

    def expand_one_child_rave(self, moves: List[int]) -> "RaveNode":
        nodeclass = type(self)
        move = random.choice(self.possible_moves)  # wähle zufälligen Zug
        child_state = self.game_state.copy().play_move(move)
        move_name = child_state.get_move_name(move, played=True)
        moves.append(move_name)
        self.children[move] = nodeclass(game_state=child_state, parent=self)
        self.rave_children[move_name] = self.children[move]
        self.possible_moves.remove(move)
        if len(self.possible_moves) == 0:
            self.expanded = True

        return self.children[move]

    def increment_rave_visit_and_add_reward(self, reward):
        self.rave_count += 1
        self.rave_score += (reward - self.rave_score) / self.rave_count

    def __repr__(self):
        return f"RaveNode(n: {self.number_visits}, v: {self.average_value}, rave_n: {self.rave_count}, rave_v: {self.rave_score})"


class RavePlayer(MCTSPlayer):
    name = "RavePlayer"

    def __init__(self, b: float = 0.0, *args, **kwargs):
        super(RavePlayer, self).__init__(*args, **kwargs)
        self.b = b

    def tree_policy_rave(self, root: "RaveNode", moves) -> "RaveNode":
        current = root

        while not current.game_state.is_terminal():
            if current.is_expanded():
                current, m = current.best_child(self.exploration_constant, b=self.b)
                # Hole den eindeutigen Namen des Spielzugs und merke ihn
                move_name = current.game_state.get_move_name(m, played=True)
                moves.append(move_name)
            else:
                return current.expand_one_child_rave(moves)

        return current

    def evaluate_game_state_rave(self, game_state: ConnectFour, moves: List[int]) -> float:
        game = game_state.copy()
        scoring = game.get_other_player(game.get_current_player())
        while not game.is_terminal():
            m = random.choice(game.list_moves())
            move_name = game.get_move_name(m)
            moves.append(move_name)  # merke welche Züge gespielt wurden
            game.play_move(m)

        return game.get_reward(scoring)

    def backup_rave(self, node: RaveNode, reward: float, moves: List[int]):
        move_set = set(moves)
        current = node
        while current is not None:
            current.increment_visit_and_add_reward(reward)

            # Wenn eines der Kinder dieses Knotens über einen in der Simulation gemachten Spielzug
            # erreicht werden könnte, aktualisiere seine Rave Statistik
            for mov, child in current.rave_children.items():
                if mov in move_set:
                    child.increment_rave_visit_and_add_reward(-reward)

            reward = -reward
            current = current.parent

    def perform_search(self, root):
        while self.has_resources():
            moves = []  # die Liste aller gemachten Spielzüge in dieser Iteration
            leaf = self.tree_policy_rave(root, moves)
            reward = self.evaluate_game_state_rave(leaf.game_state, moves)
            self.backup_rave(leaf, reward, moves)

        return self.best_move(root)

    def init_root_node(self, root_game):
        return RaveNode(game_state=root_game)

    def best_move(self, node: RaveNode) -> int:
        move, n = max(node.children.items(), key=lambda c: (1 - c[1].beta(0)) * c[1].Q() + c[1].beta(0) * c[1].QRave())
        return move


if __name__ == "__main__":
    from bachelorarbeit.games import Observation, Configuration, ConnectFour
    from bachelorarbeit.tools import timer

    steps = 200
    pl = RavePlayer(max_steps=steps, exploration_constant=0.5, b=0.0)
    conf = Configuration()
    game = ConnectFour(columns=conf.columns, rows=conf.rows, inarow=conf.inarow, mark=1)
    obs = Observation(board=game.board.copy(), mark=game.mark)

    with timer(f"{steps} steps"):
        m = pl.get_move(obs, conf)
        print(m)
