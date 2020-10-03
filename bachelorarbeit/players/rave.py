from typing import List, Dict, Tuple
import math
import random

from bachelorarbeit.games import ConnectFour
from bachelorarbeit.players.mcts import Node, MCTSPlayer, Evaluator


class RaveEvaluator(Evaluator):
    def __init__(self, move_list):
        self.moves = move_list

    def __call__(self, game_state: ConnectFour):
        game = game_state.copy()
        scoring = game.get_other_player(game.get_current_player())
        while not game.is_terminal():
            m = random.choice(game.list_moves())
            move_name = game.get_move_name(m)
            self.moves.append(move_name)  # merke welche Züge gespielt wurden
            game.play_move(m)

        return game.get_reward(scoring)


class RaveNode(Node):
    def __init__(self, *args, **kwargs):
        super(RaveNode, self).__init__(*args, **kwargs)
        self.rave_count: int = 0
        self.rave_score: float = 0
        self.rave_children: Dict[int, "RaveNode"] = {}

    def beta(self, k: int = 1) -> float:
        return math.sqrt(k / (3 * self.number_visits + k))

    def QRave(self) -> float:
        return self.rave_score

    def exploration(self, parent_visits):
        return math.sqrt(parent_visits / self.number_visits)

    def best_child(self, C_p: float = 1.0, k: int = 1, alpha: float = None) -> Tuple["RaveNode", int]:
        n_p = math.log(self.number_visits)

        def UCT_Rave(child: RaveNode):
            if alpha is not None:
                beta = alpha
            else:
                beta = child.beta(k)
            return (1 - beta) * child.Q() + beta * child.QRave() \
                   + C_p * math.sqrt(n_p / child.number_visits)

        m, c = max(self.children.items(), key=lambda c: UCT_Rave(c[1]))
        return c, m

    def expand_one_child_rave(self, moves: List[int]) -> "RaveNode":
        nodeclass = type(self)
        move = random.choice(self.possible_moves)  # wähle zufälligen Zug

        self.possible_moves.remove(move)
        if len(self.possible_moves) == 0:
            self.expanded = True

        child_state = self.game_state.copy().play_move(move)
        move_name = child_state.get_move_name(move, played=True)
        moves.append(move_name)
        self.children[move] = nodeclass(game_state=child_state, parent=self)
        self.rave_children[move_name] = self.children[move]

        return self.children[move]

    def increment_rave_visit_and_add_reward(self, reward):
        self.rave_count += 1
        self.rave_score += (reward - self.rave_score) / self.rave_count

    def __repr__(self):
        return f"RaveNode(n: {self.number_visits}, v: {self.average_value}, rave_n: {self.rave_count}, rave_v: {self.rave_score})"


class RavePlayer(MCTSPlayer):
    name = "RavePlayer"

    def __init__(self, k: int = 100, alpha: float = None, *args, **kwargs):
        super(RavePlayer, self).__init__(*args, **kwargs)
        self.move_list = []
        self.evaluate = RaveEvaluator(self.move_list)
        self.k = k
        self.alpha = alpha

    def tree_policy(self, root: "RaveNode") -> "RaveNode":
        current = root

        while not current.game_state.is_terminal():
            if current.is_expanded():
                current, m = current.best_child(self.exploration_constant, k=self.k, alpha=self.alpha)
                # Hole den eindeutigen Namen des Spielzugs und merke ihn
                self.move_list.append(current.game_state.get_move_name(m, played=True))
            else:
                return current.expand_one_child_rave(self.move_list)

        return current

    def backup(self, node: RaveNode, reward: float):
        move_set = set(self.move_list)
        current = node
        while current is not None:
            current.number_visits += 1
            current.average_value += (reward - current.average_value) / current.number_visits

            # Wenn eines der Kinder dieses Knotens über einen in der Simulation gemachten Spielzug
            # erreicht werden könnte, aktualisiere seine Rave Statistik
            for mov, child in current.rave_children.items():
                if mov in move_set:
                    child.increment_rave_visit_and_add_reward(-reward)

            reward = -reward
            current = current.parent

    def perform_search(self, root):
        while self.has_resources():
            self.move_list.clear()  # Bereinige die move_list aber behalte die selbe Referenz bei, da der Evaluator sie braucht
            leaf = self.tree_policy(root)
            reward = self.evaluate(leaf.game_state)
            self.backup(leaf, reward)

        return self.best_move(root)

    def init_root_node(self, root_game):
        return RaveNode(game_state=root_game)

    def best_move(self, node: RaveNode) -> int:
        n, move = node.best_child(C_p=0, k=self.k, alpha=self.alpha)
        return move


if __name__ == "__main__":
    from bachelorarbeit.games import Observation, Configuration, ConnectFour
    from bachelorarbeit.tools import timer

    steps = 3000
    pl = RavePlayer(max_steps=steps, exploration_constant=0.5, k=100)
    conf = Configuration()
    game = ConnectFour(columns=conf.columns, rows=conf.rows, inarow=conf.inarow, mark=1)
    obs = Observation(board=game.board.copy(), mark=game.mark)

    with timer(f"{steps} steps"):
        m = pl.get_move(obs, conf)
        print(m)
