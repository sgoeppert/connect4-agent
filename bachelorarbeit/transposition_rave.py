from typing import List, Dict, Tuple
import math
import numpy as np
import random
from collections import defaultdict

from bachelorarbeit.games import Observation, Configuration, ConnectFour
from bachelorarbeit.mcts import Node, MCTSPlayer


def normalize(val):
    return (val + 1) / 2


class TpRaveNode(Node):
    def __init__(self, *args, **kwargs):
        super(TpRaveNode, self).__init__(*args, **kwargs)

        self.parents = []
        self.rave_count = 0
        self.rave_score = 0
        self.rave_children: Dict[int, "TpRaveNode"] = {}

        self.child_values = defaultdict(float)
        self.child_visits = defaultdict(int)
        self.UCT3_val = 0
        self.sim_reward = 0

    def beta(self, b: float = 0.0) -> float:
        return self.rave_count / (self.rave_count + self.number_visits + self.rave_count * self.number_visits * b * b)

    def QRave(self) -> float:
        return self.rave_score

    def exploration(self, parent_visits):
        return math.sqrt(parent_visits / self.number_visits)

    def Qsa(self, move):
        return self.child_values[move]

    def Nsa(self, move):
        return self.child_visits[move]

    def best_child(self,
                   C_p: float = 1.0,
                   b: float = 0.0,
                   uct_method: str="default",
                   alpha: float = None) -> Tuple["TpRaveNode", int]:

        n_p = math.log(self.number_visits)

        def combineWithRave(value, child):
            if alpha is not None:
                beta = alpha
            else:
                beta = child.beta(b)
            return (1 - beta) * value + beta * child.QRave()

        parent = self
        def UCT1(action, child):
            val = normalize(parent.Qsa(action))
            return combineWithRave(val, child) + C_p * math.sqrt(n_p / parent.Nsa(action))

        def UCT2(action, child):
            val = normalize(child.Q())
            return combineWithRave(val, child) + C_p * math.sqrt(n_p / parent.Nsa(action))

        def UCT3(action, child):
            val = normalize(child.UCT3_val)
            return combineWithRave(val, child) + C_p * math.sqrt(n_p / parent.Nsa(action))

        def default(_, child):
            val = normalize(child.Q())
            return combineWithRave(val, child) + C_p * math.sqrt(n_p / child.number_visits)

        selection_method = default
        if uct_method == "UCT1":
            selection_method = UCT1
        elif uct_method == "UCT2":
            selection_method = UCT2
        elif uct_method == "UCT3":
            selection_method = UCT3

        m, c = max(self.children.items(), key=lambda ch: selection_method(*ch))
        return c, m

    def increment_rave_visit_and_add_reward(self, reward):
        self.rave_count += 1
        self.rave_score += (reward - self.rave_score) / self.rave_count

    def find_child_action(self, child):
        for m, _c in self.children.items():
            if _c == child:
                return m
        return None

    def increment_child_visit_and_add_reward(self, move, reward):
        self.child_visits[move] += 1
        self.child_values[move] += (reward - self.child_values[move]) / self.child_visits[move]


    def get_random_action_and_state(self) -> Tuple[int, ConnectFour]:
        m = random.choice(self.possible_moves)
        self.possible_moves.remove(m)
        if len(self.possible_moves) == 0:
            self.expanded = True

        s = self.game_state.copy().play_move(m)
        return m, s

    def add_child_with_rave(self, node: "TpRaveNode", move: int, move_name: int):
        self.children[move] = node
        self.rave_children[move_name] = node
        node.parents.append(self)

    def update_QUCT3(self):
        if not self.children:
            self.UCT3_val = self.sim_reward
        else:
            summed = 0
            for a, c in self.children.items():
                c_n = self.child_visits[a]
                c_v = -c.UCT3_val
                summed += c_n * c_v
            self.UCT3_val = (self.sim_reward + summed) / self.number_visits

    def expand(self, transpositions, moves) -> Tuple["TpRaveNode", bool]:
        nodeclass = type(self)
        move = random.choice(self.possible_moves)  # wähle zufälligen Zug
        child_state = self.game_state.copy().play_move(move)
        state_hash = hash(child_state)

        new_node = False
        if state_hash not in transpositions:
            new_node = True
            transpositions[state_hash] = nodeclass(game_state=child_state)

        move_name = child_state.get_move_name(move, played=True)
        moves.append(move_name)
        self.add_child_with_rave(transpositions[state_hash], move, move_name)

        self.possible_moves.remove(move)
        if len(self.possible_moves) == 0:
            self.expanded = True

        return self.children[move], new_node

    def remove_children(self, player: "TpRavePlayer", keep_node: "TpRaveNode"):
        children = list(self.children.items())
        for m, child in children:
            if child != keep_node:
                try:
                    del(self.children[m])
                except KeyError:
                    pass
                try:
                    del(player.transpositions[hash(child.game_state)])
                except KeyError:
                    pass

    def remove_parent(self, player: "TpRavePlayer"):
        self.parent = None
        for parent in self.parents:
            parent.remove_parent(player)
            parent.remove_children(player, keep_node=self)
            hsh = hash(parent.game_state)
            try:
                del(player.transpositions[hsh])
            except KeyError:
                pass

        self.parents = []

    def __repr__(self):
        return f"TpRaveNode(n: {self.number_visits}, v: {normalize(self.average_value)}, rave_n: {self.rave_count}, rave_v: {self.rave_score})"


class TpRavePlayer(MCTSPlayer):
    name = "TpRavePlayer"

    def __init__(self, b: float = 0.0, uct_method: str = "default", alpha: float = None, *args, **kwargs):
        super(TpRavePlayer, self).__init__(*args, **kwargs)
        self.b = b
        self.uct_method = uct_method
        self.transpositions = {}
        self.alpha = alpha

    def reset(self, *args, **kwargs):
        super(TpRavePlayer, self).reset(*args, **kwargs)
        if not self.keep_tree:
            self.transpositions = {}

    def tree_policy_rave(self, root: "TpRaveNode", moves) -> List["TpRaveNode"]:
        current = root
        path = [root]  # Der Pfad, der während der Selektion durchlaufen wird
        while not current.game_state.is_terminal():
            if current.is_expanded():
                current, m = current.best_child(self.exploration_constant, uct_method=self.uct_method, b=self.b, alpha=self.alpha)
                move_name = current.game_state.get_move_name(m, played=True)
                moves.append(move_name)
                path.append(current)
            else:
                # Wähle einen zufälligen nächsten Zustand
                current, new_node = current.expand(self.transpositions, moves)
                path.append(current)
                if new_node:
                    return path
        return path

    def backup_rave(self, path: List["TpRaveNode"], reward: float, moves: List[int]):
        prev = None
        move_set = set(moves)

        leaf = path[-1]
        leaf.sim_reward = reward
        nodes_to_update = set()

        for _node in reversed(path):
            _node.increment_visit_and_add_reward(reward)

            if prev is not None:
                m = _node.find_child_action(prev)
                _node.increment_child_visit_and_add_reward(m, -reward)

            # Wenn eines der Kinder dieses Knotens über einen in der Simulation gemachten Spielzug
            # erreicht werden könnte, aktualisiere seine Rave Statistik
            for mov, child in _node.rave_children.items():
                if mov in move_set:
                    child.increment_rave_visit_and_add_reward(-reward)

            if self.uct_method == "UCT3":
                nodes_to_update.add(_node)

                new_updates = set()
                for node in nodes_to_update:
                    node.update_QUCT3()
                    if node.parents:
                        new_updates.update(node.parents)
                nodes_to_update = new_updates

            prev = _node
            reward = -reward

    def evaluate_game_state_rave(self, game_state: ConnectFour, moves: List[int]) -> float:
        _g = game_state.copy()
        scoring = _g.get_other_player(_g.get_current_player())
        while not _g.is_terminal():
            m = random.choice(_g.list_moves())
            move_name = _g.get_move_name(m)
            moves.append(move_name)  # merke welche Züge gespielt wurden
            _g.play_move(m)

        return _g.get_reward(scoring)

    def perform_search(self, root):
        while self.has_resources():
            # print(root.rave_children)
            # print(root.children)
            moves = []
            path = self.tree_policy_rave(root, moves)
            reward = self.evaluate_game_state_rave(path[-1].game_state, moves)
            # if self.uct_method == "UCT3":
            #     self.backup_uct3(path, reward)
            # else:
            self.backup_rave(path, reward, moves)

        return self.best_move(root)

    def init_root_node(self, root_game):
        return TpRaveNode(game_state=root_game)

    def best_move(self, node: "TpRaveNode") -> int:
        n, move = node.best_child(C_p=0, b=self.b, alpha=self.alpha)
        # move, n = max(node.children.items(), key=lambda c: (1 - c[1].beta(0)) * c[1].Q() + c[1].beta(0) * c[1].QRave())
        return move


if __name__ == "__main__":
    from bachelorarbeit.games import Observation, Configuration, ConnectFour
    from bachelorarbeit.tools import timer

    steps = 100
    pl = TpRavePlayer(max_steps=steps, exploration_constant=0.9, b=0.0, uct_method="default")
    conf = Configuration()
    game = ConnectFour(columns=conf.columns, rows=conf.rows, inarow=conf.inarow, mark=1)
    obs = Observation(board=game.board.copy(), mark=game.mark)

    with timer(f"{steps} steps"):
        m = pl.get_move(obs, conf)
        print(m)
