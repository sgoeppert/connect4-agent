from typing import List
import random
import math
from collections import defaultdict

from bachelorarbeit.players.mcts import Node, MCTSPlayer
from bachelorarbeit.tools import flip_board


def Q(v, a=None):
    if a is None:
        return v.average_value
    else:
        return v.child_values[a]


def N(v, a=None):
    if a is None:
        return v.number_visits
    else:
        return v.child_visits[a]


class TranspositionNode(Node):
    def __init__(self, *args, **kwargs):
        super(TranspositionNode, self).__init__( *args, **kwargs)
        self.parents = []

        self.child_values = defaultdict(float)
        self.child_visits = defaultdict(int)
        self.UCT3_val = 0
        self.sim_reward = 0

    def find_child_actions(self, child):
        actions = []
        for m, _c in self.children.items():
            if _c == child:
                actions.append(m)
        return actions

    def best_child(self, C_p: float = 1.0, uct_method: str = "UCT") -> "Node":
        n_p = math.log(self.number_visits)
        v = self

        def UCT1(a, _): return Q(v, a) + C_p * math.sqrt(n_p / N(v, a))

        def UCT2(a, child):
            return Q(child) + C_p * math.sqrt(n_p / N(v, a))

        def UCT3(a, child):
            return child.UCT3_val + C_p * math.sqrt(n_p / N(v, a))

        def default(_, child):
            return Q(child) + C_p * math.sqrt(n_p / N(child))

        selection_method = default
        if uct_method == "UCT1":
            selection_method = UCT1
        elif uct_method == "UCT2":
            selection_method = UCT2
        elif uct_method == "UCT3":
            selection_method = UCT3

        _, c = max(self.children.items(), key=lambda ch: selection_method(*ch))
        return c

    def expand(self, player):
        # Wähle einen zufälligen nächsten Zustand
        move = random.choice(self.possible_moves)
        self.possible_moves.remove(move)
        if len(self.possible_moves) == 0:
            self.expanded = True

        next_state = self.game_state.copy().play_move(move)
        # Und prüfe ob er bereits in der Transpositionstabelle enthalten ist
        next_node = player.get_or_store_transposition(next_state)
        self.add_child(next_node, move)

        return next_node

    def add_child(self, node: "TranspositionNode", move: int):
        self.children[move] = node
        node.parents.append(self)

    def update_QUCT3(self):
        if not self.children:
            self.UCT3_val = self.sim_reward
        else:
            summed = 0
            for a, c in self.children.items():
                c_n = N(self, a)
                c_v = -c.UCT3_val
                summed += c_n * c_v
            self.UCT3_val = (self.sim_reward + summed) / N(self)

    def remove_children(self, player: "TranspositionPlayer", keep_node: "TranspositionNode"):
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

    def remove_parent(self, player: "TranspositionPlayer"):
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
        me = f"TransposNode(n: {self.number_visits}, v: {self.average_value}, uct3: {self.UCT3_val})"
        return me


class TranspositionPlayer(MCTSPlayer):
    name = "TranspositionPlayer"

    def __init__(self, uct_method: str = "UCT", with_symmetry: bool = False, **kwargs):
        super(TranspositionPlayer, self).__init__(**kwargs)
        self.transpositions = {}
        self.uct_method = uct_method
        self.with_symmetry = with_symmetry

    def __repr__(self) -> str:
        return self.name

    def get_or_store_transposition(self, state):
        flipped = None

        if state in self.transpositions:
            return self.transpositions[state]

        if self.with_symmetry:
            flipped = tuple(flip_board(state.board))
            if flipped in self.transpositions:
                return self.transpositions[flipped]

        tp_node = TranspositionNode(game_state=state)
        self.transpositions[state] = tp_node
        if self.with_symmetry:
            self.transpositions[flipped] = tp_node

        return tp_node

    def reset(self, *args, **kwargs):
        super(TranspositionPlayer, self).reset(*args, **kwargs)
        if not self.keep_tree:
            self.transpositions = {}

    def tree_policy(self, root: TranspositionNode) -> List[TranspositionNode]:
        current = root
        path = [root]  # Der Pfad, der während der Selektion durchlaufen wird
        while not current.game_state.is_terminal():
            if current.is_expanded():
                current = current.best_child(self.exploration_constant, uct_method=self.uct_method)
                path.append(current)
            else:
                current = current.expand(self)
                path.append(current)
                return path
        return path

    def backup(self, path: List[TranspositionNode], reward: float):
        nodes_to_update = set()

        leaf = path[-1]
        leaf.sim_reward = reward

        prev = None
        for _node in reversed(path):
            _node.number_visits += 1
            _node.average_value += (reward - _node.average_value) / _node.number_visits

            if prev is not None:
                ms = _node.find_child_actions(prev)
                for m in ms:
                    _node.child_visits[m] += 1
                    _node.child_values[m] += (-reward - _node.child_values[m]) / _node.child_visits[m]

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

    def perform_search(self, root):
        while self.has_resources():
            path = self.tree_policy(root)
            reward = self.evaluate(path[-1].game_state)
            self.backup(path, reward)
        return self.best_move(root)

    def init_root_node(self, root_game):
        return TranspositionNode(root_game)


if __name__ == "__main__":
    from bachelorarbeit.games import Observation, Configuration, ConnectFour
    from bachelorarbeit.tools import timer

    steps = 3000
    conf = Configuration()
    p = TranspositionPlayer(max_steps=steps, uct_method="default", exploration_constant=1.0, with_symmetry=True)
    game = ConnectFour(columns=conf.columns, rows=conf.rows, inarow=conf.inarow, mark=1)
    obs = Observation(board=game.board.copy(), mark=game.mark)

    with timer(f"{steps} steps"):
        m = p.get_move(obs, conf)
        print(m)
