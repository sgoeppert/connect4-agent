from typing import List
import random
import math
from collections import defaultdict

from bachelorarbeit.players.mcts import Node, MCTSPlayer


def normalize(val):
    return (val + 1) / 2

class TranspositionNode(Node):
    def __init__(self, *args, **kwargs):
        super(TranspositionNode, self).__init__( *args, **kwargs)
        self.parents = []

        self.child_values = defaultdict(float)
        self.child_visits = defaultdict(int)
        self.UCT3_val = 0
        self.sim_reward = 0

    def find_child_action(self, child):
        for m, _c in self.children.items():
            if _c == child:
                return m
        return None

    def Qsa(self, move):
        return self.child_values[move]

    def Nsa(self, move):
        return self.child_visits[move]


    def best_child(self, C_p: float = 1.0, uct_method: str = "UCT") -> "Node":
        n_p = math.log(self.number_visits)
        parent = self

        def UCT1(action, _):
            return normalize(parent.Qsa(action)) + C_p * math.sqrt(n_p / parent.Nsa(action))

        def UCT2(action, child):
            return normalize(child.Q()) + C_p * math.sqrt(n_p / parent.Nsa(action))

        def UCT3(action, child):
            return normalize(child.UCT3_val) + C_p * math.sqrt(n_p / parent.Nsa(action))

        def default(_, child):
            return normalize(child.Q()) + C_p * math.sqrt(n_p / child.number_visits)

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
        hash_state = hash(next_state)
        if hash_state not in player.transpositions:
            # Wenn nicht wird ein neuer Knoten hinzugefügt
            player.transpositions[hash_state] = TranspositionNode(game_state=next_state)
        next_node = player.transpositions[hash_state]
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
                c_n = self.child_visits[a]
                c_v = -c.UCT3_val
                summed += c_n * c_v
            self.UCT3_val = (self.sim_reward + summed) / self.number_visits

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
        me = f"TransposNode(n: {self.number_visits}, v: {normalize(self.average_value)}, uct3: {normalize(self.UCT3_val)})"
        return me


class TranspositionPlayer(MCTSPlayer):
    name = "Transpositionplayer"

    def __init__(self, uct_method: str = "UCT", **kwargs):
        super(TranspositionPlayer, self).__init__(**kwargs)
        self.transpositions = {}
        self.uct_method = uct_method

    def __repr__(self) -> str:
        return self.name

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
                m = _node.find_child_action(prev)
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
            reward = self.evaluate_game_state(path[-1].game_state)
            self.backup(path, reward)
        return self.best_move(root)

    def init_root_node(self, root_game):
        return TranspositionNode(root_game)


if __name__ == "__main__":
    from bachelorarbeit.games import Observation, Configuration, ConnectFour
    from bachelorarbeit.tools import timer


    # def backup(path, reward):
    #
    #     leaf = path[-1]
    #     leaf.sim_reward = reward
    #
    #     nodes_to_update = set()
    #     prev = None
    #     for _node in reversed(path):
    #         _node.increment_visit_and_add_reward(reward)
    #         nodes_to_update.add(_node)
    #         if prev is not None:
    #             m = _node.find_child_action(prev)
    #             _node.increment_child_visit_and_add_reward(m, -reward)
    #
    #         new_updates = set()
    #         for node in nodes_to_update:
    #             node.update_QUCT3()
    #             if node.parents:
    #                 new_updates.update(node.parents)
    #         nodes_to_update = new_updates
    #
    #         prev = _node
    #         reward = -reward
    #
    #
    # s = ConnectFour()
    # root = TranspositionNode(s)
    #
    # child = TranspositionNode(s.copy().play_move(3))
    # root.add_child(child, 3)
    # path = [root, child]
    # backup(path, 1)
    #
    # c2 = TranspositionNode(s.copy().play_move(1))
    # root.add_child(c2, 1)
    # path = [root, c2]
    # backup(path, 0)
    #
    # c3 = TranspositionNode(s.copy().play_move(6))
    # root.add_child(c3, 6)
    # path = [root, c3]
    # backup(path, -1)
    #
    # c4 = TranspositionNode(s.copy().play_move(3).play_move(4))
    # child.add_child(c4, 4)
    # path = [root, child, c4]
    # backup(path, 1)
    #
    # print(root)
    # print("Children:", *list(root.children.items()), sep="\n\t")


    def memory_usage_psutil():
        # return the memory usage in MB
        import os
        import psutil
        process = psutil.Process(os.getpid())
        mem = process.memory_info().rss / float(2 ** 20)
        return mem

    #
    steps = 10000
    conf = Configuration()
    p = TranspositionPlayer(max_steps=steps, uct_method="UCT3", keep_tree=True, exploration_constant=0.707)
    game = ConnectFour(columns=conf.columns, rows=conf.rows, inarow=conf.inarow, mark=1)
    obs = Observation(board=game.board.copy(), mark=game.mark)

    with timer(f"{steps} steps"):
        m = p.get_move(obs, conf)

    print(memory_usage_psutil(), "MiB")
    # print(len(p.transpositions))
    print(m)
    g = game.play_move(m).play_move(4)
    obs.board = g.board.copy()
    obs.mark = g.mark
    # print(len(p.transpositions))
    m = p.get_move(obs, conf)
    print(memory_usage_psutil(), "MiB")

    g = game.play_move(m).play_move(4)
    obs.board = g.board.copy()
    obs.mark = g.mark
    m = p.get_move(obs, conf)
    print(memory_usage_psutil(), "MiB")

    # print(len(p.transpositions))
    print(m)