import random
import math
import numpy as np
from typing import List
import time

from bachelorarbeit.games import Observation, Configuration, ConnectFour
from bachelorarbeit.base_players import Player


class Node:
    def __init__(self, game_state: ConnectFour, parent: "Node" = None):
        self.average_value = 0  # Q(v)
        self.total_value = 0  # V(v)
        self.number_visits = 0  # N(v)
        self.children = {}  # C(v), Kinder des Knotens

        self.parent = parent  # der direkte Elternknoten

        self.game_state = game_state  # der Spielzustand in diesem Knoten
        self.possible_moves = game_state.list_moves()  # Aktionen der noch nicht erforschten Kindknoten
        self.expanded = False  # ob der Knoten vollständig expandiert ist

    def Q(self) -> float:
        return self.average_value

    def best_child(self, C_p: float = 1.0) -> "Node":
        n_p = math.log(self.number_visits)

        def UCT(child: Node):
            """
            Berechnet den UCT Wert UCT = Q(v') + C_p * sqrt(ln(N(v))/N(v'))
            :param child: Knoten v'
            :return:
            """
            return child.Q() + C_p * math.sqrt(n_p / child.number_visits)

        _, c = max(self.children.items(), key=lambda entry: UCT(entry[1]))
        return c

    def increment_visit_and_add_reward(self, reward: float):
        self.number_visits += 1
        self.average_value += (reward - self.average_value) / self.number_visits

    def is_expanded(self) -> bool:
        return self.expanded

    def expand_one_child(self) -> "Node":
        node_class = type(self)

        move = random.choice(self.possible_moves)
        self.children[move] = node_class(game_state=self.game_state.copy().play_move(move), parent=self)
        self.possible_moves.remove(move)
        if len(self.possible_moves) == 0:
            self.expanded = True

        return self.children[move]

    def remove_parent(self, player):
        self.parent = None

    def __repr__(self):
        return f"Node(Q:{self.Q()}, N:{self.number_visits})"


class MCTSPlayer(Player):
    name = "MCTSPlayer"

    def __init__(
            self,
            exploration_constant: float = 0.8,
            max_steps: int = 1000,
            keep_tree: bool = False,
            time_buffer: float = 0.3,
            **kwargs
    ):
        super(MCTSPlayer, self).__init__(**kwargs)

        # UCT Exploration Konstante Cp
        self.exploration_constant = exploration_constant
        self.keep_tree = keep_tree  # ob der Baum zwischen Zügen erhalten bleibt
        self.root = None  # die Wurzel des Baumes

        # Limitiert die Ausführungszeit
        self.start_time = 0
        self.time_limit = 0
        self.time_buffer = time_buffer

        # Verbrauchte und maximale Schritte pro Zug
        self.max_steps = max_steps
        self.steps_taken = 0

    def reset(self, conf: Configuration = None):
        self.steps_taken = 0
        self.start_time = time.time()

        if not self.keep_tree:
            self.root = None

        if conf is not None:
            if conf.timeout > 0:
                self.time_limit = conf.timeout - self.time_buffer

    def has_resources(self) -> bool:
        if self.time_limit > 0:
            return time.time() - self.start_time < self.time_limit
        else:
            self.steps_taken += 1
            return self.steps_taken <= self.max_steps

    def evaluate_game_state(self, game_state: ConnectFour) -> float:
        game = game_state.copy()
        # Die Bewertung Geschieht aus Sicht des Spielers, der uns in diesen Zustand geführt hat, darum muss der Spieler
        # geholt werden, der zuletzt gezogen hat, nicht der der gerade an der Reihe ist.
        scoring = game.get_other_player(game.get_current_player())
        while not game.is_terminal():
            game.play_move(random.choice(game.list_moves()))

        return game.get_reward(scoring)

    def tree_policy(self, root: Node) -> Node:
        current = root
        while not current.game_state.is_terminal():
            if current.is_expanded():
                current = current.best_child(self.exploration_constant)
            else:
                return current.expand_one_child()
        return current

    def backup(self, node: Node, reward: float):
        current = node
        while current is not None:
            current.increment_visit_and_add_reward(reward)
            reward = -reward
            current = current.parent

    def best_move(self, node: Node) -> int:
        move, n = max(node.children.items(), key=lambda c: c[1].Q())
        return move

    def determine_opponent_move(self, new_board: List[int], old_board: List[int], columns: int = 7) -> int:
        i = 0
        for new_s, old_s in zip(new_board, old_board):
            if new_s != old_s:
                return i % columns
            i += 1
        return -1

    def _restore_root(self, observation, configuration):
        root = None
        # if we're keeping the tree and have a stored node, try to determine the opponents move and apply that move
        # the resulting node is out starting root-node
        if self.keep_tree and self.root is not None:
            root = self.root
            opp_move = self.determine_opponent_move(observation.board, self.root.game_state.board, configuration.columns)
            if opp_move in root.children:
                root = root.children[opp_move]
                root.remove_parent(self)
            else:
                root = None
        return root

    def _store_root(self, new_root):
        if self.keep_tree:
            self.root = new_root
            self.root.remove_parent(self)

    def init_root_node(self, root_game):
        return Node(root_game)

    def perform_search(self, root):
        """
        Führe die Monte-Carlo-Baumsuche ausgehend von einem Wurzelknoten durch.

        So lange noch Ressourcen verfügbar sind, dies können eine begrenzte Anzahl an Iterationen oder ein Zeitlimit
        sein, wird der MCTS Algorithmus ausgeführt. Die tree_policy durchläuft den bereits existierenden Baum bis ein
        Blatt gefunden wurde, welches wenn möglich in der tree_policy expandiert wird. Danach wird der Spielzustand
        in diesem Blatt mit evaluate_game_state zu Ende simuliert. Das Ergebnis dieser Simulation - -1, 0  oder 1 - wird
        mit in backup benutzt, um die Statistiken der Knoten zu aktualisieren.
        Nach Ablauf der Ressourcen wird der beste Zug durch best_move ausgewählt. Dies ist der Zug mit der höchsten
        durchschnittlichen Belohnung.

        :param root: Der Wurzelknoten v0
        :return:
        """
        while self.has_resources():
            leaf = self.tree_policy(root)
            reward = self.evaluate_game_state(leaf.game_state)
            self.backup(leaf, reward)
        return self.best_move(root)

    def get_move(self, observation: Observation, conf: Configuration) -> int:
        self.reset(conf)

        # load the root if it was persisted
        root = self._restore_root(observation, conf)

        # if no root could be determined, create a new tree from scratch
        if root is None:
            root_game = ConnectFour(columns=conf.columns, rows=conf.rows, inarow=conf.inarow,
                                    mark=observation.mark, board=observation.board)
            root = self.init_root_node(root_game)

        # run the search
        best = self.perform_search(root)
        # print(root)
        # print("Children:", *list(root.children.items()), sep="\n\t")
        # persist the root if we're keeping the tree
        self._store_root(root.children[best])

        return best


if __name__ == "__main__":
    from bachelorarbeit.games import Observation, Configuration, ConnectFour
    from bachelorarbeit.tools import timer

    steps = 2000
    p = MCTSPlayer(max_steps=steps)
    conf = Configuration()
    game = ConnectFour(columns=conf.columns, rows=conf.rows, inarow=conf.inarow, mark=1)
    obs = Observation(board=game.board.copy(), mark=game.mark)

    with timer(f"{steps} steps"):
        m = p.get_move(obs, conf)
    print(m)