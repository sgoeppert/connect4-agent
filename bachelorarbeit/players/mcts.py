import random
import math
from typing import Type, TypeVar

from bachelorarbeit.games import Observation, Configuration, ConnectFour
from bachelorarbeit.players.base_players import TreePlayer


class Evaluator:
    def __call__(self, game_state: ConnectFour) -> float:
        game = game_state.copy()
        scoring = game.get_other_player(game.get_current_player())
        while not game.is_terminal():
            game.play_move(random.choice(game.list_moves()))

        return game.get_reward(scoring)

    def reset(self):
        pass


class Node:
    def __init__(self, game_state: ConnectFour, parent=None):
        self.average_value = 0  # Q(v)
        self.number_visits = 0  # N(v)
        self.children = {}  # C(v), Kinder des Knotens

        self.parent = parent

        self.game_state = game_state  # der Spielzustand in diesem Knoten
        self.possible_moves = game_state.list_moves()  # Aktionen der noch nicht erforschten Kindknoten
        self.expanded = False  # ob der Knoten vollständig expandiert ist

    def Q(self) -> float:
        return self.average_value

    def best_child(self, C_p: float = 1.0):
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

    def expand_one_child(self):
        node_class = type(self)

        move = random.choice(self.possible_moves)
        next_state = self.game_state.copy().play_move(move)
        self.children[move] = node_class(
            game_state=next_state,
            parent=self
        )
        self.possible_moves.remove(move)
        if len(self.possible_moves) == 0:
            self.expanded = True

        return self.children[move]

    def remove_parent(self, player):
        self.parent = None

    def __repr__(self):
        return f"Node(Q:{self.Q()}, N:{self.number_visits})"


class MCTSPlayer(TreePlayer):
    name = "MCTSPlayer"

    def __init__(
            self,
            exploration_constant: float = 1.0,
            max_steps: int = 1000,
            keep_tree: bool = False,
            time_buffer_pct: float = 0.05,
            **kwargs
    ):
        super(MCTSPlayer, self).__init__(max_steps=max_steps, keep_tree=keep_tree, time_buffer_pct=time_buffer_pct)

        # UCT Exploration Konstante Cp
        self.exploration_constant = exploration_constant
        self.evaluate = Evaluator()

    def reset(self, conf: Configuration = None):
        super(MCTSPlayer, self).reset(conf)
        self.evaluate.reset()

    def tree_policy(self, root):
        current = root
        while not current.game_state.is_terminal():
            if current.is_expanded():
                current = current.best_child(self.exploration_constant)
            else:
                return current.expand_one_child()
        return current

    def backup(self, node, reward: float):
        current = node
        while current is not None:
            current.increment_visit_and_add_reward(reward)
            reward = -reward
            current = current.parent

    def best_move(self, node) -> int:
        move, n = max(node.children.items(), key=lambda c: c[1].Q())
        return move

    def init_root_node(self, root_game) -> Node:
        return Node(root_game)

    def perform_search(self, root) -> int:
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
            leaf = self.tree_policy(root)  # type: Node
            reward = self.evaluate(leaf.game_state)
            self.backup(leaf, reward)
        return self.best_move(root)


if __name__ == "__main__":
    from bachelorarbeit.games import Observation, Configuration, ConnectFour
    from bachelorarbeit.tools import timer

    steps = 50000
    p = MCTSPlayer(max_steps=steps)
    conf = Configuration()
    game = ConnectFour(columns=conf.columns, rows=conf.rows, inarow=conf.inarow, mark=1)
    obs = Observation(board=game.board.copy(), mark=game.mark)

    with timer(steps):
        p.get_move(obs, conf)
