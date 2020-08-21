from typing import List, Tuple
import random
import math
from collections import defaultdict

from bachelorarbeit.games import Observation, Configuration, ConnectFour
from bachelorarbeit.mcts import Node, MCTSPlayer


class TranspositionNode(Node):
    def __init__(self, *args, **kwargs):
        super(TranspositionNode, self).__init__( *args, **kwargs)
        self.child_values = defaultdict(float)
        self.child_visits = defaultdict(int)

    def find_child_action(self, child):
        for m, _c in self.children.items():
            if _c == child:
                return m
        return None

    def increment_child_visit_and_add_reward(self, move, reward):
        self.child_visits[move] += 1
        self.child_values[move] += (reward - self.child_values[move]) / self.child_visits[move]

    # def Qsa(self, parent, mymove):
    #     return parent.child_values[mymove]
    #
    # def Nsa(self, parent, mymove):
    #     v = parent.child_visits[mymove]
    #     return v

    def Qsa(self, move):
        return self.child_values[move]

    def Nsa(self, move):
        return self.child_visits[move]

    def best_child(self, exploration_constant: float = 1.0, uct_method: str = "UCT") -> "Node":
        n_p = math.log(self.number_visits)

        parent = self

        def UCT1(action, _):
            return parent.Qsa(action) + exploration_constant * math.sqrt(n_p / parent.Nsa(action))

        def UCT2(action, child):
            return child.Q() + exploration_constant * math.sqrt(n_p / parent.Nsa(action))

        def default(_, child):
            return child.Q() + exploration_constant * math.sqrt(n_p / child.number_visits)

        selection_method = default
        if uct_method == "UCT1":
            selection_method = UCT1
        elif uct_method == "UCT2":
            selection_method = UCT2

        _, c = max(self.children.items(), key=lambda ch: selection_method(*ch))
        return c

    def get_random_action_and_state(self) -> Tuple[int, ConnectFour]:
        m = random.choice(self.possible_moves)
        self.possible_moves.remove(m)
        if len(self.possible_moves) == 0:
            self.expanded = True

        s = self.game_state.copy().play_move(m)
        return m, s

    def __repr__(self):
        return f"TransposNode(n: {self.number_visits}, v: {self.total_value}, \n\tchild_n: {self.child_visits}, \n\tchild_v: {self.child_values})"



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
        self.transpositions = {}

    def tree_policy(self, root: TranspositionNode) -> List[TranspositionNode]:
        current = root
        path = [root]  # Der Pfad, der während der Selektion durchlaufen wird
        while not current.game_state.is_terminal():
            if current.is_expanded():
                current = current.best_child(self.exploration_constant, uct_method=self.uct_method)
                path.append(current)
            else:
                # Wähle einen zufälligen nächsten Zustand
                move, next_state = current.get_random_action_and_state()
                # Und prüfe ob er bereits in der Transpositionstabelle enthalten ist
                hash_state = hash(next_state)
                if hash_state not in self.transpositions:
                    # Wenn nicht wird ein neuer Knoten hinzugefügt
                    self.transpositions[hash_state] = TranspositionNode(game_state=next_state, parent=current)
                next_node = self.transpositions[hash_state]
                current.add_child(next_node, move)
                path.append(next_node)
                return path
        return path

    def backup(self, path: List[TranspositionNode], reward: float):
        prev = None
        for _node in reversed(path):
            _node.increment_visit_and_add_reward(reward)

            if prev is not None:
                m = _node.find_child_action(prev)
                _node.increment_child_visit_and_add_reward(m, -reward)

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

    steps = 2000
    conf = Configuration()
    p = TranspositionPlayer(max_steps=steps, uct_method="UCT2", keep_tree=True)
    game = ConnectFour(columns=conf.columns, rows=conf.rows, inarow=conf.inarow, mark=1)
    obs = Observation(board=game.board.copy(), mark=game.mark)

    with timer(f"{steps} steps"):
        m = p.get_move(obs, conf)

    print(m)