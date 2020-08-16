from typing import List
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

    def Qsa(self, parent, mymove):
        return parent.child_values[mymove] / parent.child_visits[mymove]

    def Nsa(self, parent, mymove):
        v = parent.child_visits[mymove]
        return v

    def best_child(self, exploration_constant: float = 1.0, uct_method: str = "UCT") -> "Node":
        n_p = math.log(self.number_visits)

        if uct_method == "UCT1":
            _, c = max(self.children.items(),
                       key=lambda c: c[1].Qsa(self, c[0]) + exploration_constant * math.sqrt(n_p / c[1].Nsa(self, c[0])))
        elif uct_method == "UCT2":
            _, c = max(self.children.items(),
                       key=lambda c: c[1].Q() + exploration_constant * math.sqrt(n_p / c[1].Nsa(self, c[0])))
        else:
            _, c = max(self.children.items(),
                       key=lambda c: c[1].Q() + exploration_constant * math.sqrt(n_p / c[1].number_visits))

        return c

    def get_random_action_and_state(self):
        m = random.choice(self.possible_moves)
        self.possible_moves.remove(m)
        if len(self.possible_moves) == 0:
            self.expanded = True

        s = self.game_state.copy()
        s.play_move(m)
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

    def reset(self):
        super(TranspositionPlayer, self).reset()
        self.transpositions = {}

    def tree_policy(self, root: TranspositionNode) -> List[TranspositionNode]:
        current = root
        path = [root]
        while not current.game_state.is_terminal():
            if current.is_expanded():
                current = current.best_child(self.exploration_constant, uct_method=self.uct_method)
                path.append(current)
            else:
                move, next_state = current.get_random_action_and_state()
                hash_state = hash(next_state)
                if hash_state not in self.transpositions:
                    self.transpositions[hash_state] = TranspositionNode(game_state=next_state, parent=current, move=move)
                next_node = self.transpositions[hash_state]
                current.add_child(next_node, move)
                path.append(next_node)
                return path
        return path

    def backup(self, path: List[TranspositionNode], reward: float):
        prev = None
        for _node in reversed(path):
            _node.number_visits += 1
            _node.total_value += reward

            if prev is not None:
                m = _node.find_child_action(prev)
                _node.child_values[m] += -reward
                _node.child_visits[m] += 1

            prev = _node
            reward = -reward

    def get_move(self, observation: Observation, conf: Configuration) -> int:
        self.reset()

        # load the root if it was persisted
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
            root = TranspositionNode(root_game)

        while self.has_resources():
            path = self.tree_policy(root)
            # path = self.find_leaf(root)
            # child = self.expand(path)
            reward = self.evaluate_game_state(path[-1].game_state)
            self.backup(path, reward)

        # print(root)
        # print(root.children)
        best = self.best_move(root)
        self._store_root(root.children[best])
        return best


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
        game.play_move(m)
        game.play_move(3)
        p.get_move(Observation(board=game.board.copy(), mark=game.mark), conf)