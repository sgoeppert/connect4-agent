import random
import math
import numpy as np
from typing import List

from bachelorarbeit.games import Observation, Configuration, ConnectFour
from bachelorarbeit.base_players import Player


class Node:
    def __init__(self, game_state: ConnectFour, parent: "Node" = None, move: int = None):
        self.total_value = 0
        self.number_visits = 0
        self.children = {}

        self.parent = parent
        self.move = move

        self.game_state = game_state
        self.possible_moves = game_state.list_moves()
        self.expanded = False

    def Q(self) -> float:
        return self.total_value / self.number_visits

    def best_child(self, exploration_constant: float = 1.0) -> "Node":
        n_p = math.log(self.number_visits)
        _, c = max(self.children.items(),
                   key=lambda c: c[1].Q() + exploration_constant * math.sqrt(n_p / c[1].number_visits))
        return c

    def is_expanded(self) -> bool:
        return self.expanded

    def expand_one_child(self) -> "Node":
        move = random.choice(self.possible_moves)
        child_state = self.game_state.copy()
        child_state.play_move(move)
        myclass = type(self)
        self.children[move] = myclass(game_state=child_state, parent=self, move=move)
        self.possible_moves.remove(move)
        if len(self.possible_moves) == 0:
            self.expanded = True

        return self.children[move]

    def add_child(self, node: "Node", move: int):
        self.children[move] = node


class MCTSPlayer(Player):
    name = "MCTSPlayer"

    def __init__(
            self,
            exploration_constant: float = 0.8,
            max_steps: int = 1000,
            keep_tree: bool = False,
            **kwargs
    ):
        super(MCTSPlayer, self).__init__(**kwargs)
        self.max_steps = max_steps
        self.steps_taken = 0

        self.num_nodes = 0
        self.exploration_constant = exploration_constant
        self.keep_tree = keep_tree
        self.root = None

    def __repr__(self) -> str:
        return self.name

    def reset(self):
        self.steps_taken = 0
        if not self.keep_tree:
            self.root = None

    def has_resources(self) -> bool:
        self.steps_taken += 1
        return self.steps_taken <= self.max_steps

    def evaluate_game_state(self, game_state: ConnectFour) -> float:
        game = game_state.copy()
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

    def find_leaf(self, root: Node) -> Node:
        current = root
        while current.is_expanded():
            current = current.best_child(self.exploration_constant)
        return current

    def expand(self, node: Node) -> Node:
        # Only expand the node if it isn't a terminal state. Terminal states don't have children
        if not node.game_state.is_terminal():
            return node.expand_one_child()
        else:
            return node

    def backup(self, node: Node, reward: float):
        current = node
        while current is not None:
            current.number_visits += 1
            current.total_value += reward
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
                root.parent = None
            else:
                root = None
        return root

    def _store_root(self, new_root):
        if self.keep_tree:
            self.root = new_root
            self.root.parent = None


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
            root = Node(root_game)

        while self.has_resources():
            leaf = self.tree_policy(root)
            reward = self.evaluate_game_state(leaf.game_state)
            self.backup(leaf, reward)

        best = self.best_move(root)
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
        p.get_move(obs, conf)