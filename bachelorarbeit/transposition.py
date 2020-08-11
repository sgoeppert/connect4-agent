from typing import List
import random

from bachelorarbeit.games import Observation, Configuration, ConnectFour
from bachelorarbeit.mcts import Node, MCTSPlayer


class TranspositionPlayer(MCTSPlayer):
    name = "Transpositionplayer"

    def __init__(self, **kwargs):
        super(TranspositionPlayer, self).__init__(**kwargs)
        self.transpositions = {}

    def __repr__(self) -> str:
        return self.name

    def reset(self):
        super(TranspositionPlayer, self).reset()
        self.transpositions = {}

    def find_leaf(self, root: Node) -> List[Node]:
        current = root
        path = [root]
        while current.is_expanded():
            current = current.best_child(self.exploration_constant)
            path.append(current)
        return path

    def expand(self, path: List[Node]) -> Node:
        node = path[-1]
        # Only expand the node if it isn't a terminal state. Terminal states don't have children
        if not node.game_state.is_terminal():
            move = random.choice(node.possible_moves)
            node.possible_moves.remove(move)

            child_state = node.game_state.copy()
            child_state.play_move(move)

            hash_state = hash(child_state)
            if hash_state not in self.transpositions:
                self.transpositions[hash_state] = Node(child_state, parent=node, move=move)
            child_node = self.transpositions[hash_state]

            node.children[move] = child_node
            if len(node.possible_moves) == 0:
                node.expanded = True

            path.append(child_node)
            return child_node
        else:
            return node

    def backup(self, path: List[Node], reward: float):
        for _node in reversed(path):
            _node.number_visits += 1
            _node.total_value += reward

            reward = -reward

    def get_move(self, observation: Observation, conf: Configuration) -> int:
        self.reset()
        root_game = ConnectFour(
            columns=conf.columns,
            rows=conf.rows,
            inarow=conf.inarow,
            mark=observation.mark,
            board=observation.board
        )
        root = Node(root_game)

        while self.has_resources():
            path = self.find_leaf(root)
            child = self.expand(path)
            reward = self.evaluate_game_state(child.game_state)
            self.backup(path, reward)

        best = self.best_move(root)
        return best
