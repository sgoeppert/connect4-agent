import random
import math
from abc import ABC, abstractmethod
from collections import defaultdict
import time
from bachelorarbeit.games import Observation, Configuration, ConnectFour
from typing import List, Optional


class Player(ABC):
    name = "Player"

    def __init__(self, **kwargs):
        if kwargs:
            print("WARNING: Not all parameters used when initializing player:\n", kwargs, self)

    @abstractmethod
    def get_move(self, observation: Observation, configuration: Configuration) -> int:
        pass

    def __repr__(self) -> str:
        return self.name


class RandomPlayer(Player):
    name = "RandomPlayer"

    def __init__(self, **kwargs):
        super(RandomPlayer, self).__init__(**kwargs)

    def get_move(self, observation: Observation, configuration: Configuration) -> int:
        return random.choice([c for c in range(configuration.columns) if observation.board[c] == 0])


class TreePlayer(Player):
    def __init__(self, keep_tree: bool = False, max_steps: int = 1000, time_buffer_pct: float = 0.05, **kwargs):
        super(TreePlayer, self).__init__(**kwargs)
        self.keep_tree = keep_tree
        self.root = None
        # Verbrauchte und maximale Schritte pro Zug
        self.max_steps = max_steps
        self.steps_taken = 0
        self.initialized = False
        # Limitiert die Ausführungszeit
        self.start_time = 0
        self.time_limit = 0
        self.time_buffer_pct = time_buffer_pct

    def reset(self, conf):
        self.start_time = time.time()
        self.steps_taken = 0

        if not self.keep_tree:
            self.root = None

        if conf is not None:
            if self.initialized:
                limit = conf.actTimeout
            else:
                limit = conf.agentTimeout

            if limit > 0:
                self.time_limit = limit * (1-self.time_buffer_pct)
        self.initialized = True

    def has_resources(self) -> bool:
        if self.time_limit > 0:
            return time.time() - self.start_time < self.time_limit
        else:
            self.steps_taken += 1
            return self.steps_taken <= self.max_steps

    @staticmethod
    def determine_opponent_move(new_board: List[int], old_board: List[int], columns: int = 7) -> int:
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
            opp_move = TreePlayer.determine_opponent_move(observation.board,
                                                          self.root.game_state.board,
                                                          configuration.columns)
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

    @abstractmethod
    def init_root_node(self, game_state: ConnectFour):
        pass

    @abstractmethod
    def perform_search(self, root):
        pass

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
        # persist the root if we're keeping the tree
        self._store_root(root.children[best])

        return best

def simulate_game(game):
    while not game.is_terminal():
        move = random.choice(game.list_moves())
        game.play_move(move)
    return game

class FlatMonteCarlo(Player):
    name = "FlatMonteCarloPlayer"

    def __init__(self,
                 max_steps: int = 1000,
                 exploration_constant: float = 0.8,
                 ucb_selection: bool = True,
                 **kwargs):

        super(FlatMonteCarlo, self).__init__(**kwargs)
        self.max_steps = max_steps
        self.ucb_selection = ucb_selection
        self.c_p = exploration_constant

    def get_move(self,
                 observation: Observation,
                 configuration: Configuration) -> int:

        game = ConnectFour(board=observation.board,
                           rows=configuration.rows,
                           columns=configuration.columns,
                           inarow=configuration.inarow,
                           mark=observation.mark)

        n = defaultdict(int)
        q = defaultdict(float)
        scoring_player = game.get_current_player()

        def ucb(m, N=1):
            return q[m] + self.c_p * math.sqrt(math.log(N) / max(1,n[m]))

        for i in range(1, self.max_steps+1):
            if self.ucb_selection:
                a = max(game.list_moves(), key=lambda m: ucb(m, i))
            else:
                a = random.choice(game.list_moves())

            end_state = simulate_game(game.copy().play_move(a))
            r = end_state.get_reward(scoring_player)
            n[a] += 1
            q[a] += (r - q[a])/n[a]

        return max(n.keys(), key=lambda m: ucb(m))


if __name__ == "__main__":
    from bachelorarbeit.selfplay import Arena
    from bachelorarbeit.tools import timer

    steps = 20
    arena = Arena(
        players=(FlatMonteCarlo, RandomPlayer),
        constructor_args=({"max_steps": steps, "exploration_constant": 1 / math.sqrt(2)},
                          None),
        num_games=100,
        num_processes=4
    )
    results = arena.run_game_mp()