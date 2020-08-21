import random
import math
from abc import ABC, abstractmethod
from collections import defaultdict

from bachelorarbeit.games import Observation, Configuration, ConnectFour


class Player(ABC):
    name = "Player"

    def __init__(self, **kwargs):
        if kwargs:
            print("WARNING: Not all parameters used when initializing player:\n", kwargs, self)

    @abstractmethod
    def get_move(self, observation: Observation, configuration: Configuration) -> int:
        pass


class RandomPlayer(Player):
    name = "RandomPlayer"

    def __init__(self, **kwargs):
        super(RandomPlayer, self).__init__(**kwargs)

    def get_move(self, observation: Observation, configuration: Configuration) -> int:
        return random.choice([c for c in range(configuration.columns) if observation.board[c] == 0])

    def __repr__(self) -> str:
        return self.name


class FlatMonteCarlo(Player):
    name = "FlatMonteCarloPlayer"

    def __init__(self, max_steps: int = 1000, exploration_constant: float = 1.0, ucb_selection: bool = True, **kwargs):
        super(FlatMonteCarlo, self).__init__(**kwargs)
        self.max_steps = max_steps
        self.ucb_selection = ucb_selection
        self.exploration_constant = exploration_constant

    def evaluate_game_state(self, game: ConnectFour, scoring_player: int) -> int:
        while not game.is_terminal():
            move = random.choice(game.list_moves())
            game.play_move(move)

        return game.get_reward(scoring_player)

    def get_move(self, observation: Observation, configuration: Configuration) -> int:
        game = ConnectFour(board=observation.board, rows=configuration.rows,
                           columns=configuration.columns, inarow=configuration.inarow,
                           mark=observation.mark)

        visits = defaultdict(int)
        rewards = defaultdict(int)
        scoring_player = game.get_current_player()

        def move_score(move: int, steps_elapsed: int, exploration: float) -> float:
            if visits[move] == 0:
                return 10
            else:
                return rewards[move] / visits[move] \
                       + exploration * math.sqrt(2 * math.log(steps_elapsed) / visits[move])

        for i in range(self.max_steps):
            if self.ucb_selection:
                move = max(game.list_moves(), key=lambda m: move_score(m, i + 1, self.exploration_constant))
            else:
                move = random.choice(game.list_moves())
            result = self.evaluate_game_state(game.copy().play_move(move), scoring_player)
            visits[move] += 1
            rewards[move] += result

        chosen_move = max(visits.keys(), key=lambda m: move_score(m, 1, 0))

        return chosen_move


if __name__ == "__main__":
    from bachelorarbeit.selfplay import Arena
    from bachelorarbeit.tools import timer

    steps = 1000
    arena = Arena(
        players=(FlatMonteCarlo, RandomPlayer),
        constructor_args=({"max_steps": steps, "exploration_constant": 1 / math.sqrt(2)},
                          None),
        num_games=100,
        num_processes=4
    )
    results = arena.run_game_mp()
