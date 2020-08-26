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

    def __repr__(self) -> str:
        return self.name


class RandomPlayer(Player):
    name = "RandomPlayer"

    def __init__(self, **kwargs):
        super(RandomPlayer, self).__init__(**kwargs)

    def get_move(self, observation: Observation, configuration: Configuration) -> int:
        return random.choice([c for c in range(configuration.columns) if observation.board[c] == 0])


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
