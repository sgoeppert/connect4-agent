import random
from abc import ABC, abstractmethod

from bachelorarbeit.games import Observation, Configuration


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
