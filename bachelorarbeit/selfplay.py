from typing import Tuple, Callable, List, Type
import multiprocessing as mp
import numpy as np
from tqdm import tqdm

from bachelorarbeit.games import Configuration, Observation, ConnectFour
from bachelorarbeit.base_players import Player

GameResult = Tuple[float, float]


class Arena:
    def __init__(
            self,
            players: Tuple[Callable, Callable],
            constructor_args: Tuple[any, any] = (None, None),
            num_games: int = 30,
            num_processes: int = 8
    ):
        assert len(players) == 2, "Arena requires two players"
        assert num_processes > 0, "Argument num_processes must be positive"
        self.player_classes = players
        self.constructor_args = constructor_args
        self.num_games = num_games
        self.num_processes = num_processes

    def update_players(self, player_classes: Tuple[Callable, Callable], constructor_args: Tuple[any, any] = (None, None)):
        self.player_classes = player_classes
        self.constructor_args = constructor_args

    def instantiate_players(self) -> List[Player]:
        players = []
        for pclass, arg in zip(self.player_classes, self.constructor_args):
            if arg:
                players.append(pclass(**arg))
            else:
                players.append(pclass())
        return players

    def run_game(self, _dummy=0) -> GameResult:
        conf = Configuration()
        game = ConnectFour(columns=conf.columns, rows=conf.rows, inarow=conf.inarow, mark=1)

        players = self.instantiate_players()

        s = 0
        while not game.is_terminal():
            obs = Observation(board=game.board.copy(), mark=game.mark)
            active_player = s & 1
            m = players[active_player].get_move(obs, conf)
            game.play_move(m)
            s += 1

        return game.get_reward(1), game.get_reward(2)

    def run_game_mp(self, show_progress_bar: bool = True) -> List[GameResult]:
        with mp.Pool(self.num_processes, maxtasksperchild=10) as pool:
            pending_results = pool.imap_unordered(self.run_game, range(self.num_games))
            game_results = []

            if show_progress_bar:
                pending_results = tqdm(pending_results, total=self.num_games)

            for res in pending_results:
                game_results.append(res)
                if show_progress_bar:
                    mean_results = np.mean((np.array(game_results) + 1) / 2, axis=0)
                    pending_results.set_postfix({
                        "Mean result": mean_results
                    })

            if show_progress_bar:
                pending_results.close()

        return game_results
