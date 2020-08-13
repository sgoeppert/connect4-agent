from typing import Tuple, Callable, List, Union, Type
import multiprocessing as mp
import numpy as np
from tqdm import tqdm
import json

from bachelorarbeit.games import Configuration, Observation, ConnectFour
from bachelorarbeit.base_players import Player
import config

GameResult = Tuple[float, float]


class Arena:
    def __init__(
            self,
            players: Tuple[Callable, Callable],
            constructor_args: Tuple[any, any] = (None, None),
            num_games: int = 30,
            num_processes: int = config.NUM_PROCESSES
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


class MoveEvaluation:
    def __init__(self,
                 player: Type[Player],
                 dataset_file: str,
                 player_config: Union[dict, None] = None,
                 num_processes: int = config.NUM_PROCESSES
    ):
        self.player_class = player
        self.player_config = player_config
        self.dataset_file = dataset_file
        self.num_processes = num_processes

    def instantiate_player(self):
        if self.player_config:
            return self.player_class(**self.player_config)
        else:
            return self.player_class()

    @staticmethod
    def win_loss_draw(score):
        if score > 0:
            return "win"
        elif score < 0:
            return "loss"
        return "draw"

    def evaluate_position(self, line: str):
        data = json.loads(line)
        board = data["board"]

        player = self.instantiate_player()
        _conf = Configuration()

        agent_move = player.get_move(Observation(board=board), _conf)

        moves = data["move score"]
        perfect_score = max(moves)
        perfect_moves = [i for i in range(len(moves)) if moves[i] == perfect_score]

        perfect = False
        good = False
        if agent_move in perfect_moves:
            perfect = True

        if MoveEvaluation.win_loss_draw(moves[agent_move]) == MoveEvaluation.win_loss_draw(perfect_score):
            good = True

        return good, perfect

    def score_player(self, show_progress_bar: bool = True) -> Tuple[int, int, int]:
        count = 0
        good_move_count = 0
        perfect_move_count = 0

        with open(self.dataset_file, "r") as f:
            lines = f.readlines()
            with mp.Pool(processes=self.num_processes) as pool:
                pending_results = pool.imap_unordered(self.evaluate_position, lines)
                if show_progress_bar:
                    pending_results = tqdm(pending_results, total=len(lines))

                for res in pending_results:
                    good, perfect = res
                    count += 1
                    good_move_count += good
                    perfect_move_count += perfect

                    if show_progress_bar:
                        perfect_perc = perfect_move_count / count
                        good_perc = good_move_count / count

                        pending_results.set_postfix({
                            "perfect": perfect_perc,
                            "good": good_perc
                        })

                if show_progress_bar:
                    pending_results.close()

        return good_move_count, perfect_move_count, count
