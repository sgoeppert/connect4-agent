import pickle
from typing import Optional, Tuple, Callable, List, Dict, Union, Type
import multiprocessing as mp
import numpy as np
import math

from tqdm import tqdm

import os
from pathlib import Path
import json

from bachelorarbeit.games import Configuration, Observation, ConnectFour
from bachelorarbeit.players.base_players import Player
import config

GameResult = Tuple[float, float]

class Arena:
    def __init__(
            self,
            players: Tuple[Type[Player], Type[Player]],
            constructor_args: Tuple[any, any] = (None, None),
            num_games: int = 30,
            flip_halfway: bool = True,
            num_processes: int = config.NUM_PROCESSES,
            memory: "Memory" = None
    ):
        assert len(players) == 2, "Arena requires two players"
        assert num_processes > 0, "Argument num_processes must be positive"
        self.player_classes = players
        self.constructor_args = constructor_args
        self.num_games = num_games
        self.flip_halfway = flip_halfway
        self.num_processes = num_processes
        self.flip_players = False
        self.memory = memory

    def update_players(self, player_classes: Optional[Tuple[Callable, Callable]], constructor_args: Optional[Tuple[any, any]] = None):
        if player_classes is not None:
            self.player_classes = player_classes
        if constructor_args is not None:
            self.constructor_args = constructor_args

    def instantiate_players(self) -> List[Player]:
        players = []
        for pclass, arg in zip(self.player_classes, self.constructor_args):
            if arg:
                players.append(pclass(**arg))
            else:
                players.append(pclass())

        if self.flip_players:
            players = players[::-1]
        return players

    def run_game(self, _dummy=0) -> Tuple[GameResult, List]:
        conf = Configuration()
        game = ConnectFour(columns=conf.columns, rows=conf.rows, inarow=conf.inarow, mark=1)

        players = self.instantiate_players()

        s = 0
        game_states = []
        while not game.is_terminal():
            obs = Observation(board=game.board.copy(), mark=game.mark)
            game_states.append({"board": obs.board.copy(), "mark": obs.mark, "result": 0})
            active_player = s & 1
            m = players[active_player].get_move(obs, conf)
            # print(m)
            game.play_move(m)
            s += 1

        game_states.append({"board": game.board.copy(), "mark": game.mark, "result": 0})

        rewards = (game.get_reward(1), game.get_reward(2))
        step = 0
        for state in game_states[1:]:
            state["result"] = rewards[step % 2]
            step += 1
        # print(game_states, rewards)
        return rewards, game_states

    def run_game_mp(self, pool=None, show_progress_bar: bool = True, max_tasks: int = 10) -> List[GameResult]:
        self.flip_players = False
        mp.set_start_method("spawn", force=True)
        pbar = None
        if show_progress_bar:
            pbar = tqdm(total=self.num_games)

        n_games = [self.num_games]
        if self.flip_halfway:
            n_games = [self.num_games // 2, self.num_games - (self.num_games // 2)]

        game_results = []
        if pool is None:
            close_pool = True
            pool = mp.Pool(self.num_processes, maxtasksperchild=max_tasks)
        else:
            close_pool = False

        for num_games in n_games:
            pending_results = pool.imap_unordered(self.run_game, range(num_games))

            for res, game_states in pending_results:
                if self.flip_players:
                    game_results.append(res[::-1])
                else:
                    game_results.append(res)

                if self.memory is not None:
                    self.memory.add_full_game(game_states)

                if show_progress_bar:
                    mean_results = np.mean((np.array(game_results) + 1) / 2, axis=0)
                    pbar.set_postfix({
                        "Mean result": mean_results
                    }, refresh=False)
                    pbar.update()

            self.flip_players = True

        if close_pool:
            pool.close()

        if show_progress_bar:
            pbar.close()

        if self.memory is not None:
            self.memory.save_data()

        return game_results

# class TensorFlowArena(Arena):
#     def run_game(self, _dummy=0) -> Tuple[GameResult, List]:
#         from tensorflow import keras
#         return super(TensorFlowArena, self).run_game(_dummy)


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

        mark = (np.count_nonzero(board) % 2) + 1
        agent_move = player.get_move(Observation(board=board, mark=mark), _conf)

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


class Memory:
    def __init__(self, file_name: str, save_interval: int = 5000):
        self.file_name = Path(config.ROOT_DIR) / "memory" / file_name
        self.save_interval = save_interval
        self.game_data = []
        self.init()
        self.num_states = len(self.game_data)
        self.num_games = 0
        self.added_since_save = 0

    def set_file_name(self, file_name):
        self.file_name = Path(config.ROOT_DIR) / "memory" / file_name

    def init(self):
        try:
            self.load_data()
        except FileNotFoundError as e:
            print(f"{e}\nCreating new memory")
            self.game_data = []

    def load_data(self):
        if os.path.isfile(self.file_name):
            with open(self.file_name, "rb") as f:
                self.game_data = pickle.load(f)
                self.num_states = len(self.game_data)
        else:
            raise FileNotFoundError(f"Could not open file {self.file_name}.")

    def save_data(self):
        os.makedirs(os.path.dirname(self.file_name), exist_ok=True)

        with open(self.file_name, "wb") as f:
            pickle.dump(self.game_data, f)

    def add_full_game(self, game_states: List):
        for state in game_states:
            self.add_state(state)
        self.num_games += 1

    def add_state(self, state: Dict):
        self.game_data.append(state)
        self.num_states += 1

        self.added_since_save += 1
        if self.save_interval > 0 and self.added_since_save > self.save_interval:
            self.save_data()
            self.added_since_save = 0

    def add_other_memory(self, other: "Memory"):
        self.game_data.extend(other.game_data)

    def forget(self, amount: float = 0.2):
        num_to_forget = math.floor(self.num_states * amount)
        self.game_data = self.game_data[num_to_forget:]
        self.num_states -= num_to_forget

if __name__ == "__main__":
    from bachelorarbeit.players.mcts import MCTSPlayer
    # memory = Memory(file_name="random_data.pkl")
    arena = Arena(players=(MCTSPlayer, MCTSPlayer), num_games=1, num_processes=6)

    arena.run_game_mp()
