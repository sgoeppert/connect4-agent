import json
import os
from datetime import datetime
from typing import Tuple, Optional, Type
import numpy as np
from pathlib import Path
from contextlib import contextmanager
import time

from bachelorarbeit.selfplay import Arena, MoveEvaluation
from bachelorarbeit.base_players import Player, RandomPlayer, FlatMonteCarlo
import config

def run_selfplay_experiment(
        title: str,
        players: Tuple[Type[Player], Type[Player]],
        constructor_args: Tuple[any, any] = (None, None),
        num_games: int = 100,
        num_processes: int = config.NUM_PROCESSES
):
    print("Running Experiment: ", title)
    arena = Arena(players, constructor_args=constructor_args, num_processes=num_processes, num_games=num_games)

    results = arena.run_game_mp()
    mean_scores = (np.mean(results, axis=0) + 1) / 2  # calculate the mean score per player as value between 0 and 1

    return {
        "title": title,
        "num_games": num_games,
        "players": [p.name for p in players],
        "configurations": constructor_args,
        "raw_results": results,
        "mean": mean_scores.tolist()
    }


def evaluate_against_random(
        player: Type[Player],
        constructor_args: Optional[dict] = None,
        num_games: int = 100,
        num_processes: int = config.NUM_PROCESSES
):
    print("Evaluation against random player: ", player, constructor_args)
    arena = Arena(players=(player, RandomPlayer), constructor_args=(constructor_args, None), num_processes=num_processes, num_games=num_games)

    results = arena.run_game_mp()
    mean_scores = (np.mean(results, axis=0) + 1) / 2  # calculate the mean score per player as value between 0 and 1

    return {
        "title": "Eval against random",
        "num_games": num_games,
        "player": player.name,
        "configuration": constructor_args,
        # "raw_results": results,
        "mean": mean_scores.tolist()
    }


def evaluate_against_flat_monte_carlo(
        player: Type[Player],
        constructor_args: Optional[dict] = None,
        opponent_steps: int = 500,
        num_games: int = 100,
        num_processes: int = config.NUM_PROCESSES
):
    print(f"Evaluation against flat Monte Carlo player({opponent_steps} steps): ", player.name, constructor_args)
    arena = Arena(players=(player, FlatMonteCarlo), constructor_args=(constructor_args, {"max_steps": opponent_steps}), num_processes=num_processes, num_games=num_games)

    results = arena.run_game_mp()
    mean_scores = (np.mean(results, axis=0) + 1) / 2  # calculate the mean score per player as value between 0 and 1

    return {
        "title": "Eval against flat MC",
        "num_games": num_games,
        "player": player.name,
        "opponent_steps": opponent_steps,
        "configuration": constructor_args,
        # "raw_results": results,
        "mean": mean_scores.tolist()
    }


def run_move_evaluation_experiment(
        title: str,
        player: Type[Player],
        player_config: Optional[dict] = None,
        num_processes: int = config.NUM_PROCESSES,
        repeats: int = 1
):
    dataset_file = str(Path(config.ROOT_DIR) / "auswertungen" / "data" / "refmoves1k_kaggle")

    good, perfect, total = 0, 0, 0
    for it in range(repeats):
        evaluator = MoveEvaluation(
            player=player,
            player_config=player_config,
            dataset_file=dataset_file,
            num_processes=num_processes
        )
        _good, _perfect, _total = evaluator.score_player()
        good += _good
        perfect += _perfect
        total += _total

    return {
        "title": title,
        "player": player.name,
        "configuration": player_config,
        "repeats": repeats,
        "n_positions": total // repeats,
        "perfect_pct": perfect/total,
        "good_pct": good/total,
        "raw_results": {
            "total": total,
            "perfect": perfect,
            "good": good,
        }
    }

def dump_json(filename, data):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_file = filename.format(timestamp)
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    with open(out_file, "w+") as f:
        json.dump(data, f)
    print("Wrote results to file ", out_file)


@contextmanager
def timer(name="Timer"):
    tick = time.time()
    yield
    tock = time.time()
    print(f"{name} took {tock-tick}s")
