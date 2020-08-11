import json
import os
from datetime import datetime
from typing import Tuple, Callable, Type
from bachelorarbeit.selfplay import Arena
from bachelorarbeit.base_players import Player
import numpy as np

NUM_PROCESSES = 8


def run_experiment(
        title: str,
        players: Tuple[Type[Player], Type[Player]],
        constructor_args: Tuple[any, any] = (None, None),
        num_games: int = 100,
        num_processes: int = NUM_PROCESSES
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


def dump_json(filename, data):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_file = filename.format(timestamp)
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    with open(out_file, "w+") as f:
        json.dump(data, f)
    print("Wrote results to file ", out_file)
