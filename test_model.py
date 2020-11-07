from bachelorarbeit.games import ConnectFour, Observation, Configuration
from bachelorarbeit.players.network_player import NetworkPlayer
from bachelorarbeit.tools import transform_board_cnn, timer, denormalize, transform_board_nega
from bachelorarbeit.players.mcts import MCTSPlayer
from bachelorarbeit.selfplay import Arena
import config

import os
from datetime import datetime

MAX_STEPS = 100

base_configs = [
    {
        "max_steps": MAX_STEPS,
        "exploration_constant": 0.8,
        "network_weight": 0.5,
    },
    {
        "max_steps": MAX_STEPS,
        "exploration_constant": 0.8,
        "network_weight": 0.85,
    },
]

configs = [
    {
        "model_path": config.ROOT_DIR + "/best_models/400000/padded_cnn_norm",
        "transform_func": transform_board_cnn,
        "transform_output": denormalize,
    },
    {
        "model_path": config.ROOT_DIR + "/best_models/400000/padded_cnn",
        "transform_func": transform_board_cnn,
        "transform_output": None,
    },
    {
        "model_path": config.ROOT_DIR + "/best_models/400000/regular_norm",
        "transform_func": transform_board_nega,
        "transform_output": denormalize,
    },
    {
        "model_path": config.ROOT_DIR + "/best_models/400000/regular",
        "transform_func": transform_board_nega,
        "transform_output": None,
    },
]

LOGFILE = "network_test/testrun_{}.log"
TIMESTAMP = datetime.now().strftime("%Y%m%d-%H%M%S")
LOGFILE = LOGFILE.format(TIMESTAMP)
os.makedirs(os.path.dirname(LOGFILE), exist_ok=True)

def log(*args):
    with open(LOGFILE, "a+") as f:
        f.write(" ".join([str(a) for a in args]) + "\n")
    print(*args)


if __name__ == "__main__":
    import numpy as np
    import time

    for base_config in base_configs:
        for conf in configs:
            player_config = {**base_config, **conf}
            log(player_config)

            start = time.time()
            arena = Arena(players=(NetworkPlayer, MCTSPlayer),
                          constructor_args=(player_config, {"max_steps": 400, "exploration_constant": 0.8}),
                          num_processes=8,
                          num_games=500,
                          )
            results = arena.run_game_mp()
            elapsed = time.time() - start

            mean_results = np.mean((np.array(results) + 1) / 2, axis=0)
            log("Elapsed: ", elapsed)
            log("Weight: ", player_config["network_weight"])
            log("Model: ", player_config["model_path"])
            log("Result: ", mean_results)
            log("=" * 10)
