from bachelorarbeit.players.adaptive_network_player import AdaptiveNetworkPlayer
from bachelorarbeit.players.adaptive_playout import AdaptivePlayoutPlayer
from bachelorarbeit.players.adaptive_rave_network_player import AdaptiveRaveNetworkPlayer
from bachelorarbeit.players.network_player import NetworkPlayer
from bachelorarbeit.players.rave import RavePlayer
from bachelorarbeit.players.transposition import TranspositionPlayer

from datetime import datetime
import os
import time
import numpy as np

LOGFILE = "scaling_test/run_vs_fixed_long_selection_{}.log"
TIMESTAMP = datetime.now().strftime("%Y%m%d-%H%M%S")
LOGFILE = LOGFILE.format(TIMESTAMP)
os.makedirs(os.path.dirname(LOGFILE), exist_ok=True)


def log(*args):
    with open(LOGFILE, "a+") as f:
        f.write(" ".join([str(a) for a in args]) + "\n")
    print(*args)


if __name__ == "__main__":
    from bachelorarbeit.games import Observation, Configuration, ConnectFour
    from bachelorarbeit.tools import timer, transform_board_cnn, transform_board_nega
    from bachelorarbeit.players.mcts import MCTSPlayer
    from bachelorarbeit.players.adaptive_rave import AdaptiveRavePlayer
    from bachelorarbeit.selfplay import Arena
    import config

    configs = [
        {
            "player": MCTSPlayer,
            "conf": {
                "_steps": 1000,
                "exploration_constant": 0.9,
            },
        },
        {
            "player": TranspositionPlayer,
            "conf": {
                "_steps": 850,
                "exploration_constant": 0.9,
                "uct_method": "UCT2",
                "with_symmetry": True,
            },
        },
        {
            "player": TranspositionPlayer,
            "conf": {
                "_steps": 695,
                "exploration_constant": 0.9,
                "uct_method": "UCT3",
                "with_symmetry": True,
            },
        },
        {
            "player": RavePlayer,
            "conf": {
                    "_steps": 809,
                    "exploration_constant": 0.4,
                    "alpha": 0.5,
            },
        },
        {
            "player": RavePlayer,
            "conf": {
                "_steps": 834,
                "exploration_constant": 0.4,
                "alpha": None,
            },
        },
        {
            "player": AdaptivePlayoutPlayer,
            "conf": {
                  "_steps": 654,
                  "exploration_constant": 0.8,
                  "keep_replies": True,
              },
        },
        {
            "player": AdaptiveRavePlayer,
            "conf": {
                  "_steps": 563,
                  "exploration_constant": 0.4,
                  "alpha": 0.5,
                  "keep_replies": True,
              },
        },
        {
            "player": NetworkPlayer,
            "conf": {
                  "model_path": config.ROOT_DIR + "/best_models/400000/padded_cnn_norm",
                  "transform_func": transform_board_cnn,
                  "_steps": 52,
                  "exploration_constant": 0.8,
                  "network_weight": 0.5
              },
        },{
            "player": NetworkPlayer,
            "conf": {
                  "model_path": config.ROOT_DIR + "/best_models/400000/regular_norm",
                  "transform_func": transform_board_nega,
                  "_steps": 129,
                  "exploration_constant": 0.8,
                  "network_weight": 0.5
              },
        },
        {
            "player": AdaptiveNetworkPlayer,
            "conf": {
                  "model_path": config.ROOT_DIR + "/best_models/400000/padded_cnn_norm",
                  "transform_func": transform_board_cnn,
                  "_steps": 52,
                  "exploration_constant": 0.8,
                  "network_weight": 0.5,
                  "keep_replies": True
              },
        },{
            "player": AdaptiveNetworkPlayer,
            "conf": {
                  "model_path": config.ROOT_DIR + "/best_models/400000/regular_norm",
                  "transform_func": transform_board_nega,
                  "_steps": 119,
                  "exploration_constant": 0.8,
                  "network_weight": 0.5,
                  "keep_replies": True
              },
        },
        {
            "player": AdaptiveRaveNetworkPlayer,
            "conf": {
                  "model_path": config.ROOT_DIR + "/best_models/400000/padded_cnn_norm",
                  "transform_func": transform_board_cnn,
                  "_steps": 52,
                  "exploration_constant": 0.4,
                  "network_weight": 0.5,
                  "alpha": None,
                  "keep_replies": True
              },
        },
        {
            "player": AdaptiveRaveNetworkPlayer,
            "conf": {
                  "model_path": config.ROOT_DIR + "/best_models/400000/regular_norm",
                  "transform_func": transform_board_nega,
                  "_steps": 110,
                  "exploration_constant": 0.4,
                  "network_weight": 0.5,
                  "alpha": None,
                  "keep_replies": True
              },
        }
    ]

    configs = [configs[1], configs[2], configs[3], configs[4]]

    keep_tree = True
    mcts_steps = 1000
    for scale in [10]:
        for config in configs:
            p = config["player"]
            conf = config["conf"]

            conf["max_steps"] = int(conf["_steps"] * scale)
            conf["keep_tree"] = keep_tree

            log("Player: ", p)
            log("Scale: ", scale)
            log("Config: ", conf)

            arena = Arena(players=(p, MCTSPlayer),
                          constructor_args=(conf,
                              {
                                  "max_steps": mcts_steps * scale,
                                  "exploration_constant": 0.8,
                                  "keep_tree": keep_tree
                              }),
                          num_games=400,
                          num_processes=10
                          )
            results = arena.run_game_mp(show_progress_bar=True)

            mean_results = np.mean((np.array(results) + 1) / 2, axis=0)
            log("Result: ", mean_results)
            log("=" * 10)
