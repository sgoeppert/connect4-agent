"""
Vergleicht die Agenten mit gleicher Rechenzeit pro Zug. Für jeden Agenten wurde zuvor die Anzahl Iterationen pro
Sekunde bestimmt und ins Verhältnis zum normalen MCTS-Agenten gesetzt.
"""

from bachelorarbeit.players.adaptive_network_player import AdaptiveNetworkPlayer
from bachelorarbeit.players.adaptive_playout import AdaptivePlayoutPlayer
from bachelorarbeit.players.adaptive_rave import AdaptiveRavePlayer
from bachelorarbeit.players.adaptive_rave_network_player import AdaptiveRaveNetworkPlayer
from bachelorarbeit.players.network_player import NetworkPlayer
from bachelorarbeit.players.transposition import TranspositionPlayer
from bachelorarbeit.players.rave import RavePlayer
from bachelorarbeit.players.mcts import MCTSPlayer

from bachelorarbeit.tools import timer, transform_board_cnn, transform_board_nega
from bachelorarbeit.selfplay import Arena
import config

from datetime import datetime
import os
import numpy as np

LOGFILE = "scaling_test/run_keep_tree_{}.log"
TIMESTAMP = datetime.now().strftime("%Y%m%d-%H%M%S")
LOGFILE = LOGFILE.format(TIMESTAMP)
os.makedirs(os.path.dirname(LOGFILE), exist_ok=True)


def log(*args):
    with open(LOGFILE, "a+") as f:
        f.write(" ".join([str(a) for a in args]) + "\n")
    print(*args)


if __name__ == "__main__":

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
                "exploration_constant": 0.8,
                "uct_method": "UCT1",
                "with_symmetry": False,
            },
        },
        {
            "player": TranspositionPlayer,
            "conf": {
                "_steps": 850,
                "exploration_constant": 1.0,
                "uct_method": "UCT2",
                "with_symmetry": False,
            },
        },
        {
            "player": TranspositionPlayer,
            "conf": {
                "_steps": 695,
                "exploration_constant": 0.9,
                "uct_method": "UCT3",
                "with_symmetry": False,
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
                    "_steps": 809,
                    "exploration_constant": 0.2,
                    "k": 100,
            },
        },
        {
            "player": AdaptivePlayoutPlayer,
            "conf": {
                  "_steps": 654,
                  "exploration_constant": 1.0,
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
            "player": AdaptiveRavePlayer,
            "conf": {
                  "_steps": 563,
                  "exploration_constant": 0.2,
                  "k": 50,
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
                  "exploration_constant": 0.3,
                  "network_weight": 0.5,
                  "k": 100,
                  "keep_replies": True
              },
        },
        {
            "player": AdaptiveRaveNetworkPlayer,
            "conf": {
                  "model_path": config.ROOT_DIR + "/best_models/400000/regular_norm",
                  "transform_func": transform_board_nega,
                  "_steps": 110,
                  "exploration_constant": 0.3,
                  "network_weight": 0.5,
                  "k": 100,
                  "keep_replies": True
              },
        }
    ]

    # configs = configs[-2:]

    keep_tree = True
    mcts_steps = 1000
    for scale in [1,2,3,4]:
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
                          num_games=500,
                          num_processes=10,
                          )
            results = arena.run_game_mp(show_progress_bar=True, max_tasks=2)

            mean_results = np.mean((np.array(results) + 1) / 2, axis=0)
            log("Result: ", mean_results)
            log("=" * 10)
