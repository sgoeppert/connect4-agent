from bachelorarbeit.mcts import MCTSPlayer
from bachelorarbeit.sarsa_rave import SarsaPlayer
from bachelorarbeit.tools import run_selfplay_experiment, dump_json

TITLE = "Sarsa vs MCTS"
NUM_GAMES = 500
NUM_PROCESSES = 9

test_name = "lamda09"

if __name__ == "__main__":
    settings = {
        "baseline": None,
        "beta09": {"beta": 0.9},
        "beta09_lambda095": {"beta": 0.9, "lamda": 0.95},
        "lamda09": {"lamda": 0.9},
        "lamda095": {"lamda": 0.95},
        "lamda099": {"lamda": 0.99},
    }

    res = run_selfplay_experiment(
        TITLE,
        players=(SarsaPlayer, MCTSPlayer),
        constructor_args=(
            settings[test_name],
            None
        ),
        num_games=NUM_GAMES,
        num_processes=NUM_PROCESSES)

    dump_json("data/test_sarsa_mcts_"+test_name+"_{}.json", res)
