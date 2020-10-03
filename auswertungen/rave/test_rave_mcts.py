from bachelorarbeit.players.mcts import MCTSPlayer
from bachelorarbeit.players.rave import RavePlayer
from bachelorarbeit.tools import run_selfplay_experiment, dump_json

TITLE = "Rave vs MCTS"
NUM_GAMES = 1000
NUM_PROCESSES = 9
MAX_STEPS = 1000

test_name = "alpha05"
settings = {
    "baseline": {},
    "alpha099": {"alpha": 0.99},
    "alpha09": {"alpha": 0.9},
    "alpha05": {"alpha": 0.5},
    "alpha0": {"alpha": 0.0},
    "k50": {"k": 50},
    "k100": {"k": 100},
    "k1000": {"k": 1000},
}

if __name__ == "__main__":
    base_sett = {"exploration_constant": 0.2, "max_steps": MAX_STEPS}

    experiment_setting = {**base_sett, **settings[test_name]}

    res = run_selfplay_experiment(
        TITLE,
        players=(RavePlayer, MCTSPlayer),
        constructor_args=(
            experiment_setting,
            {"exploration_constant": 0.85, "max_steps": MAX_STEPS}
        ),
        num_games=NUM_GAMES,
        num_processes=NUM_PROCESSES,
        show_progress_bar=True
    )

    dump_json("data/rave_mcts_"+test_name+"_{}.json", res)
