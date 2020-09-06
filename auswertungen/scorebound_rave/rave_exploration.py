from bachelorarbeit.players.mcts import MCTSPlayer
from bachelorarbeit.players.scorebound_rave import ScoreBoundedRavePlayer
from bachelorarbeit.tools import run_selfplay_experiment, dump_json

TITLE = "Rave exploration"
NUM_GAMES = 300
NUM_PROCESSES = 10

p1_values = [0.6, 1.0, 1.3]
p2 = 0.85

steps = 1000

test_name = "baseline"
settings = {
    "baseline": {},
    "sb_opt": {"delta": 0.2},
    "alpha099": {"alpha": 0.99},
    "alpha09": {"alpha": 0.9},
    "alpha05": {"alpha": 0.5},
    "alpha0": {"alpha": 0.0},
    "b0001": {"b": 0.001},
    "b01": {"b": 0.1},
}

if __name__ == "__main__":
    experiments = []

    for p1 in p1_values:

        base_sett = {"exploration_constant": p1, "max_steps": steps}
        experiment_setting = {**base_sett, **settings[test_name]}

        res = run_selfplay_experiment(
            TITLE + " {} vs MCTS {}".format(p1, p2),
            (ScoreBoundedRavePlayer, MCTSPlayer),
            (base_sett,
             {"exploration_constant": p2, "max_steps": steps}),
            num_games=NUM_GAMES,
            num_processes=NUM_PROCESSES,
            show_progress_bar=True)
        experiments.append(res)

    dump_json("data/exploration_"+test_name+"_{}.json", experiments)
