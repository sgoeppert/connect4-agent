from bachelorarbeit.mcts import MCTSPlayer
from bachelorarbeit.sarsa_rave import SarsaPlayer
from bachelorarbeit.tools import run_selfplay_experiment, dump_json

TITLE = "Sarsa exploration"
NUM_GAMES = 500
NUM_PROCESSES = 8

p1_values = [0.1, 0.15, 0.2, 0.25, 0.3]
p2 = 0.9

steps = 1000

test_name = "lamda09"
settings = {
    "baseline": {},
    "beta09": {"beta": 0.9},
    "beta09_lambda095": {"beta": 0.9, "lamda": 0.95},
    "lamda09": {"lamda": 0.9},
    "lamda095": {"lamda": 0.95},
    "lamda099": {"lamda": 0.99},
}

if __name__ == "__main__":
    experiments = []

    for exp in p1_values:
        base_sett = {"exploration_constant": exp, "max_steps": steps}
        experiment_setting = {**base_sett, **settings[test_name]}

        res = run_selfplay_experiment(
            TITLE + " {} vs MCTS {}".format(exp, p2),
            (SarsaPlayer, MCTSPlayer),
            (experiment_setting,
             {"exploration_constant": p2, "max_steps": steps}),
            num_games=NUM_GAMES,
            num_processes=NUM_PROCESSES)
        experiments.append(res)

    dump_json("data/test_sarsa_exploration_"+test_name+"_{}.json", experiments)
