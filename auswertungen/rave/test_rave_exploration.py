from bachelorarbeit.mcts import MCTSPlayer
from bachelorarbeit.rave import RavePlayer
from bachelorarbeit.tools import run_selfplay_experiment, dump_json

TITLE = "Rave exploration"
NUM_GAMES = 300
NUM_PROCESSES = 10

p1_values = [0.3, 0.4, 0.5, 0.6]
p2 = 0.9

steps = 500

if __name__ == "__main__":
    experiments = []

    for p1 in p1_values:
        res = run_selfplay_experiment(
            TITLE + " {} vs MCTS {}".format(p1, p2),
            (RavePlayer, MCTSPlayer),
            ({"exploration_constant": p1, "max_steps": steps, "beta": 0.98},
             {"exploration_constant": p2, "max_steps": steps}),
            num_games=NUM_GAMES,
            num_processes=NUM_PROCESSES)
        experiments.append(res)

    dump_json("data/test_rave_exploration_detail_{}.json", experiments)
