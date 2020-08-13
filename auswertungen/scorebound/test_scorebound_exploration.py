from bachelorarbeit.mcts import MCTSPlayer
from bachelorarbeit.scorebounded import ScoreboundedPlayer
from bachelorarbeit.tools import run_selfplay_experiment, dump_json

TITLE = "Scorebounded exploration"
NUM_GAMES = 200
NUM_PROCESSES = 6

p1_values = [0.5, 0.8, 1.0, 1.2, 1.5]
p2 = 0.9

steps = 400

if __name__ == "__main__":
    experiments = []

    for p1 in p1_values:
        res = run_selfplay_experiment(
            TITLE + " {} vs MCTS {}".format(p1, p2),
            (ScoreboundedPlayer, MCTSPlayer),
            ({"exploration_constant": p1, "max_steps": steps},
             {"exploration_constant": p2, "max_steps": steps}),
            num_games=NUM_GAMES,
            num_processes=NUM_PROCESSES)
        experiments.append(res)

    dump_json("data/test_scorebound_exploration_{}.json", experiments)
