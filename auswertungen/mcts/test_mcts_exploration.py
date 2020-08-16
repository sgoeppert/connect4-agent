from bachelorarbeit.mcts import MCTSPlayer
from bachelorarbeit.tools import run_selfplay_experiment, dump_json

TITLE = "MCTS exploration"
NUM_GAMES = 500
NUM_PROCESSES = 8

p1_values = [0.8, 0.9, 1.0]
p2_values = [0.85]

if __name__ == "__main__":
    experiments = []

    for p1 in p1_values:
        for p2 in p2_values:
            if p1 == p2:  # skip mirror matches
                continue

            res = run_selfplay_experiment(TITLE + " {} vs {}".format(p1, p2), (MCTSPlayer, MCTSPlayer),
                                          ({"exploration_constant": p1}, {"exploration_constant": p2}), num_games=NUM_GAMES,
                                          num_processes=NUM_PROCESSES)
            experiments.append(res)

    dump_json("data/test_mcts_exploration_detailed_3_{}.json", experiments)
