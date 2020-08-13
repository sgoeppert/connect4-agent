from bachelorarbeit.mcts import MCTSPlayer
from bachelorarbeit.tools import run_selfplay_experiment, dump_json

TITLE = "MCTS Keep tree"
NUM_GAMES = 100
NUM_PROCESSES = 4

p1_values = [True, False]
p2_values = [True, False]

exploration_constant = 0.9


if __name__ == "__main__":

    experiments = []

    for p1 in p1_values:
        for p2 in p2_values:
            res = run_selfplay_experiment(TITLE + " {} vs {}".format(p1, p2), (MCTSPlayer, MCTSPlayer),
                                          ({"exploration_constant": exploration_constant, "keep_tree": p1},
                                           {"exploration_constant": exploration_constant, "keep_tree": p2}),
                                          num_games=NUM_GAMES, num_processes=NUM_PROCESSES)
            experiments.append(res)

    dump_json("data/test_mcts_tree_{}.json", experiments)
