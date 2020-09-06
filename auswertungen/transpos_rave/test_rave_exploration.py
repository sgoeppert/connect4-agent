from bachelorarbeit.players.mcts import MCTSPlayer
from bachelorarbeit.players.transposition_rave import TpRavePlayer
from bachelorarbeit.tools import run_selfplay_experiment, dump_json

TITLE = "Rave exploration"
NUM_GAMES = 300
NUM_PROCESSES = 10

p1_values = [0.05, 0.1, 0.15, 0.2]
p2 = 0.85

steps = 1000

if __name__ == "__main__":
    experiments = []

    for p1 in p1_values:
        res = run_selfplay_experiment(
            TITLE + " {} vs MCTS {}".format(p1, p2),
            (TpRavePlayer, MCTSPlayer),
            ({"exploration_constant": p1, "max_steps": steps},
             {"exploration_constant": p2, "max_steps": steps}),
            num_games=NUM_GAMES,
            num_processes=NUM_PROCESSES,
            show_progress_bar=True)
        experiments.append(res)

    dump_json("data/test_rave_exploration_{}.json", experiments)
