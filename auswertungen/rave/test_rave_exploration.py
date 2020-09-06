from bachelorarbeit.players.mcts import MCTSPlayer
from bachelorarbeit.players.rave import RavePlayer
from bachelorarbeit.tools import run_selfplay_experiment, dump_json

TITLE = "Rave exploration"
NUM_GAMES = 300
NUM_PROCESSES = 10

p1_values = [0.2, 0.25, 0.3, 0.35, 0.4]
p2 = 0.85

steps = 1000

if __name__ == "__main__":
    experiments = []

    for p1 in p1_values:
        res = run_selfplay_experiment(
            TITLE + " {} vs MCTS {}".format(p1, p2),
            (RavePlayer, MCTSPlayer),
            ({"exploration_constant": p1, "max_steps": steps},
             {"exploration_constant": p2, "max_steps": steps}),
            num_games=NUM_GAMES,
            num_processes=NUM_PROCESSES,
            show_progress_bar=True)
        experiments.append(res)

    dump_json("data/test_rave_exploration_detail_{}.json", experiments)
