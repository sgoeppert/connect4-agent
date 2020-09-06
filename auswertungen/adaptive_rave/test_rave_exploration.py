from bachelorarbeit.players.mcts import MCTSPlayer
from bachelorarbeit.players.adaptive_rave import AdaptiveRavePlayer
from bachelorarbeit.tools import run_selfplay_experiment, dump_json

TITLE = "Rave exploration"
NUM_GAMES = 300
NUM_PROCESSES = 10

p1_values = [0.1, 0.2, 0.3, 0.4, 0.5]
p2 = 0.85

steps = 1000

if __name__ == "__main__":
    experiments = []

    for p1 in p1_values:
        res = run_selfplay_experiment(
            TITLE + " {} vs MCTS {}".format(p1, p2),
            (AdaptiveRavePlayer, MCTSPlayer),
            ({"exploration_constant": p1, "max_steps": steps, "keep_replies": True},
             {"exploration_constant": p2, "max_steps": steps}),
            num_games=NUM_GAMES,
            num_processes=NUM_PROCESSES,
            show_progress_bar=True)
        experiments.append(res)

    dump_json("data/explo_keep{}.json", experiments)
