from bachelorarbeit.players.mcts import MCTSPlayer
from bachelorarbeit.players.transposition import TranspositionPlayer
from bachelorarbeit.tools import run_selfplay_experiment, dump_json

TITLE = "Transposition exploration"
NUM_GAMES = 300
NUM_PROCESSES = 10

p1_values = [0.4, 0.45, 0.5, 0.55, 0.6]
p2 = 0.9

steps = 500

test_name = "baseline"
settings = {
    "baseline": {},
    "uct1": {"uct_method": "UCT1"},
    "uct2": {"uct_method": "UCT2"},
    "uct3": {"uct_method": "UCT3"},
    "keep_tree": {"keep_tree": True}
}

if __name__ == "__main__":
    experiments = []

    for exp in p1_values:
        base_sett = {"exploration_constant": exp, "max_steps": steps}
        experiment_setting = {**base_sett, **settings[test_name]}
        t = TITLE + " {} vs MCTS {}".format(exp, p2)
        print(t)
        res = run_selfplay_experiment(
            t,
            (TranspositionPlayer, MCTSPlayer),
            (experiment_setting,
             {"exploration_constant": p2, "max_steps": steps}),
            num_games=NUM_GAMES,
            num_processes=NUM_PROCESSES,
            show_progress_bar=True
        )
        experiments.append(res)

    dump_json("data/test_transposition_exploration_"+test_name+"_{}.json", experiments)
