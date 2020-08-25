from bachelorarbeit.transposition import TranspositionPlayer
from bachelorarbeit.mcts import MCTSPlayer
from bachelorarbeit.tools import run_selfplay_experiment, dump_json


if __name__ == "__main__":

    MAX_STEPS = 10000
    METHODS = ["UCT1", "UCT2", "UCT3", "Standard"]
    EXPLORATION_CONSTANT = 0.5
    MCTS_CP = 0.85

    results = {"max_steps": MAX_STEPS, "Cp": EXPLORATION_CONSTANT}
    for method in METHODS:
        res = run_selfplay_experiment(
            "Transpos vs MCTS, keeping trees",
            players=(TranspositionPlayer, MCTSPlayer),
            constructor_args=(
                {"max_steps": MAX_STEPS, "keep_tree": True, "uct_method": method, "exploration_constant": EXPLORATION_CONSTANT},
                {"max_steps": MAX_STEPS, "keep_tree": True, "exploration_constant": MCTS_CP}
            ),
            num_games=500,
            show_progress_bar=True
        )
        results[method] = res["mean"]

    dump_json("long_runs/transposition_{}.json", results)

