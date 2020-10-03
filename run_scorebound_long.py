from bachelorarbeit.players.scorebounded import ScoreboundedPlayer
from bachelorarbeit.players.mcts import MCTSPlayer
from bachelorarbeit.tools import run_selfplay_experiment, dump_json


if __name__ == "__main__":

    MAX_STEPS = 10000
    SETTINGS = [(0,0), (0,0.2)]
    # METHODS = ["UCT1", "UCT2", "UCT3", "Standard"]
    EXPLORATION_CONSTANT = 0.95
    MCTS_CP = 0.85

    results = {"max_steps": MAX_STEPS, "Cp": EXPLORATION_CONSTANT}
    for sett in SETTINGS:
        res = run_selfplay_experiment(
            "Score bounded vs MCTS, keeping trees",
            players=(ScoreboundedPlayer, MCTSPlayer),
            constructor_args=(
                {"max_steps": MAX_STEPS, "keep_tree": True, "exploration_constant": EXPLORATION_CONSTANT,
                 "cut_gamma": sett[0], "cut_delta": sett[1]},
                {"max_steps": MAX_STEPS, "keep_tree": True, "exploration_constant": MCTS_CP}
            ),
            num_games=500,
            show_progress_bar=True
        )
        results[str(sett)] = res["mean"]

    dump_json("long_runs/scorebound_{}.json", results)

