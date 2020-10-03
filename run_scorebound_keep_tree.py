from bachelorarbeit.players.scorebounded import ScoreboundedPlayer
from bachelorarbeit.players.mcts import MCTSPlayer
from bachelorarbeit.tools import run_selfplay_experiment


if __name__ == "__main__":

    MAX_STEPS = 500
    cut_gamma = 0.0
    cut_delta = -0.1
    EXPLORATION_CONSTANT = 0.95
    MCTS_CP = 0.85

    # Keep tree both: 0.665, 0.335
    # Keep tree False, True: 0.633, 0.366
    res = run_selfplay_experiment(
        "Score Bounded vs MCTS, keeping trees",
        players=(ScoreboundedPlayer, MCTSPlayer),
        constructor_args=(
            {"max_steps": MAX_STEPS, "keep_tree": True,
             "cut_gamma": cut_gamma, "cut_delta": cut_delta,
             "exploration_constant": EXPLORATION_CONSTANT},
            {"max_steps": MAX_STEPS, "keep_tree": True, "exploration_constant": MCTS_CP}
        ),
        num_games=300,
        num_processes=10,
        show_progress_bar=True
    )

