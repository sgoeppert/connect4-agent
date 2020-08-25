from bachelorarbeit.transposition import TranspositionPlayer
from bachelorarbeit.mcts import MCTSPlayer
from bachelorarbeit.tools import run_selfplay_experiment


if __name__ == "__main__":

    MAX_STEPS = 10000
    UCT_METHOD = "default"
    EXPLORATION_CONSTANT = 0.5
    MCTS_CP = 0.85

    # 5000 default - Keep tree both:
    res = run_selfplay_experiment(
        "Transpos vs MCTS, keeping trees",
        players=(TranspositionPlayer, MCTSPlayer),
        constructor_args=(
            {"max_steps": MAX_STEPS, "keep_tree": True, "uct_method": UCT_METHOD, "exploration_constant": EXPLORATION_CONSTANT},
            {"max_steps": MAX_STEPS, "keep_tree": True, "exploration_constant": MCTS_CP}
        ),
        num_games=300,
        num_processes=10,
        show_progress_bar=True
    )

