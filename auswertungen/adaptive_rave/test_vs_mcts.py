from bachelorarbeit.players.mcts import MCTSPlayer
from bachelorarbeit.players.adaptive_rave import AdaptiveRavePlayer
from bachelorarbeit.tools import run_selfplay_experiment, dump_json

TITLE = "AdaptiveRave vs MCTS"
NUM_GAMES = 100
NUM_PROCESSES = 7
MAX_STEPS = 500

if __name__ == "__main__":
    res = run_selfplay_experiment(
        TITLE,
        players=(AdaptiveRavePlayer, MCTSPlayer),
        constructor_args=({"max_steps": MAX_STEPS, "keep_tree": False, "keep_replies": True, "forgetting": False},
                          {"max_steps": MAX_STEPS, "keep_tree": True}),
        num_games=NUM_GAMES,
        num_processes=NUM_PROCESSES,
        show_progress_bar=True
    )

    dump_json("data/vs_mcts_{}.json", res)
