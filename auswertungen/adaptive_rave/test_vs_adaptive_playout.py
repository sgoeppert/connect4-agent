from bachelorarbeit.mcts import MCTSPlayer
from bachelorarbeit.adaptive_rave import AdaptiveRavePlayer
from bachelorarbeit.adaptive_playout import AdaptivePlayoutPlayer
from bachelorarbeit.tools import run_selfplay_experiment, dump_json

TITLE = "AdaptiveRave vs Adaptive Playout"
NUM_GAMES = 500
NUM_PROCESSES = 7
MAX_STEPS = 600

if __name__ == "__main__":
    res = run_selfplay_experiment(
        TITLE,
        players=(AdaptiveRavePlayer, AdaptivePlayoutPlayer),
        constructor_args=({"max_steps": MAX_STEPS, "keep_replies": True, "forgetting": False},
                          {"max_steps": MAX_STEPS, "keep_replies": True, "forgetting": False}),
        num_games=NUM_GAMES,
        num_processes=NUM_PROCESSES,
        show_progress_bar=True
    )

    dump_json("data/vs_adaptive_playout_{}.json", res)
