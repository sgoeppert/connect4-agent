from bachelorarbeit.players.mcts import MCTSPlayer
from bachelorarbeit.players.adaptive_playout import AdaptivePlayoutPlayer
from bachelorarbeit.tools import run_selfplay_experiment, dump_json

TITLE = "AdaptivePlayout vs MCTS"
NUM_GAMES = 500
NUM_PROCESSES = 10
MAX_STEPS = 1000

if __name__ == "__main__":
    res = run_selfplay_experiment(
        TITLE,
        players=(AdaptivePlayoutPlayer, MCTSPlayer),
        constructor_args=({"max_steps": MAX_STEPS, "keep_replies": True, "forgetting": False},
                          {"max_steps": MAX_STEPS}),
        num_games=NUM_GAMES,
        num_processes=NUM_PROCESSES,
        show_progress_bar=True
    )

    dump_json("data/vs_mcts_{}.json", res)
