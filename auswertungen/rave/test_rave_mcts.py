from bachelorarbeit.mcts import MCTSPlayer
from bachelorarbeit.rave import RavePlayer
from bachelorarbeit.tools import run_selfplay_experiment, dump_json

TITLE = "Rave vs MCTS"
NUM_GAMES = 500
NUM_PROCESSES = 9

if __name__ == "__main__":
    res = run_selfplay_experiment(
        TITLE,
        players=(RavePlayer, MCTSPlayer),
        num_games=NUM_GAMES,
        num_processes=NUM_PROCESSES)

    dump_json("data/test_rave_mcts_{}.json", res)
