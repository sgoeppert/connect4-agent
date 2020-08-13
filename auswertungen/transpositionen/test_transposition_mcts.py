from bachelorarbeit.transposition import TranspositionPlayer
from bachelorarbeit.mcts import MCTSPlayer
from bachelorarbeit.tools import run_selfplay_experiment, dump_json

TITLE = "Transposition vs MCTS"
NUM_GAMES = 500
NUM_PROCESSES = 6

if __name__ == "__main__":
    res = run_selfplay_experiment(TITLE,
                                  players=(TranspositionPlayer, MCTSPlayer),
                                  num_games=NUM_GAMES,
                                  num_processes=NUM_PROCESSES)

    dump_json("data/test_transposition_mcts_{}.json", res)
