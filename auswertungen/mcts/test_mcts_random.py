from bachelorarbeit.base_players import RandomPlayer
from bachelorarbeit.mcts import MCTSPlayer
from bachelorarbeit.tools import run_selfplay_experiment, dump_json

TITLE = "MCTS vs Random"
NUM_GAMES = 100
NUM_PROCESSES = 8

if __name__ == "__main__":
    res = run_selfplay_experiment(TITLE, (MCTSPlayer, RandomPlayer), num_games=NUM_GAMES)

    dump_json("data/test_mcts_random_{}.json", res)
