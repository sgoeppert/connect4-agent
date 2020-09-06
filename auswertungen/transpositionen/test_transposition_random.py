from bachelorarbeit.players.base_players import RandomPlayer
from bachelorarbeit.players.transposition import TranspositionPlayer
from bachelorarbeit.tools import run_selfplay_experiment, dump_json

TITLE = "Transposition vs Random"
NUM_GAMES = 100
NUM_PROCESSES = 8

if __name__ == "__main__":
    res = run_selfplay_experiment(TITLE, (TranspositionPlayer, RandomPlayer), num_games=NUM_GAMES)

    dump_json("data/test_transposition_random_{}.json", res)
