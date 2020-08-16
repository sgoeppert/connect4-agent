from bachelorarbeit.base_players import RandomPlayer
from bachelorarbeit.sarsa_rave import SarsaPlayer
from bachelorarbeit.tools import run_selfplay_experiment, dump_json

TITLE = "Sarsa vs Random"
NUM_GAMES = 100
NUM_PROCESSES = 8

if __name__ == "__main__":
    res = run_selfplay_experiment(TITLE, (SarsaPlayer, RandomPlayer), num_games=NUM_GAMES)

    dump_json("data/test_sarsa_random_{}.json", res)
