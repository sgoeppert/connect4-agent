from bachelorarbeit.players.base_players import RandomPlayer
from bachelorarbeit.players.scorebounded import ScoreboundedPlayer
from bachelorarbeit.tools import run_selfplay_experiment, dump_json

TITLE = "Scorebounded vs Random"
NUM_GAMES = 100
NUM_PROCESSES = 6

if __name__ == "__main__":
    res = run_selfplay_experiment(TITLE, (ScoreboundedPlayer, RandomPlayer), num_games=NUM_GAMES)

    dump_json("data/test_scorebounded_random_{}.json", res)
