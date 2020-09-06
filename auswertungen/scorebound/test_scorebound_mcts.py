from bachelorarbeit.players.scorebounded import ScoreboundedPlayer
from bachelorarbeit.players.mcts import MCTSPlayer
from bachelorarbeit.tools import run_selfplay_experiment, dump_json

TITLE = "Scorebounded vs MCTS"
NUM_GAMES = 200
NUM_PROCESSES = 6

if __name__ == "__main__":
    res = run_selfplay_experiment(TITLE,
                                  players=(ScoreboundedPlayer, MCTSPlayer),
                                  num_games=NUM_GAMES,
                                  num_processes=NUM_PROCESSES)

    dump_json("data/test_scorebounded_mcts_{}.json", res)
