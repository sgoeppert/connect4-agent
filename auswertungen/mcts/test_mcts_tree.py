from bachelorarbeit.players.mcts import MCTSPlayer
from bachelorarbeit.tools import run_selfplay_experiment, dump_json, explore_parameter_against_fixed_opponent

TITLE = "MCTS Keep tree"
NUM_GAMES = 200
REPEATS = 10
NUM_PROCESSES = 10
DATA_DIR = "data/"
RAW_DIR = "raw/"
FNAME = "test_tree_{}.json"

p1, p2 = (True, False)

exploration_constant = 1.0
MAX_STEPS = 1000


if __name__ == "__main__":

    raw_res = run_selfplay_experiment(TITLE + " {} vs {}".format(p1, p2), (MCTSPlayer, MCTSPlayer),
                                  ({"exploration_constant": exploration_constant, "keep_tree": p1, "max_steps": MAX_STEPS},
                                   {"exploration_constant": exploration_constant, "keep_tree": p2, "max_steps": MAX_STEPS}),
                                  num_games=NUM_GAMES,
                                  num_processes=NUM_PROCESSES,
                                  repeats=REPEATS,
                                  show_progress_bar=True)
    shortened = {**raw_res}
    shortened.pop("raw_results")
    shortened.pop("raw_means")

    dump_json(DATA_DIR + FNAME, shortened)
    dump_json(RAW_DIR + FNAME, raw_res)
