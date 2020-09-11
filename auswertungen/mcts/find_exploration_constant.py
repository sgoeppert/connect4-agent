from bachelorarbeit.players.mcts import MCTSPlayer
from bachelorarbeit.tools import run_selfplay_experiment, dump_json, explore_parameter_against_fixed_opponent

TITLE = "MCTS exploration"
NUM_GAMES = 100
REPEATS = 10
RUN_NUMBER = 2
DATA_DIR = "data/"
RAW_DIR = "raw/"
FNAME = "find_exploration_" + str(RUN_NUMBER) + "_{}.json"

MAX_STEPS = 1000
NUM_PROCESSES = 7

explo_vals = [0.8, 0.9, 1.1, 1.2]
compare_against = 1.0

if __name__ == "__main__":

    experiments, raw_game_results = explore_parameter_against_fixed_opponent(
        player=MCTSPlayer,
        opponent=MCTSPlayer,
        player_conf={"max_steps": MAX_STEPS},
        opponent_conf={"max_steps": MAX_STEPS, "exploration_constant": compare_against},
        values=explo_vals, parameter="exploration_constant",
        num_games=NUM_GAMES, repeats=REPEATS
    )
    #
    # experiments = []
    # raw_game_results = []
    # for p1 in p1_values:
    #     for p2 in p2_values:
    #         if p1 == p2:  # skip mirror matches
    #             continue
    #
    #         res = run_selfplay_experiment(
    #             TITLE + " {} vs {}".format(p1, p2),
    #             (MCTSPlayer, MCTSPlayer),
    #             ({"exploration_constant": p1, "max_steps": MAX_STEPS},
    #              {"exploration_constant": p2, "max_steps": MAX_STEPS}),
    #             num_games=NUM_GAMES,
    #             repeats=REPEATS,
    #             num_processes=NUM_PROCESSES,
    #             show_progress_bar=True
    #         )
    #         experiments.append({**res, "raw_results": []})
    #         raw_game_results.append(res)

    dump_json(DATA_DIR + FNAME, experiments)
    dump_json(RAW_DIR + FNAME, raw_game_results)
