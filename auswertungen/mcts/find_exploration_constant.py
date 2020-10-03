from bachelorarbeit.players.mcts import MCTSPlayer
from bachelorarbeit.tools import get_range, dump_json, explore_parameter_against_fixed_opponent

TITLE = "MCTS exploration"
NUM_GAMES = 200
REPEATS = 5
RUN_NUMBER = 1
DATA_DIR = "data/"
RAW_DIR = "raw/"
FNAME = "find_exploration_" + str(RUN_NUMBER) + "_{}.json"

MAX_STEPS = 1000
NUM_PROCESSES = 10

explo_vals = [0.25, 0.5, 0.8, 1.0, 1.2, 1.5]
compare_against = 1.0

if __name__ == "__main__":

    # repeats = 2
    # experiments, raw_game_results = explore_parameter_against_fixed_opponent(
    #     player=MCTSPlayer,
    #     opponent=MCTSPlayer,
    #     player_conf={"max_steps": MAX_STEPS},
    #     opponent_conf={"max_steps": MAX_STEPS, "exploration_constant": compare_against},
    #     values=explo_vals, parameter="exploration_constant",
    #     num_games=NUM_GAMES, repeats=repeats
    # )
    #
    # dump_json(DATA_DIR + FNAME, experiments)
    # dump_json(RAW_DIR + FNAME, raw_game_results)

    RUN_NUMBER = 3
    FNAME = "find_exploration_" + str(RUN_NUMBER) + "_{}.json"

    # best = 0.0
    # std = 0.0
    # config = None
    # for experiment in experiments:
    #     mean = experiment["mean"][0]
    #     if mean > best:
    #         best = mean
    #         std = experiment["std"]
    #         config = experiment["configurations"][0]
    #
    # best_explo = config["exploration_constant"]
    # print(f"Best mean score was {config} with mean {best} (Std. {std})")

    new_values = get_range(0.8, 5)

    experiments, raw_game_results = explore_parameter_against_fixed_opponent(
        player=MCTSPlayer,
        opponent=MCTSPlayer,
        player_conf={"max_steps": MAX_STEPS},
        opponent_conf={"max_steps": MAX_STEPS, "exploration_constant": compare_against},
        values=new_values, parameter="exploration_constant",
        num_games=NUM_GAMES, repeats=REPEATS
    )

    dump_json(DATA_DIR + FNAME, experiments)
    dump_json(RAW_DIR + FNAME, raw_game_results)
