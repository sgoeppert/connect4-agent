from bachelorarbeit.players.scorebounded import ScoreboundedPlayer
from bachelorarbeit.players.mcts import MCTSPlayer
from bachelorarbeit.tools import run_selfplay_experiment, dump_json

TITLE = "Scorebounded vs MCTS"
NUM_GAMES = 100
NUM_PROCESSES = 6

deltas = [-0.2, -0.1, -0.01, 0.0, 0.01, 0.1, 0.2]
gammas = [-0.2, -0.1, -0.01, 0.0, 0.01, 0.1, 0.2]

steps = 200

if __name__ == "__main__":
    results = []
    for d in deltas:
        for g in gammas:
            res = run_selfplay_experiment(
                TITLE + f" delta {d} gamma {g} steps {steps}",
                players=(ScoreboundedPlayer, MCTSPlayer),
                constructor_args=(
                    {"cut_delta": d, "cut_gamma": g, "max_steps": steps},
                    {"max_steps": steps}
                ),
                num_games=NUM_GAMES,
                num_processes=NUM_PROCESSES)
            results.append(res)

    dump_json("data/test_scorebounded_guidance_{}.json", results)
