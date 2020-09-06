from bachelorarbeit.players.base_players import FlatMonteCarlo
from bachelorarbeit.tools import evaluate_against_random, dump_json

import math

base_settings = {"max_steps": 800}

exploration_values = [0.1, 0.25, 0.5, 1/math.sqrt(2), 1.0, 1.2, math.sqrt(2), 2.0]

if __name__ == "__main__":

    all_results = []
    for exp in exploration_values:
        settings = {**base_settings, "exploration_constant": exp}

        results = evaluate_against_random(
            player=FlatMonteCarlo,
            constructor_args=settings,
            num_games=100,
            num_processes=6
        )
        print(results)
        all_results.append(results)

    dump_json("data/flat_mc_vs_random_{}.json", all_results)
