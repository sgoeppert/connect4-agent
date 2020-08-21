from bachelorarbeit.base_players import FlatMonteCarlo
from bachelorarbeit.tools import evaluate_against_flat_monte_carlo, dump_json

import math

base_settings = {"max_steps": 1000}

exploration_values = [0.1, 0.25, 0.5, 1/math.sqrt(2), 1.0, 1.2, math.sqrt(2), 2.0]
# exploration_values = [1/math.sqrt(2), 1.0, 1.2, math.sqrt(2)]

if __name__ == "__main__":

    all_results = []
    for exp in exploration_values:
        settings = {**base_settings, "exploration_constant": exp, "ucb_selection": True}

        results = evaluate_against_flat_monte_carlo(
            player=FlatMonteCarlo,
            constructor_args=settings,
            num_games=300,
            num_processes=6,
            opponent_steps=1000
        )
        print(results)
        all_results.append(results)

    dump_json("data/flat_mc_exploration_{}.json", all_results)
