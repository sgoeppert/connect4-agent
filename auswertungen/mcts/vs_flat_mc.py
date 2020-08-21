from bachelorarbeit.mcts import MCTSPlayer
from bachelorarbeit.tools import evaluate_against_flat_monte_carlo, dump_json

if __name__ == "__main__":

    results = evaluate_against_flat_monte_carlo(
        player=MCTSPlayer,
        num_games=100,
        num_processes=6,
        opponent_steps=1000
    )
    print(results)

    dump_json("data/vs_flat_mc_{}.json", results)
