from bachelorarbeit.players.mcts import MCTSPlayer
from bachelorarbeit.tools import evaluate_against_random, dump_json

if __name__ == "__main__":

    results = evaluate_against_random(
        player=MCTSPlayer,
        num_games=100,
        num_processes=6
    )
    print(results)

    dump_json("data/vs_random_{}.json", results)
