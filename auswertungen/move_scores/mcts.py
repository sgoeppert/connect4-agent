from bachelorarbeit.players.mcts import MCTSPlayer
from bachelorarbeit.players.base_players import RandomPlayer
from bachelorarbeit.tools import run_move_evaluation_experiment, dump_json

NUM_PROCESSES = 6

if __name__ == "__main__":
    results = []
    print("Random")
    res = run_move_evaluation_experiment("Random", player=RandomPlayer, num_processes=NUM_PROCESSES, repeats=5)
    print(res)
    results.append(res)
    test_steps = [20, 50, 100, 200, 400, 800]
    # test_steps = [20, 50]
    for steps in test_steps:
        print(f"MCTS {steps}")
        res = run_move_evaluation_experiment(
            title="MCTS player",
            player=MCTSPlayer,
            player_config={"max_steps": steps},
            num_processes=NUM_PROCESSES,
            repeats=10,
            show_progress_bar=True
        )
        results.append(res)
        print(res)
    dump_json("data/mcts_{}.json", results)
