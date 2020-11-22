from bachelorarbeit.players.network_player import NetworkPlayer
from bachelorarbeit.tools import run_move_evaluation_experiment, dump_json
import gc

NUM_PROCESSES = 7

if __name__ == "__main__":

    player_config = {
      "exploration_constant": 0.8,
    }

    results = []
    test_steps = [20, 50, 100, 200, 400]
    # test_steps = [20, 50]
    for steps in test_steps:
        print(f"NN {steps}")
        res = run_move_evaluation_experiment(
            title="Network player",
            player=NetworkPlayer,
            player_config={**player_config, "max_steps": steps},
            num_processes=NUM_PROCESSES,
            max_tasks=30,
            repeats=3,
            show_progress_bar=True
        )
        results.append(res)
        print(res)
    dump_json("data/nn_{}.json", results)
