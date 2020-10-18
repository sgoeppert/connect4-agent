from bachelorarbeit.players.adaptive_rave import AdaptiveRavePlayer
from bachelorarbeit.players.adaptive_rave_network_player import AdaptiveRaveNetworkPlayer
from bachelorarbeit.tools import run_move_evaluation_experiment, dump_json

NUM_PROCESSES = 8

if __name__ == "__main__":

    player_config = {
        "exploration_constant": 0.3,
        "k": 10,
        "alpha": None
    }

    results = []
    test_steps = [20, 50, 100, 200, 400]
    # test_steps = [20, 50]
    for steps in test_steps:
        print(f"NN-LGR-RAVE {steps}")
        res = run_move_evaluation_experiment(
            title="NN-LGR-RAVE player",
            player=AdaptiveRaveNetworkPlayer,
            player_config={**player_config, "max_steps": steps},
            num_processes=NUM_PROCESSES,
            max_tasks=10,
            repeats=5,
            show_progress_bar=True
        )
        results.append(res)
        print(res)
    dump_json("data/nn_rave_lgr_{}.json", results)
