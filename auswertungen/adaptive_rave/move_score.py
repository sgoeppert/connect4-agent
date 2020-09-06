from bachelorarbeit.players.adaptive_rave import AdaptiveRavePlayer
from bachelorarbeit.tools import run_move_evaluation_experiment, dump_json

NUM_PROCESSES = 7

if __name__ == "__main__":
    results = []
    test_steps = [20, 50, 100, 200, 400, 800]
    for steps in test_steps:
        print(f"Adaptive Rave {steps}")
        res = run_move_evaluation_experiment(
            title="Adaptive RAVE player",
            player=AdaptiveRavePlayer,
            player_config={"max_steps": steps},
            num_processes=NUM_PROCESSES,
            repeats=2,
            show_progress_bar=True
        )
        results.append(res)
        print(res)
    dump_json("data/move_score_baseline_{}.json", results)
