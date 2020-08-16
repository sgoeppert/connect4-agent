from bachelorarbeit.rave import RavePlayer
from bachelorarbeit.tools import run_move_evaluation_experiment, dump_json

NUM_PROCESSES = 8

if __name__ == "__main__":
    results = []
    test_steps = [20, 50, 100, 200, 400, 800]
    for steps in test_steps:
        print(f"Rave {steps}")
        res = run_move_evaluation_experiment(
            title="Rave player",
            player=RavePlayer,
            player_config={"max_steps": steps, "beta": 0.95},
            num_processes=NUM_PROCESSES,
            repeats=2
        )
        results.append(res)
        print(res)
    dump_json("data/test_rave_move_score_baseline_{}.json", results)
