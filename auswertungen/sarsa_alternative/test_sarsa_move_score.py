from bachelorarbeit.sarsa_rave_alt import SarsaPlayer
from bachelorarbeit.tools import run_move_evaluation_experiment, dump_json

NUM_PROCESSES = 8

test_name = "beta09_lambda095"

settings = {
    "baseline": {},
    "beta09": {"beta": 0.9},
    "beta09_lambda095": {"beta": 0.9, "lamda": 0.95},
    "lamda095": {"lamda": 0.95},
}

if __name__ == "__main__":
    results = []
    test_steps = [20, 50, 100, 200, 400, 800]
    for steps in test_steps:
        base_sett = {"max_steps": steps}

        experiment_settings = {**base_sett, **settings[test_name]}

        print(f"Sarsa {steps}")
        res = run_move_evaluation_experiment(
            title="Sarsa player",
            player=SarsaPlayer,
            player_config=experiment_settings,
            num_processes=NUM_PROCESSES,
            repeats=2
        )
        results.append(res)
        print(res)
    dump_json("data/test_sarsa_move_score_"+test_name+"_{}.json", results)
