from bachelorarbeit.players.scorebound_rave import ScoreBoundedRavePlayer
from bachelorarbeit.tools import run_move_evaluation_experiment, dump_json

NUM_PROCESSES = 8
REPEATS = 2

test_name = "sb_opt"
settings = {
    "baseline": {},
    "sb_opt": {"delta": 0.2},
    "alpha099": {"alpha": 0.99},
    "alpha09": {"alpha": 0.9},
    "alpha05": {"alpha": 0.5},
    "alpha0": {"alpha": 0.0},
    "b0001": {"b": 0.001},
    "b01": {"b": 0.1},
}

if __name__ == "__main__":
    results = []
    test_steps = [20, 50, 100, 200, 400, 800]

    for steps in test_steps:
        base_sett = {"exploration_constant": 0.2, "max_steps": steps}
        experiment_setting = {**base_sett, **settings[test_name]}

        print(f"Rave {steps}")
        res = run_move_evaluation_experiment(
            title="Score Bounded Rave player",
            player=ScoreBoundedRavePlayer,
            player_config=experiment_setting,
            num_processes=NUM_PROCESSES,
            repeats=REPEATS,
            show_progress_bar=True
        )
        results.append(res)
        print(res)
    dump_json("data/test_rave_move_score_"+test_name+"_{}.json", results)
