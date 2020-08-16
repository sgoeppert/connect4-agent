from bachelorarbeit.scorebounded import ScoreboundedPlayer
from bachelorarbeit.tools import run_move_evaluation_experiment, dump_json

NUM_PROCESSES = 6

if __name__ == "__main__":
    results = []
    test_steps = [20, 50, 100, 200, 400, 800]
    for steps in test_steps:
        print(f"Scorebounded {steps}")
        res = run_move_evaluation_experiment(
            title="Scorebounded player",
            player=ScoreboundedPlayer,
            player_config={"max_steps": steps},
            num_processes=NUM_PROCESSES,
            repeats=2
        )
        results.append(res)
        print(res)
    dump_json("data/test_scorebound_move_score_{}.json", results)
