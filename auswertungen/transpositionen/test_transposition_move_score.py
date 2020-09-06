from bachelorarbeit.players.transposition import TranspositionPlayer
from bachelorarbeit.tools import run_move_evaluation_experiment, dump_json

NUM_PROCESSES = 8

if __name__ == "__main__":
    results = []
    test_steps = [20, 50, 100, 200, 400, 800]
    for steps in test_steps:
        print(f"Transposition {steps}")
        res = run_move_evaluation_experiment(
            title="Transposition player",
            player=TranspositionPlayer,
            player_config={"max_steps": steps, "uct_method": "UCT"},
            num_processes=NUM_PROCESSES,
            repeats=5
        )
        results.append(res)
        print(res)
    dump_json("data/test_transposition_move_score_baseline_{}.json", results)
