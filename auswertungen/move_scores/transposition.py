from bachelorarbeit.players.transposition import TranspositionPlayer
from bachelorarbeit.tools import run_move_evaluation_experiment, dump_json

NUM_PROCESSES = 6

if __name__ == "__main__":
    player_config = {
      "exploration_constant": 0.9,
      "uct_method": "UCT2",
      "with_symmetry": True
    }

    results = []
    test_steps = [20, 50, 100, 200, 400, 800]
    # test_steps = [20, 50]
    for steps in test_steps:
        print(f"Transposition {steps}")
        res = run_move_evaluation_experiment(
            title="Transposition player",
            player=TranspositionPlayer,
            player_config={**player_config, "max_steps": steps},
            num_processes=NUM_PROCESSES,
            repeats=10,
            show_progress_bar=True
        )
        results.append(res)
        print(res)
    dump_json("data/transposition_{}.json", results)
