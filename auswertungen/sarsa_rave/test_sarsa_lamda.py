from bachelorarbeit.sarsa_rave import SarsaPlayer
from bachelorarbeit.tools import run_move_evaluation_experiment, dump_json

NUM_PROCESSES = 10

steps = 800

if __name__ == "__main__":
    results = []
    lamda = [0.95, 0.97, 0.99, 1.0]
    cp = [0.25, 0.5, 0.75, 1.0]
    for c in cp:
        for ld in lamda:
            print(f"Sarsa Lambda lambda {ld} cp {c}")
            res = run_move_evaluation_experiment(
                title="Sarsa player lambda {} cp {}".format(ld, c),
                player=SarsaPlayer,
                player_config={"max_steps": steps, "lamda": ld, "exploration_constant": c, "beta": 0.9},
                num_processes=NUM_PROCESSES,
                repeats=1
            )
            results.append(res)
            print(res)
    dump_json("data/test_sarsa_lamda_cp_{}.json", results)
