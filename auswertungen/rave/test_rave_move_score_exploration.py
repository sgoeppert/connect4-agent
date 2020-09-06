from bachelorarbeit.players.rave import RavePlayer
from bachelorarbeit.tools import run_move_evaluation_experiment, dump_json

NUM_PROCESSES = 8

if __name__ == "__main__":
    results = []
    steps = 200
    exploration = [0.3, 0.5, 1.0]
    for exp in exploration:
        print(f"Rave exp: {exp} beta {0.95}")
        res = run_move_evaluation_experiment(
            title="Rave player",
            player=RavePlayer,
            player_config={"max_steps": steps, "beta": 0.0, "exploration_constant": exp},
            num_processes=NUM_PROCESSES,
            repeats=10
        )
        results.append(res)
        print(res)
    dump_json("data/test_rave_move_score_explo_{}.json", results)
