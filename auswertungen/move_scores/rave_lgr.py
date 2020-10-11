from bachelorarbeit.players.adaptive_rave import AdaptiveRavePlayer
from bachelorarbeit.players.mcts import MCTSPlayer
from bachelorarbeit.players.base_players import RandomPlayer
from bachelorarbeit.players.rave import RavePlayer
from bachelorarbeit.tools import run_move_evaluation_experiment, dump_json
import gc

NUM_PROCESSES = 6

if __name__ == "__main__":

    player_config = {
      "exploration_constant": 0.2,
      "k": 100,
      "alpha": None,
      "keep_replies": True,
      "forgetting": False
    }

    results = []
    test_steps = [20, 50, 100, 200, 400, 800]
    # test_steps = [20, 50]
    for steps in test_steps:
        print(f"Rave {steps}")
        res = run_move_evaluation_experiment(
            title="LGR-RAVE player",
            player=AdaptiveRavePlayer,
            player_config={**player_config, "max_steps": steps},
            num_processes=NUM_PROCESSES,
            repeats=10,
            show_progress_bar=True
        )
        results.append(res)
        print(res)
    dump_json("data/rave_lgr_{}.json", results)
