from kaggle_environments import evaluate
from mcts.regular import agent
import numpy as np
from tools import play_game
#
# res = evaluate("connectx", [p.play_game, "random"], num_episodes=10)
# res = np.array(res)
#
# print(res)
# print(np.mean(res, axis=0))


def p1(obs, conf):
    return agent(obs, conf, {"steps": 100, "cp": 1.0})


def p2(obs, conf):
    return agent(obs, conf, {"steps": 1000, "cp": 1.0})

results = []
for _ in range(20):
    results.append(play_game(p1, p2))

res = np.array(results)
print(np.mean(res, axis=0))
