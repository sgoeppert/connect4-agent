"""
Erzeugt Daten für das Training der neuronalen Netze.
"""
from bachelorarbeit.selfplay import Arena, Memory
from bachelorarbeit.players.mcts import MCTSPlayer

if __name__ == "__main__":

    memory = Memory("selfplay_mcts_strong_v2.pickle")
    arena = Arena(players=(MCTSPlayer, MCTSPlayer),
                  num_games=500,
                  num_processes=10,
                  memory=memory)

    iterations = 50

    for i in range(iterations):
        arena.run_game_mp()
        memory.save_data()
        print(memory.num_states, "states in memory")
