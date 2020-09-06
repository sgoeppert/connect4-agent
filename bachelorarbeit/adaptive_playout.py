from typing import List, Dict, Tuple
import math
import numpy as np
import random

from bachelorarbeit.games import Observation, Configuration, ConnectFour
from bachelorarbeit.mcts import Node, MCTSPlayer


class AdaptivePlayoutPlayer(MCTSPlayer):
    name = "AdaptivePlayoutPlayer"

    def __init__(self, forgetting=False, keep_replies=False, *args, **kwargs):
        super(AdaptivePlayoutPlayer, self).__init__(*args, **kwargs)
        self.replies = {}
        self.forgetting = forgetting
        self.keep_replies = keep_replies

    def reset(self, conf: Configuration = None):
        super(AdaptivePlayoutPlayer, self).reset(conf)
        if not self.keep_replies:
            self.replies = {}

    def evaluate_game_state(self, game_state: ConnectFour) -> float:
        game = game_state.copy()
        simulating_player = game.get_current_player()
        scoring = 3 - simulating_player

        last_move = None
        simulation_replies = {}

        while not game.is_terminal():
            move = None
            legal_moves = game.list_moves()
            if last_move is not None and last_move in self.replies:
                reply = self.replies[last_move]
                reply_move = (reply // 10) % game.cols
                mname = game.get_move_name(reply_move, played=False)
                if reply == mname:
                    move = reply_move

            if move not in legal_moves:
                move = random.choice(game.list_moves())

            game.play_move(move)

            move_name = game.get_move_name(move, played=True)
            if last_move is not None:
                simulation_replies[last_move] = move_name

            last_move = move_name

        winner = game.winner
        if winner is not None:
            loser = 3 - winner
            for move, reply in simulation_replies.items():
                replying_to = move % 10
                if replying_to == loser:
                    self.replies[move] = reply
                elif self.forgetting and replying_to == winner:
                    self.replies.pop(move, None)

        return game.get_reward(scoring)


if __name__ == "__main__":
    from bachelorarbeit.games import Observation, Configuration, ConnectFour
    from bachelorarbeit.tools import timer

    steps = 1000
    pl = AdaptivePlayoutPlayer(max_steps=steps, exploration_constant=0.85)
    conf = Configuration()
    game = ConnectFour(columns=conf.columns, rows=conf.rows, inarow=conf.inarow, mark=1)
    obs = Observation(board=game.board.copy(), mark=game.mark)

    with timer(f"{steps} steps"):
        m = pl.get_move(obs, conf)
        print(m)

    # print(pl.replies)