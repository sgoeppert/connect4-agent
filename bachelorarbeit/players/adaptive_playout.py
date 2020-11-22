import random

from bachelorarbeit.games import Configuration, ConnectFour
from bachelorarbeit.players.mcts import MCTSPlayer, Evaluator


class AdaptiveEvaluator(Evaluator):
    def __init__(self, forgetting=False, keep_replies=False):
        self.replies = {}
        self.forgetting = forgetting
        self.keep_replies = keep_replies

    def get_next_move(self, game: "ConnectFour", last_move: int) -> int:
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

        return move

    def memorize(self, replies, winner: int):
        if winner is not None:
            loser = 3 - winner
            for move, reply in replies.items():
                replying_to = move % 10
                if replying_to == loser:
                    self.replies[move] = reply
                elif self.forgetting and replying_to == winner:
                    self.replies.pop(move, None)

    def __call__(self, game_state: ConnectFour) -> float:
        game = game_state.copy()
        simulating_player = game.get_current_player()
        scoring = 3 - simulating_player

        last_move = None
        simulation_replies = {}

        while not game.is_terminal():
            move = self.get_next_move(game, last_move)
            game.play_move(move)

            move_name = game.get_move_name(move, played=True)
            if last_move is not None:
                simulation_replies[last_move] = move_name

            last_move = move_name

        self.memorize(simulation_replies, game.winner)
        return game.get_reward(scoring)

    def reset(self):
        if not self.keep_replies:
            self.replies = {}


class AdaptivePlayoutPlayer(MCTSPlayer):
    name = "AdaptivePlayoutPlayer"

    def __init__(self, forgetting=False, keep_replies=False, *args, **kwargs):
        super(AdaptivePlayoutPlayer, self).__init__(*args, **kwargs)
        self.evaluate = AdaptiveEvaluator(forgetting, keep_replies)
