from bachelorarbeit.games import Configuration, ConnectFour, Observation
from bachelorarbeit.players.mcts import MCTSPlayer
from bachelorarbeit.players.rave import RavePlayer
from bachelorarbeit.players.adaptive_playout import AdaptiveEvaluator


class AdaptiveRaveEvaluator(AdaptiveEvaluator):
    def __init__(self, forgetting=False, keep_replies=False):
        """
        :param forgetting:
        :param keep_replies:
        """
        super(AdaptiveRaveEvaluator, self).__init__(forgetting, keep_replies)
        self.moves = []

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
            self.moves.append(move_name)

            if last_move is not None:
                simulation_replies[last_move] = move_name

            last_move = move_name

        self.memorize(simulation_replies, game.winner)
        return game.get_reward(scoring)

    def reset(self):
        if not self.keep_replies:
            self.replies = {}


class AdaptiveRavePlayer(RavePlayer):
    name = "AdaptiveRavePlayer"

    def __init__(self, forgetting=False, keep_replies=False, exploration_constant=0.25, **kwargs):
        super(AdaptiveRavePlayer, self).__init__(exploration_constant=exploration_constant, **kwargs)
        self.evaluate = AdaptiveRaveEvaluator(forgetting, keep_replies)

    def reset(self, conf: Configuration = None):
        super(AdaptiveRavePlayer, self).reset(conf)
        self.evaluate.reset()
