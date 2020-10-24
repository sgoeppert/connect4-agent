from bachelorarbeit.games import ConnectFour
from bachelorarbeit.players.adaptive_rave import AdaptiveRavePlayer, AdaptiveRaveEvaluator
from bachelorarbeit.players.network_player import NNEvaluator
from bachelorarbeit.tools import denormalize, transform_board_cnn

import config


class AdaptiveRaveNNEvaluator(AdaptiveRaveEvaluator, NNEvaluator):
    def __init__(self,
                 model_path,
                 transform_input,
                 transform_output=None,
                 alpha=0.9,
                 forgetting=False,
                 keep_replies=False,
                 ):
        AdaptiveRaveEvaluator.__init__(self, forgetting, keep_replies)
        NNEvaluator.__init__(self, model_path, transform_input, transform_output, alpha)

    def __call__(self, game_state: ConnectFour):
        playout_reward = AdaptiveRaveEvaluator.__call__(self, game_state)
        pred = self._get_prediction(game_state)

        weight = self.alpha

        score = weight * pred + (1 - weight) * playout_reward
        return score


class AdaptiveRaveNetworkPlayer(AdaptiveRavePlayer):
    name = "AdaptiveRaveNetworkPlayer"

    def __init__(self,
                 forgetting=False, keep_replies=False, exploration_constant=0.25,
                 network_weight: float = 0.5,
                 model_path: str = config.DEFAULT_MODEL,
                 transform_func: callable = transform_board_cnn,
                 transform_output: callable = denormalize,
                 **kwargs):
        super(AdaptiveRaveNetworkPlayer, self).__init__(forgetting, keep_replies, exploration_constant, **kwargs)
        self.evaluate = AdaptiveRaveNNEvaluator(model_path=model_path,
                                                transform_input=transform_func,
                                                transform_output=transform_output,
                                                alpha=network_weight,
                                                forgetting=forgetting,
                                                keep_replies=keep_replies)
if __name__ == "__main__":
    from bachelorarbeit.games import Observation, Configuration, ConnectFour
    from bachelorarbeit.tools import timer, transform_board_cnn
    from bachelorarbeit.players.mcts import MCTSPlayer
    from bachelorarbeit.players.adaptive_rave import AdaptiveRavePlayer
    from bachelorarbeit.selfplay import Arena
    import config

    # steps = 400
    #
    # play = AdaptiveRaveNetworkPlayer(max_steps=steps)
    # g = ConnectFour()
    # obs = Observation(board=g.board[:], mark=g.mark)
    # conf = Configuration()
    #
    # with timer():
    #     m = play.get_move(obs, conf)
    #     print(m)

    # exit()

    rave_steps = 52
    regular_steps = 1000

    arena = Arena(players=(AdaptiveRaveNetworkPlayer, MCTSPlayer),
                  constructor_args=(
                      {
                          "max_steps": rave_steps,
                          "exploration_constant": 0.4,
                          "network_weight": 0.5,
                          "alpha": None,
                          "keep_replies": True
                      },
                      # {"max_steps": regular_steps, "exploration_constant": 0.4, "alpha": 0.5, "keep_replies": True}),
                      {}),
                  num_games=500,
                  num_processes=8
                  )
    arena.run_game_mp(show_progress_bar=True)
