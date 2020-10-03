from bachelorarbeit.games import ConnectFour
from bachelorarbeit.players.adaptive_rave import AdaptiveRavePlayer, AdaptiveRaveEvaluator
from bachelorarbeit.players.network_player import NNEvaluator
from bachelorarbeit.tools import denormalize, transform_board_cnn

import config


class AdaptiveRaveNNEvaluator(AdaptiveRaveEvaluator, NNEvaluator):
    def __init__(self,
                 model_path,
                 transform_input,
                 move_list: list,
                 transform_output=None,
                 alpha=0.9,
                 forgetting=False,
                 keep_replies=False,
                 ):
        AdaptiveRaveEvaluator.__init__(self, move_list, forgetting, keep_replies)
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
                                                move_list=self.move_list,
                                                alpha=network_weight,
                                                forgetting=forgetting,
                                                keep_replies=keep_replies)
if __name__ == "__main__":
    from bachelorarbeit.games import Observation, Configuration, ConnectFour
    from bachelorarbeit.tools import timer, transform_board_cnn
    import config

    pl = AdaptiveRaveNetworkPlayer(k=10, max_steps=200)
    # pl = AdaptiveRavePlayer(max_steps=2200)
    g = ConnectFour()
    con = Configuration()
    obs = Observation(board=g.board, mark=g.mark)
    with timer():
        pl.get_move(obs, con)

