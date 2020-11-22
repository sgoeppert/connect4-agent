from bachelorarbeit.games import ConnectFour
from bachelorarbeit.players.mcts import MCTSPlayer
from bachelorarbeit.players.network_player import NetworkPlayer, NNEvaluator
from bachelorarbeit.players.adaptive_playout import AdaptiveEvaluator
from bachelorarbeit.tools import transform_board_cnn, denormalize, transform_board_nega
import config


class AdaptiveNNEvaluator(AdaptiveEvaluator, NNEvaluator):
    def __init__(self,
                 model_path,
                 transform_input,
                 transform_output=None,
                 alpha=0.9,
                 forgetting=False,
                 keep_replies=False
                 ):
        AdaptiveEvaluator.__init__(self, forgetting, keep_replies)
        NNEvaluator.__init__(self, model_path, transform_input, transform_output, alpha)

    def __call__(self, game_state: ConnectFour):
        playout_reward = AdaptiveEvaluator.__call__(self, game_state)
        pred = self._get_prediction(game_state)

        weight = self.alpha

        score = weight * pred + (1 - weight) * playout_reward
        return score


class AdaptiveNetworkPlayer(NetworkPlayer):
    name = "AdaptiveNetworkPlayer"

    def __init__(
            self,
            network_weight: float = 0.5,
            model_path: str = config.DEFAULT_MODEL,
            transform_func: callable = transform_board_cnn,
            transform_output: callable = denormalize,
            keep_replies: bool = False,
            forgetting: bool = False,
            **kwargs
    ):
        super(NetworkPlayer, self).__init__(**kwargs)
        self.evaluate = AdaptiveNNEvaluator(model_path=model_path,
                                            transform_input=transform_func,
                                            transform_output=transform_output,
                                            alpha=network_weight,
                                            keep_replies=keep_replies,
                                            forgetting=forgetting)


if __name__ == "__main__":
    from bachelorarbeit.games import Observation, Configuration, ConnectFour
    from bachelorarbeit.tools import timer, transform_board_cnn
    from bachelorarbeit.tuner import TinyArena
    from bachelorarbeit.selfplay import Arena
    import numpy as np
    from tqdm import tqdm
    import config

    steps = 2000

    model_path = config.ROOT_DIR + "/best_models/400000/regular_norm"

    play = AdaptiveNetworkPlayer(max_steps=steps, model_path=model_path, transform_func=transform_board_nega)
    g = ConnectFour()
    obs = Observation(board=g.board[:], mark=g.mark)
    conf = Configuration()

    with timer():
        m = play.get_move(obs, conf)
        print(m)

    # arena = TinyArena(players=(AdaptiveNetworkPlayer, MCTSPlayer),
    #                   opponent_config={"max_steps": regular_steps, "exploration_constant": 1.0})
    # arena.player_config = {
    #     "max_steps": nn_steps,
    #     "network_weight": 0.5,
    #     "exploration_constant": 0.8,
    #     "keep_replies": True
    # }
    # nn_steps = 200
    # regular_steps = 2000
    #
    # arena = Arena(players=(AdaptiveNetworkPlayer, MCTSPlayer),
    #               constructor_args=(
    #                   {
    #                       "max_steps": nn_steps,
    #                       "network_weight": 0.5,
    #                       "exploration_constant": 0.8,
    #                       "keep_replies": True
    #                   },
    #                   {"max_steps": regular_steps, "exploration_constant": 1.0}),
    #               num_games=400,
    #               num_processes=8
    #               )
    # arena.run_game_mp(show_progress_bar=True)
    # all_rewards = []
    # games_per = 200
    # pbar = tqdm(total=games_per * 2)
    # for i in range(games_per):
    #     rewards = arena.run_game()
    #     all_rewards.append(rewards)
    #     m = (np.mean(all_rewards, axis=0) + 1) / 2
    #     pbar.set_postfix({"mean": m}, refresh=False)
    #     pbar.update()
    #
    # arena.flip_players = True
    # for i in range(games_per):
    #     rewards = arena.run_game()
    #     all_rewards.append(rewards[::-1])
    #
    #     m = (np.mean(all_rewards, axis=0) + 1) / 2
    #     pbar.set_postfix({"mean": m}, refresh=False)
    #     pbar.update()
    #
    # m = (np.mean(all_rewards, axis=0) + 1) / 2
    # print(m)
