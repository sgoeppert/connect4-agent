import random
import numpy as np

from bachelorarbeit.games import ConnectFour
from bachelorarbeit.players.mcts import MCTSPlayer, Evaluator


class NNEvaluator(Evaluator):
    def __init__(self, model_path, transform_input, transform_output=None):
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        import tensorflow as tf
        from tensorflow import keras
        physical_devices = tf.config.list_physical_devices('GPU')
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        except:
            # Invalid device or cannot modify virtual devices once initialized.
            pass

        self.model = keras.models.load_model(model_path, compile=False)
        self.model(np.array([transform_input([0]*42)]))
        self.transform_input_func = transform_input
        self.transform_output_func = transform_output

    def __call__(self, game_state: ConnectFour):
        playout_reward = super().__call__(game_state)
        # nn_prediction = self.model.predict(np.array([game_state.board]))
        # b = np.array(game_state.board).reshape((game_state.rows, game_state.cols))
        # flip_b = np.fliplr(b)
        # transformed = np.array([self.transform_input(b.tolist()), self.transform_input(flip_b.tolist())])
        transformed = np.array([self.transform_input_func(game_state.board)])
        nn_prediction = self.model(transformed, training=False)
        if self.transform_output_func is not None:
            nn_prediction = self.transform_output_func(nn_prediction)

        pred = np.mean(nn_prediction)

        def _weight(x):
            return (-3*x**2+124*x+160)/1600

        stones_played = sum(game_state.stones_per_column)
        # weight = 0.1 + (stones_played / 41)
        weight = _weight(stones_played)

        score = weight * pred + (1-weight) * playout_reward
        # print("Eval for board", np.array(game_state.board).reshape((game_state.rows, game_state.cols)), sep="\n")
        # # print("Transformed", transformed)
        # print("Current player", game_state.get_current_player())
        # print("Pred, playout, weight, stones", pred, playout_reward, weight, stones_played)
        # print("Calculated reward", score)

        return score


class NetworkPlayer(MCTSPlayer):
    name = "NetworkPlayer"

    def __init__(
            self,
            model_path: str,
            transform_func: callable,
            transform_output: callable = None,
            **kwargs
    ):
        super(NetworkPlayer, self).__init__(**kwargs)
        self.evaluate = NNEvaluator(model_path=model_path, transform_input=transform_func, transform_output=transform_output)


if __name__ == "__main__":
    from bachelorarbeit.games import Observation, Configuration, ConnectFour
    from bachelorarbeit.tools import timer, transform_board_cnn
    import config

    steps = 100

    # from tensorflow import keras
    model_path = config.ROOT_DIR + "/best_models/cnn_bonus_channel_aug"

    play = NetworkPlayer(model_path=model_path, transform_func=transform_board_cnn, max_steps=300)
    g = ConnectFour()
    obs = Observation(board=g.board[:], mark=g.mark)
    conf = Configuration()

    with timer():
        m = play.get_move(obs, conf)
        print(m)


    #
    # model = keras.models.load_model(config.ROOT_DIR + "/best_models/cnn_bonus_channel_aug")
    #
    # g = ConnectFour()
    # g.play_move(3)
    #
    # with timer("Setup"):
    #     evaluate = NNEvaluator(model_path=model_path, transform_input=transform_board_cnn)
    # print("Setup done")
    # with timer("Eval"):
    #     evaluate(g)

    # b = np.array([transform_board_cnn(g.board)])
    # repeats = 100
    # model.predict(np.zeros_like(b))
    # model(np.zeros_like(b),training=False)
    #
    # with timer(f"model.__call__"):
    #     for i in range(repeats):
    #         print(model(b, training=False))
    #
    # with timer(f"model.predict"):
    #     for i in range(repeats):
    #         print(model.predict(b))


    # p = NetworkPlayer(max_steps=steps, predictor=predictor)
    # conf = Configuration()
    # game = ConnectFour(columns=conf.columns, rows=conf.rows, inarow=conf.inarow, mark=1)
    # obs = Observation(board=game.board.copy(), mark=game.mark)
    #
    # with timer(f"{steps} steps"):
    #     m = p.get_move(obs, conf)
    # print(m)