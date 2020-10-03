import numpy as np

from bachelorarbeit.games import ConnectFour
from bachelorarbeit.players.mcts import MCTSPlayer, Evaluator
from bachelorarbeit.tools import flip_board, transform_board_cnn, denormalize
from requests import Session
import config

DEBUG = False


class RequestEvaluator(Evaluator):
    def __init__(self, alpha=0.9):
        self.session = Session()
        self.alpha = alpha

    def __call__(self, game_state: ConnectFour):
        payload = {"input": game_state.board[:]}
        resp = self.session.post('http://127.0.0.1:5000/predict', json=payload)
        pred = resp.json()["predictions"]
        playout_reward = super().__call__(game_state)
        score = self.alpha * pred + (1 - self.alpha) * playout_reward

        return score


class NNEvaluator(Evaluator):
    def __init__(self, model_path, transform_input, transform_output=None, alpha=0.9):
        # lazy load tensorflow so it is imported separately in each running process with memory growth enabled
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
        self.model(np.array(transform_input([0] * 42)),
                   training=False)  # run one prediction to initialize the network graph
        self.transform_input_func = transform_input
        self.transform_output_func = transform_output
        self.alpha = alpha

    def _get_prediction(self, game_state: ConnectFour):
        boards = [game_state.board, flip_board(game_state.board)]
        transformed = np.array(self.transform_input_func(boards))
        nn_prediction = self.model(transformed, training=False)

        if self.transform_output_func is not None:
            nn_prediction = self.transform_output_func(nn_prediction)

        pred = np.mean(nn_prediction)
        return pred

    def __call__(self, game_state: ConnectFour):
        playout_reward = super().__call__(game_state)
        pred = self._get_prediction(game_state)

        weight = self.alpha

        score = weight * pred + (1 - weight) * playout_reward
        if DEBUG:
            print("Eval for board", np.array(game_state.board).reshape((game_state.rows, game_state.cols)), sep="\n")
            print("Current player", game_state.get_current_player())
            print("Pred, playout, weight", pred, playout_reward, weight)
            print("Calculated reward", score)

        return score


class NetworkPlayer(MCTSPlayer):
    name = "NetworkPlayer"

    def __init__(
            self,
            network_weight: float = 0.5,
            model_path: str = config.DEFAULT_MODEL,
            transform_func: callable = transform_board_cnn,
            transform_output: callable = denormalize,
            use_server: bool = False,
            **kwargs
    ):
        super(NetworkPlayer, self).__init__(**kwargs)

        if use_server:
            self.evaluate = RequestEvaluator(alpha=network_weight)
        else:
            self.evaluate = NNEvaluator(model_path=model_path,
                                        transform_input=transform_func,
                                        transform_output=transform_output,
                                        alpha=network_weight)



if __name__ == "__main__":
    from bachelorarbeit.games import Observation, Configuration, ConnectFour
    from bachelorarbeit.tools import timer, transform_board_cnn
    import config

    steps = 100

    # from tensorflow import keras
    model_path = config.ROOT_DIR + "/best_models/400000/padded_cnn_norm"

    play = NetworkPlayer(model_path=model_path, transform_func=transform_board_cnn, max_steps=steps)
    g = ConnectFour()
    obs = Observation(board=g.board[:], mark=g.mark)
    conf = Configuration()

    with timer():
        m = play.get_move(obs, conf)
        print(m)
