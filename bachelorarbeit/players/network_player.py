import random
import numpy as np

from bachelorarbeit.games import ConnectFour
from bachelorarbeit.players.mcts import MCTSPlayer


class Predictor:
    def evaluate_game_state(self, game_state: ConnectFour):
        game = game_state.copy()
        scoring = game.get_other_player(game.get_current_player())
        while not game.is_terminal():
            game.play_move(random.choice(game.list_moves()))

        return game.get_reward(scoring)


class NNPredictor(Predictor):
    def __init__(self, model, transform_input):
        self.model = model
        self.transform_input = transform_input

    def evaluate_game_state(self, game_state: ConnectFour):
        playout_reward = super(NNPredictor, self).evaluate_game_state(game_state)
        # nn_prediction = self.model.predict(np.array([game_state.board]))
        # b = np.array(game_state.board).reshape((game_state.rows, game_state.cols))
        # flip_b = np.fliplr(b)
        # transformed = np.array([self.transform_input(b.tolist()), self.transform_input(flip_b.tolist())])
        transformed = np.array([self.transform_input(game_state.board)])
        nn_prediction = self.model(transformed, training=False)
        pred = np.mean(nn_prediction)

        print("Eval for board", np.array(game_state.board).reshape((game_state.rows, game_state.cols)))
        print("Transformed", transformed)
        print("Current player", game_state.get_current_player())
        print("Pred, playout", pred, playout_reward)

        pred = -pred
        # if game_state.get_current_player() == 2:
        #     pred = -pred

        weight = 0.8

        score = weight * pred + (1-weight) * playout_reward


        return score


class NetworkPlayer(MCTSPlayer):
    name = "NetworkPlayer"

    def __init__(
            self,
            model_path: str,
            transform_func: callable,
            **kwargs
    ):
        super(NetworkPlayer, self).__init__(**kwargs)
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        from tensorflow import keras
        model = keras.models.load_model(model_path)
        self.predictor = NNPredictor(model=model, transform_input=transform_func)

    def perform_search(self, root):
        """
        Führe die Monte-Carlo-Baumsuche ausgehend von einem Wurzelknoten durch.

        So lange noch Ressourcen verfügbar sind, dies können eine begrenzte Anzahl an Iterationen oder ein Zeitlimit
        sein, wird der MCTS Algorithmus ausgeführt. Die tree_policy durchläuft den bereits existierenden Baum bis ein
        Blatt gefunden wurde, welches wenn möglich in der tree_policy expandiert wird. Danach wird der Spielzustand
        in diesem Blatt mit evaluate_game_state zu Ende simuliert. Das Ergebnis dieser Simulation - -1, 0  oder 1 - wird
        mit in backup benutzt, um die Statistiken der Knoten zu aktualisieren.
        Nach Ablauf der Ressourcen wird der beste Zug durch best_move ausgewählt. Dies ist der Zug mit der höchsten
        durchschnittlichen Belohnung.

        :param root: Der Wurzelknoten v0
        :return:
        """
        while self.has_resources():
            leaf = self.tree_policy(root)
            reward = self.predictor.evaluate_game_state(leaf.game_state)
            self.backup(leaf, reward)
        return self.best_move(root)


if __name__ == "__main__":
    from bachelorarbeit.games import Observation, Configuration, ConnectFour
    from bachelorarbeit.tools import timer
    from bachelorarbeit.network import build_model

    steps = 2000

    predictor = NNPredictor(model=build_model())

    p = NetworkPlayer(max_steps=steps, predictor=predictor)
    conf = Configuration()
    game = ConnectFour(columns=conf.columns, rows=conf.rows, inarow=conf.inarow, mark=1)
    obs = Observation(board=game.board.copy(), mark=game.mark)

    with timer(f"{steps} steps"):
        m = p.get_move(obs, conf)
    print(m)