from tensorflow.keras import models, layers, optimizers

from bachelorarbeit.games import ConnectFour
from bachelorarbeit.tools import transform_board, transform_board_cnn
import math
import numpy as np


def dedup(data):
    seen_states = {}
    for state in data:
        b = str(state["board"])
        if b in seen_states:
            seen_states[b]["seen"] += 1
            res = seen_states[b]["result"]
            delta_res = (state["result"] - res) / seen_states[b]["seen"]
            seen_states[b]["result"] += delta_res
        else:
            seen_states[b] = {"seen": 1, "result": state["result"], "board": state["board"]}

    # _g = ConnectFour()
    # print(seen_states[str(_g.copy().play_move(0).board)])
    # print(seen_states[str(_g.copy().play_move(1).board)])
    # print(seen_states[str(_g.copy().play_move(2).board)])
    # print(seen_states[str(_g.copy().play_move(3).board)])
    # print(seen_states[str(_g.copy().play_move(4).board)])
    # print(seen_states[str(_g.copy().play_move(5).board)])
    # print(seen_states[str(_g.copy().play_move(6).board)])

    return list(seen_states.values())


def transform_memory(memory, transform_func=transform_board, sample_size=5000, remove_duplicates=False,
                     normalize_reward=False):
    if remove_duplicates:
        game_data = dedup(memory.game_data)
    else:
        game_data = memory.game_data

    if sample_size <= 0:
        sample_size = len(game_data)
    sample_size = min(sample_size, len(game_data))

    p = np.random.choice(len(game_data), sample_size)

    inputs, targets = [], []
    for idx in p:
        # for state in memory.game_data[p]:
        state = game_data[idx]
        inputs.append(transform_func(state["board"]))
        targets.append(state["result"])

    targets = np.asarray(targets)
    if normalize_reward:
        targets = (targets + 1) / 2

    return np.array(inputs), targets


def split_data(x: np.array, y: np.array, percent=0.1, shuffle=True):
    train_size = math.floor(len(x) * (1 - percent))

    if shuffle:
        p = np.random.permutation(len(x))
        shuff_x = x[p]
        shuff_y = y[p]
    else:
        shuff_x = x
        shuff_y = y

    return shuff_x[:train_size], shuff_y[:train_size], shuff_x[train_size:], shuff_y[train_size:]


def build_model(shape=(42,), hidden_layers: list = None, lr=0.0001, activation="tanh"):
    if hidden_layers is None:
        hidden_layers = [16]

    assert len(hidden_layers) > 0, "must provide at least one hidden layer e.g. [16]"

    shape_name = "I" + "-".join(map(str, shape)) + "_"
    model_short_name = ["linear"] + [shape_name] + hidden_layers + [activation]
    model_short_name = "_".join(map(str, model_short_name))
    model = models.Sequential(name=model_short_name)
    first = hidden_layers[0]
    hidden_layers = hidden_layers[1:]

    model.add(layers.Dense(first, activation="relu", input_shape=shape))
    for l in hidden_layers:
        model.add(layers.Dense(l, activation="relu"))

    model.add(layers.Dense(1, activation=activation))
    model.compile(optimizer=optimizers.Adam(learning_rate=lr),
                  loss="mse",
                  metrics=["mae"])

    return model


def build_cnn_2(shape=(6, 7, 3), filters=16, dense_neurons=16, lr=0.0001, activation="tanh"):
    model = models.Sequential()
    model.add(layers.Conv2D(filters, (3, 3), input_shape=shape, padding="same"))
    model.add(layers.Conv2D(filters, (3, 3), padding="same"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.ReLU())
    model.add(layers.Conv2D(filters // 2, (1, 1)))
    model.add(layers.ReLU())
    model.add(layers.Flatten())
    model.add(layers.Dense(dense_neurons, activation="relu"))
    model.add(layers.Dense(1, activation=activation))

    model.compile(optimizer=optimizers.Adam(learning_rate=lr),
                  loss="mse",
                  metrics=["mae"])

    return model


def build_cnn(shape=(6, 7, 3), filters=16, dense_neurons=16, lr=0.0001, activation="tanh"):
    model = models.Sequential()
    model.add(layers.Conv2D(filters, (3, 3), activation="relu", input_shape=shape, padding="same"))
    model.add(layers.Conv2D(filters, (3, 3), activation="relu", padding="same"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(dense_neurons, activation="relu"))
    model.add(layers.Dense(1, activation=activation))

    model.compile(optimizer=optimizers.Adam(learning_rate=lr),
                  loss="mse",
                  metrics=["mae"])

    return model


def load_model(path):
    return models.load_model(path)


if __name__ == "__main__":
    g = ConnectFour()
    g.play_move(3)

    model = build_cnn()

    tf = np.array([transform_board_cnn(g.board)])

    res = model(tf, training=False)
    print(res)

    # shape = (6*7,)
    #
    # model = build_model()
    #
    # memory = Memory("test.pickle")
    #
    # arena = Arena(players=(MCTSPlayer, MCTSPlayer),
    #               constructor_args=({"max_steps": 300}, {"max_steps": 300}),
    #               num_games=200,
    #               num_processes=8,
    #               memory=memory)
    # while memory.num_states < 15000:
    #     print(f"{memory.num_states} game states in memory file")
    #     arena.run_game_mp()
    #     memory.save_data()
    #
    # print(f"{memory.num_states} game states in memory file")
    #
    # X, y = transform_memory(memory)
    # train_x, train_y, test_x, test_y = train_test_split(X, y, percent=0.1)
    #
    # history = model.fit(x=train_x, y=train_y, batch_size=64, epochs=100)
    #
    # model.evaluate(test_x, test_y)
    #
    # pred = model.predict(train_x[:10])
    # tar = train_y[:10]
    # for p, t in zip(pred,tar):
    #     print(p,t)
