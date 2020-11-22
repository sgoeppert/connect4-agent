"""
Bereitet die gesammelten Daten so auf, dass die unterschiedlichen Netzwerk-Typen damit arbeiten können. Danach wird
autokeras benutzt um NUM_TRIALS Netzwerke pro Konfiguration zu trainieren und vergleichen.

Die transformierten Daten werden nach ./memory_ak/[SAMPLE_SIZE]/[PROJECT_NAME] geschrieben, z.B.
    ./memory_ak/400000/transform_board_cnn_data_400000_norm.pickle

Während des Trainings schreibt autokeras die untersuchten Netzwerke nach ./auto_models/[SAMPLE_SIZE] und nach dem
Training die fertigen Modelle nach ./best_models/[SAMPLE_SIZE]/[PROJECT_NAME] z.B.
    ./best_models/400000/padded_cnn_norm
"""
import os
import tensorflow as tf
import autokeras as ak
from pathlib import Path
import pickle
import itertools

import config
from bachelorarbeit.tools import transform_board, transform_board_nega as regular, transform_board_cnn as cnn
from bachelorarbeit.selfplay import Memory
from bachelorarbeit.network import transform_memory, split_data
from bachelorarbeit.autokeras import PaddedConvBlock

MEMORY_FILE_NAME = "selfplay_mcts_strong_v2.pickle"

GENERATE_DATA_ONLY = False
SAMPLE_SIZE = 400_000
OVERWRITE = True
AUGMENT_DATA = True

CUSTOM_CNN = True
TRANSFORMATIONS = [regular, transform_board, cnn]
# TRANSFORMATIONS = [cnn]
NORMALIZE = [True, False]
# NORMALIZE = [False]
DUPLICATES = "average"

MODEL_NAMES = {
    'transform_board': 'naive',
    'transform_board_nega': 'regular',
    'transform_board_cnn': 'cnn'
}
MODEL_DIRECTORY = Path(config.ROOT_DIR) / 'auto_models' / str(SAMPLE_SIZE)

""" Training parameters """
BATCH_SIZE = 64
NUM_TRIALS = 5
EPOCHS = 300

for transform, norm in itertools.product(TRANSFORMATIONS, NORMALIZE):
    fname_prefix = transform.__name__
    fname_suffix = ""
    custom_prefix = ""

    if norm:
        fname_suffix = "_norm"

    if CUSTOM_CNN:
        custom_prefix += "padded_"

    if DUPLICATES == "keep":
        fname_suffix += "_dup"
    elif DUPLICATES == "average":
        fname_suffix += "_avg"

    PROJECT_NAME = custom_prefix + MODEL_NAMES[fname_prefix] + fname_suffix
    DATA_FILE = Path(config.ROOT_DIR) / "memory_ak" \
        / str(SAMPLE_SIZE) / "{}_data_{}{}.pickle".format(fname_prefix, SAMPLE_SIZE, fname_suffix)

    if not os.path.exists(DATA_FILE):
        GENERATE_DATA = True
    else:
        GENERATE_DATA = False

    if GENERATE_DATA:
        memory = Memory(MEMORY_FILE_NAME)
        print(f"Memory has {memory.num_states} states")

        X, y = transform_memory(memory,
                                duplicates=DUPLICATES,
                                sample_size=SAMPLE_SIZE,
                                transform_func=transform,
                                normalize_reward=norm,  # Output zwischen (-1 und 1) oder (0 und 1)
                                augment_data=AUGMENT_DATA,
                                )
        print(f"After transformation storing {len(X)} states")
        os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)

        with open(DATA_FILE, "wb") as f:
            pickle.dump((X, y), f)

        if GENERATE_DATA_ONLY:
            continue
    else:
        if GENERATE_DATA_ONLY:
            continue

        with open(DATA_FILE, "rb") as f:
            X, y = pickle.load(f)

    TENSORBOARD_DIR = Path("tensorboard_log_autokeras") / str(SAMPLE_SIZE) / PROJECT_NAME
    CALLBACKS = [
        tf.keras.callbacks.TensorBoard(log_dir=TENSORBOARD_DIR),
    ]

    train_X, train_y, split_X, split_y = split_data(X, y, percent=0.1, shuffle=False)
    test_X, test_y, val_X, val_y = split_data(split_X, split_y, percent=0.5, shuffle=False)

    X = train_X
    X_val = val_X
    X_test = test_X

    if fname_prefix == "transform_board_cnn":
        input_node = ak.ImageInput()
        if CUSTOM_CNN:
            output_node = PaddedConvBlock()(input_node)
        else:
            output_node = ak.ConvBlock()(input_node)
        output_node = ak.DenseBlock()(output_node)
    else:
        input_node = ak.Input()
        output_node = ak.DenseBlock()(input_node)

    output_node = ak.RegressionHead(metrics=["mae"], loss="mse")(output_node)

    reg = ak.AutoModel(
        inputs=input_node,
        outputs=output_node,
        overwrite=OVERWRITE,
        max_trials=NUM_TRIALS,
        project_name=PROJECT_NAME,
        directory=MODEL_DIRECTORY
    )

    reg.fit(X, train_y,
            validation_data=(X_val, val_y),
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            verbose=1,
            callbacks=CALLBACKS)

    try:
        mae, _ = reg.evaluate(X_test, test_y, verbose=0)
        print(f"Evaluation score {mae}")
    except:
        print("Error evaluating")

    model = reg.export_model()
    try:
        model.save('best_models/' + str(SAMPLE_SIZE) + "/" + PROJECT_NAME, save_format="tf")
    except:
        model.save('best_models/' + str(SAMPLE_SIZE) + "/" + PROJECT_NAME + ".h5")
