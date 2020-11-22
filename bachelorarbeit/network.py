"""
In diesem Modul werden einige Funktion für die Verarbeitung von Spielzuständen und Trainingsdaten gesammelt
"""
from typing import Tuple

from bachelorarbeit.tools import transform_board, transform_board_cnn
import math
import numpy as np


def dedup(data):
    """
    Entfernt doppelte Spielzustände und reduziert die Bewertungen dieser Duplikate auf ihren Durchschnittswert
    :param data:
    :return:
    """
    seen_states = {}
    for state in data:
        b = tuple(state["board"])
        if b in seen_states:
            seen_states[b]["seen"] += 1
            res = seen_states[b]["result"]
            delta_res = (state["result"] - res) / seen_states[b]["seen"]
            seen_states[b]["result"] += delta_res
        else:
            seen_states[b] = {"seen": 1, "result": state["result"], "board": state["board"]}

    return list(seen_states.values())


def augment(data: list):
    """
    Augmentiert die Spielzustände durch horizontale Spiegelung
    :param data:
    :return:
    """
    additional_states = []
    for state in data:
        flipped = np.array(state["board"]).reshape((6, 7))[:, ::-1].reshape(-1).tolist()
        additional_states.append({
            "board": flipped,
            "result": state["result"]
        })
    return data + additional_states


def remove_null_states(data: list):
    """
    Entfernt Zustände mit leerem Spielfeld
    :param data:
    :return:
    """
    return [entry for entry in data if any(entry["board"])]


def average_duplicates(data: list):
    """
    Ersetzt die Bewertung jedes Spielzustands durch den Durchschnittswert aller Spiele in denen dieser Zustand auftrat.
    Entfernt Duplikate nicht.
    :param data:
    :return:
    """
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

    out_states = []
    for state in seen_states.values():
        for i in range(state["seen"]):
            out_states.append(state)

    return out_states


def transform_memory(memory, transform_func=transform_board, sample_size=5000, duplicates="remove",
                     normalize_reward=False, augment_data=False, scale_reward=1):
    """
    Wendet verschiedenste Transformationen auf ein Memory Objekt an wie Augmentierung und Entfernung von Duplikaten.
    :param memory:
    :param transform_func:
    :param sample_size:
    :param duplicates:
    :param normalize_reward:
    :param augment_data:
    :param scale_reward:
    :return:
    """
    game_data = memory.game_data

    if augment_data:
        game_data = augment(game_data)
        print(f"After augmenting total states: {len(game_data)}.")

    if duplicates == "remove":
        game_data = dedup(game_data)
        print(f"Without duplicates total states: {len(game_data)}.")
    elif duplicates == "average":
        game_data = average_duplicates(game_data)
        game_data = remove_null_states(game_data)
        print(f"Averaged without null state: {len(game_data)}.")
    else:
        game_data = remove_null_states(game_data)
        print(f"Without null state: {len(game_data)}.")

    if sample_size <= 0:
        sample_size = len(game_data)
    sample_size = min(sample_size, len(game_data))

    p = np.random.choice(len(game_data), sample_size)

    inputs, targets = [], []
    for idx in p:
        state = game_data[idx]
        inputs.append(state["board"])
        targets.append(state["result"])

    inputs = transform_func(inputs)
    targets = np.asarray(targets)
    if normalize_reward:
        targets = (targets + 1) / 2

    targets *= scale_reward

    return np.array(inputs), targets


def split_data(x: np.array, y: np.array, percent=0.1, shuffle=True) -> Tuple[np.array, np.array, np.array, np.array]:
    """
    Mischt die Daten, die Relation zwischen x und y bleibt dabei erhalten, und teilt sie in zwei Datensätze auf
    :param x:
    :param y:
    :param percent:
    :param shuffle:
    :return:
    """
    train_size = math.floor(len(x) * (1 - percent))

    if shuffle:
        p = np.random.permutation(len(x))
        shuff_x = x[p]
        shuff_y = y[p]
    else:
        shuff_x = x
        shuff_y = y

    return shuff_x[:train_size], shuff_y[:train_size], shuff_x[train_size:], shuff_y[train_size:]

