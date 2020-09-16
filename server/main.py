from flask import Flask, request, jsonify
from flask_caching import Cache
from waitress import serve
import tensorflow as tf
import numpy as np
from queue import Empty, Queue
import threading
import logging
import time

from bachelorarbeit.tools import transform_board_cnn
import config

cache_conf = {
    "CACHE_TYPE": "simple",
    "CACHE_DEFAULT_TIMEOUT": 1000,
    "CACHE_THRESHOLD": 100_000
}

BATCH_SIZE = 64
BATCH_TIMEOUT = 0.01
CHECK_INTERVAL = 0.001

requests_queue = Queue()

app = Flask(__name__)
app.config.from_mapping(cache_conf)
cache = Cache(app)

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)


MODEL_PATH = config.ROOT_DIR + "/best_models/cnn_bonus_channel_aug"

# model = tf.keras.models.load_model(MODEL_PATH)


def handle_requests_by_batch():
    while True:
        requests_batch = []
        while not (len(requests_batch) > BATCH_SIZE
                   or (len(requests_batch) > 0
                       and time.time() - requests_batch[0]["time"] > BATCH_TIMEOUT)
        ):
            try:
                requests_batch.append(requests_queue.get(timeout=CHECK_INTERVAL))
            except Empty:
                continue

        batch_inputs = np.array([req["input"] for req in requests_batch])
        batch_inputs = transform_board_cnn(batch_inputs)
        # print(batch_inputs)
        # batch_outputs = model(batch_inputs, training=False)
        batch_outputs = [np.asarray(x).reshape((1,-1))[0] for x in batch_inputs]
        # print(batch_outputs)
        for req, output in zip(requests_batch, batch_outputs):
            req["output"] = output


threading.Thread(target=handle_requests_by_batch).start()


@app.route("/predict", methods=["POST"])
def predict():
    global stats, request_count
    data = request.json

    # print(data)

    key = str(data["input"][0])
    if cache.get(key) is not None:
        # print("cache hit")
        return cache.get(key)

    req = {
        "input": data["input"][0],
        "time": time.time()
    }
    requests_queue.put(req)

    while "output" not in req:
        time.sleep(CHECK_INTERVAL)

    cache.set(key, {"predictions": req["output"].tolist()})
    return {"predictions": req["output"].tolist()}


"""

    board = request.json["input"]
    transformed = transform_board_cnn(board)
    received_input = np.array([transformed])
    model_output = model(received_input, training=False)
    return {"predictions": np.mean(model_output).astype(float)}
"""

if __name__ == "__main__":
    serve(app, host="0.0.0.0", port=5000, threads=64)
    # app.run(host="127.0.0.1", port=5000)
