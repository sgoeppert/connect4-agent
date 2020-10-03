from flask import Flask, request, jsonify
from flask_caching import Cache
from waitress import serve
import tensorflow as tf
import numpy as np
from queue import Empty, Queue
import threading
import logging
import time

from bachelorarbeit.tools import transform_board_cnn, flip_board, denormalize
import config

cache_conf = {
    "CACHE_TYPE": "simple",
    "CACHE_DEFAULT_TIMEOUT": 1000,
    "CACHE_THRESHOLD": 1_000_000
}

BATCH_SIZE = 15
BATCH_TIMEOUT = 0.05
CHECK_INTERVAL = 0.0001

requests_queue = Queue()

app = Flask(__name__)
app.config.from_mapping(cache_conf)
cache = Cache(app)

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)


MODEL_PATH = config.ROOT_DIR + "/best_models/400000/padded_cnn_norm"

model = tf.keras.models.load_model(MODEL_PATH)
model(np.array(transform_board_cnn([0] * 42)), training=False)


def handle_requests_by_batch():
    start = time.time()
    items = 0
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

        items += len(requests_batch)
        # if len(requests_batch) > BATCH_SIZE:
        #     print("Batch full")
        # else:
        #     print("timeout with batch size: ", len(requests_batch))

        batch_inputs = []
        for req in requests_batch:
            batch_inputs.append(req["input"])
            batch_inputs.append(flip_board(req["input"]))

        batch_inputs = transform_board_cnn(batch_inputs)
        batch_outputs = model(np.array(batch_inputs), training=False).numpy()
        batch_outputs = denormalize(batch_outputs)

        for i, req in enumerate(requests_batch):
            req["output"] = np.mean([batch_outputs[2*i], batch_outputs[2*i+1]])

        if time.time() - start > 1:
            start = time.time()
            print(f"Requests per second: {items}")
            items = 0


threading.Thread(target=handle_requests_by_batch).start()


@app.route("/predict", methods=["POST"])
def predict():
    global stats, request_count
    data = request.json

    key = tuple(data["input"])
    if cache.get(key) is not None:
        return cache.get(key)

    req = {
        "input": data["input"],
        "time": time.time()
    }
    requests_queue.put(req)

    while "output" not in req:
        time.sleep(CHECK_INTERVAL)

    cache.set(key, {"predictions": float(req["output"])})
    return {"predictions": float(req["output"])}


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
