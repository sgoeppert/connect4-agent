import multiprocessing as mp
import numpy as np
import os
import json
import importlib
import itertools
import argparse

from datetime import datetime
from tqdm import tqdm
from tools import play_game, setup_logger


def play_game_mp(params):
    _p1_settings, _p2_settings = params

    def p1(obs, conf):
        return _p1_settings["func"](obs, conf, _p1_settings)

    def p2(obs, conf):
        return _p2_settings["func"](obs, conf, _p2_settings)

    return play_game(p1, p2)


def write_json_results_to_file(file, data):
    os.makedirs(os.path.dirname(file), exist_ok=True)
    with open(file.format(datetime.now().strftime("%Y-%m-%d_%H-%M-%S")), "w+") as f:
        json.dump(data, f)


def parse_json_config(json_data):
    name = json_data["name"]
    module = json_data["module"]
    imported_mod = importlib.import_module(module)
    func = getattr(imported_mod, "agent")
    configurable = getattr(imported_mod, "get_configurable_options")()

    settings = json_data["settings"]
    keys = settings.keys()

    missing = [k for k in configurable if k not in keys]
    if len(missing) > 0:
        print("Configuration is missing values: {}\n\tIn data {}".format(missing, json_data))
        exit(1)

    for k, v in settings.items():
        if type(v) is not list:
            settings[k] = [v]

    vals = list(settings.values())
    val_combinations = itertools.product(*vals)

    base_setting = {
        "name": name,
        "func": func
    }

    all_settings = []
    for cmb in val_combinations:
        sett = base_setting.copy()
        for i, key in enumerate(keys):
            sett[key] = cmb[i]
        all_settings.append(sett)
    return all_settings


def load_configuration(file):
    try:
        with open(file, "r") as f:
            json_data = json.load(f)
            settings = parse_json_config(json_data)
    except FileNotFoundError:
        print("Could not open configuration file {}".format(file))
        exit(1)
    except json.decoder.JSONDecodeError:
        print("Could not parse JSON in file {}".format(file))
        exit(1)

    return settings


logger = None


def log(msg):
    global logger
    if logger is not None:
        logger.info(msg)
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{timestamp} INFO {msg}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run players against eachother")
    parser.add_argument("playerconfig",
                        type=str,
                        help="Player configuration file(s). Will use the same configuration for both players if only "
                             "one is provided.",
                        nargs="+")
    parser.add_argument("-games", type=int,
                        help="Number of simulations to run for each value combination. The default is 300.",
                        default=300)
    parser.add_argument("-outfile",
                        type=str,
                        help="File to write experiment results to. If the file name contains a placeholder {} it will "
                             "be replaced with the current datetime in the format Y-m-d_H-M-S, e.g 2020-07-02_15-20-55."
                             " Results will be written in JSON format. Results are written to "
                             "\"data/experiment_{}.json\" by default.",
                        default="data/experiment_{}.json")
    parser.add_argument("-logfile", type=str, nargs="?",
                        help="File to log output to. Will by default print to stdout.")
    parser.add_argument("-label",
                        type=str,
                        help="The label for this experiment. This is added to the output file to give a meaningful "
                             "name to the experiment performed.")
    parser.add_argument("-watch_param",
                        type=str,
                        help="The main parameter to be watched. Must be a key of the configuration file for "
                             "both players. This will add information in the live output about which parameters "
                             "are being compared.")
    parser.add_argument("-p", type=int,
                        help="Number of processes run in parallel. If the number provided is higher than the number of "
                             "available logical cores, this will use all but one core.", default=8)

    args = parser.parse_args()

    label = "MCTS comparison"
    if args.label is not None:
        label = args.label

    results_file = args.outfile
    player_config_files = args.playerconfig

    if len(player_config_files) > 2:
        # if more than 2 config files are provided, throw an error
        msg = "More than 2 player configurations provided. Got {}".format(player_config_files)
        raise RuntimeError(msg)

    if len(player_config_files) == 1:
        # if only one config is provided, use it for both players
        player_config_files.append(player_config_files[0])

    watch_param = "cp"
    if args.watch_param is not None:
        watch_param = args.watch_param

    watch = True
    games_per_value = args.games

    num_processes = max(1, args.p)  # limit num_processes to positive values
    if num_processes >= mp.cpu_count():
        if mp.cpu_count() == 1:  # if only one core is available, use that
            num_processes = 1
        else:  # else use all but one core
            num_processes = mp.cpu_count() - 1

    player_settings = [load_configuration(f) for f in player_config_files]

    if (watch_param not in player_settings[0][0].keys()) or (watch_param not in player_settings[1][0].keys()):
        print("WARNING: Could not find watch_param \"{}\" in configurations. ".format(watch_param))
        watch = False

    if args.logfile is not None:
        logger = setup_logger("experiment", args.logfile.format(datetime.now().strftime("%Y-%m-%d_%H-%M")))

    combinations = list(itertools.product(*player_settings))

    ctx = mp.get_context("spawn")

    with ctx.Pool(num_processes, maxtasksperchild=10) as pool:
        log(f"Running experiment {label}. Total {len(combinations)} combinations with {games_per_value} games each.")
        experiment = {
            "label": label,
            "result_sets": []
        }
        for pair in combinations:
            player_info = [p.copy() for p in pair]
            for i, p in enumerate(player_info):
                del(p["func"])
                log(f"p{i+1}: {p}")

            result_set = {
                "players": player_info,
                "results": []
            }

            labels = ["p1", "p2"]
            if watch:
                comparison = []
                for i in range(len(labels)):
                    watch_val = pair[i][watch_param]
                    comparison.append("{}: {}".format(watch_param, watch_val))
                    labels[i] += f"({comparison[-1]})"

            work = pool.imap_unordered(play_game_mp, (pair for _ in range(games_per_value)))
            for game_result in tqdm(work, total=games_per_value):
                result_set["results"].append(game_result)

            mean_scores = (1 + np.mean(result_set["results"], axis=0)) / 2
            msg = " ".join([f"{lbl} {mn}" for lbl, mn in zip(labels, mean_scores)])
            log(msg)

            experiment["result_sets"].append(result_set)

    write_json_results_to_file(results_file, experiment)
