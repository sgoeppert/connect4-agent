import multiprocessing
import random
import itertools
from math import sqrt, log
from typing import List, Tuple, Any, Type
from tqdm import tqdm
import os
import pickle
import json
from datetime import datetime
from pathlib import Path

import config
from bachelorarbeit.selfplay import Arena
from bachelorarbeit.players.base_players import Player
from bachelorarbeit.players.mcts import MCTSPlayer

DEFAULT_DIR = "MCTSTuner"


class Parametrization:
    def __init__(self):
        self.options = []
        self.default_config = {}

    def pop_option(self) -> List:
        return list(self.options.pop(0))

    def _check_key(self, key: str):
        if key in self.default_config:
            raise KeyError("Duplicate name " + key)

    def choice(self, name: str, values: List, default: Any):
        self._check_key(name)

        self.default_config[name] = default
        self.options.append(tuple([(name, val) for val in values]))

    def boolean(self, name: str, default: bool = False):
        self._check_key(name)
        self.default_config[name] = default
        self.options.append(tuple([(name, b) for b in [default, not default]]))

    def xor(self, *options, default: Tuple = None):
        if default is None:
            raise ValueError("XOR needs default values")
        if len(options) != len(default):
            raise ValueError("A default value for each option needs to be supplied as a tuple")

        new_options = []
        for name, vals in options:
            self._check_key(name)
            for val in vals:
                new_options.append((name, val))

        for name, val in default:
            self._check_key(name)
            self.default_config[name] = val

        self.options.append(tuple(new_options))

    def __len__(self) -> int:
        return len(self.options)

    def copy(self) -> "Parametrization":
        cp = Parametrization()
        cp.options = self.options[:]
        cp.default_config = self.default_config.copy()
        return cp

    def toJson(self):
        return json.dumps(self, default=lambda o: o.__dict__,
                          sort_keys=True, indent=4)
    @classmethod
    def fromJson(cls, str):
        obj = cls()
        dict = json.loads(str)
        obj.default_config = dict["default_config"]
        obj.options = [ tuple([ tuple(inner) for inner in outer ]) for outer in dict["options"] ]
        return obj

class TunerNode:
    def __init__(self, free_parameters: Parametrization, fixed_parameters: dict, parent: "TunerNode" = None):
        self.free_parameters = free_parameters
        self.fixed_parameters = fixed_parameters

        if len(free_parameters):
            self.options = self.free_parameters.pop_option()
            self._expanded = False
            self._terminal = False
        else:
            self._expanded = True
            self._terminal = True

        self.children = {}
        self.parent = parent
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.n = 0

    def q(self) -> float:
        if self.n == 0:
            return -1
        else:
            return (self.wins + 0.5 * self.draws) / sum([self.wins, self.losses, self.draws])

    def explo(self):
        if self.n == 0:
            return float("inf")
        else:
            return sqrt(log(self.parent.n) / self.n)

    def is_expanded(self) -> bool:
        return self._expanded

    def is_terminal(self) -> bool:
        return self._terminal

    def expand(self) -> "TunerNode":
        for name, val in self.options:
            fixed_params = {**self.fixed_parameters, name: val}
            self.children[(name, val)] = TunerNode(self.free_parameters.copy(), fixed_params, parent=self)
        self._expanded = True
        return random.choice(list(self.children.values()))

    def best_child(self, c: float) -> "TunerNode":
        return max(self.children.values(), key=lambda ch: ch.q() + c * ch.explo())

    def __repr__(self):
        return f"Node(n:{self.n}, q:{self.q()}, W/D/L: {self.wins}/{self.draws}/{self.losses})"


class MCTSTuner:
    def __init__(self, player: Type[Player], parameters: Parametrization, opponent: Type[Player], opponent_config: dict,
                 name: str = None, directory: str = DEFAULT_DIR, checkpoint_interval=100):

        self.player = player
        self.parameters = parameters
        self.opponent = opponent
        self.opponent_config = opponent_config

        self.root = None

        self.checkpoint_interval = checkpoint_interval
        self.name = self.player.name if name is None else name
        self.base_dir = Path(config.ROOT_DIR) / directory
        self.project_dir = self.base_dir / self.name

        self.run_number = self._get_next_run_number()
        self.checkpoint_number = 0
        self.terminal_nodes = []
        self.n = 0

        self.process_pool = None

    def collect_terminal_node_stats(self):
        statistics = []
        for n in self.terminal_nodes:  # type: TunerNode
            statistics.append({
                "config": n.fixed_parameters,
                "wins": n.wins,
                "draws": n.draws,
                "losses": n.losses,
                "q": n.q(),
                "n": n.n
            })
        return statistics

    def _load_meta(self):
        meta_file_path = self.project_dir / "meta.json"
        if os.path.exists(meta_file_path):
            with open(meta_file_path, "r") as meta_file:
                return json.load(meta_file)
        return {
            "run_nr": -1,
            "last_run": {},
            "runs": []
        }

    def _write_meta(self, meta_dict: dict, meta_file_path):
        os.makedirs(os.path.dirname(meta_file_path), exist_ok=True)
        with open(meta_file_path, "w+") as meta_file:
            json.dump(meta_dict, meta_file)

    def _load_checkpoint_meta(self, run_dir):
        cp_meta = run_dir / "cp.json"
        if os.path.exists(cp_meta):
            with open(cp_meta, "r") as f:
                return json.load(f)
        return {
            "cp_num": -1,
            "last_cp": {},
            "checkpoints": []
        }

    def _write_checkpoint_meta(self, meta_dict: dict, run_dir):
        cp_meta = run_dir / "cp.json"
        os.makedirs(run_dir, exist_ok=True)
        with open(cp_meta, "w+") as f:
            json.dump(meta_dict, f)

    def _get_next_run_number(self) -> int:
        meta = self._load_meta()
        return meta["run_nr"] + 1

    def _update_meta(self):
        timestamp: str = datetime.now().strftime("%Y%m%d-%H%M%S")
        meta_file_path = self.project_dir / "meta.json"

        meta = self._load_meta()
        if meta["run_nr"] < self.run_number:
            meta["run_nr"] = self.run_number

        run_info = {
            "time": timestamp,
            "run": self.run_number,
            "checkpoint": self.checkpoint_number,
            "n": self.n
        }

        meta["last_run"] = run_info
        found = False
        for i, run in enumerate(meta["runs"]):
            if run["run"] == self.run_number:
                found = True
                meta["runs"][i] = run_info
                break

        if not found:
            meta["runs"].append(run_info)

        self._write_meta(meta, meta_file_path)

    def _update_checkpoint_meta(self, run_dir):
        timestamp: str = datetime.now().strftime("%Y%m%d-%H%M%S")
        cp_meta = self._load_checkpoint_meta(run_dir)
        # else get next cp number from cp.json
        self.checkpoint_number = cp_meta["cp_num"] + 1
        cp_data = {
            "time": timestamp,
            "cp": self.checkpoint_number,
            "n": self.n
        }
        cp_meta["last_cp"] = cp_data
        cp_meta["cp_num"] = self.checkpoint_number
        cp_meta["checkpoints"].append(cp_data)
        self._write_checkpoint_meta(cp_meta, run_dir)

    def _write_checkpoint_data(self, run_dir):
        cp_dir = run_dir / "checkpoint_{}".format(self.checkpoint_number)
        os.makedirs(cp_dir, exist_ok=True)
        # write pickled self to dir
        with open(cp_dir / "tuner.dat", "wb+") as f:
            self.process_pool.close()
            self.process_pool = None
            pickle.dump(self, f)
        # write terminal_nodes to dir
        with open(cp_dir / "node_stats.json", "w+") as f:
            json.dump(self.collect_terminal_node_stats(), f)

    def write_checkpoint(self):
        # update meta_json next_run_number and last_run
        run_dir = self.project_dir / "run_{}".format(self.run_number)
        self._update_meta()
        self._update_checkpoint_meta(run_dir)
        self._write_checkpoint_data(run_dir)

    @staticmethod
    def tree_policy(root, c) -> TunerNode:
        current = root
        while not current.is_terminal():
            if current.is_expanded():
                current = current.best_child(c)
            else:
                current = current.expand()
        return current

    def evaluate(self, leaf) -> Tuple[int, int, int]:
        num_processes = 10
        num_games = 30

        if self.process_pool is None:
            self.process_pool = multiprocessing.Pool(processes=num_processes, maxtasksperchild=20)

        players = (self.player, self.opponent)
        player_args = (leaf.fixed_parameters, self.opponent_config)
        arena = Arena(players=players, constructor_args=player_args, num_games=num_games, num_processes=num_processes)
        results = arena.run_game_mp(pool=self.process_pool, show_progress_bar=False, max_tasks=20)

        wins = results.count((1, -1))
        draws = results.count((0, 0))
        losses = results.count((-1, 1))

        return wins, draws, losses

    @staticmethod
    def backup(leaf, score):
        current = leaf
        win, draw, lose = score
        while current is not None:
            current.n += 1
            current.wins += win
            current.draws += draw
            current.losses += lose
            current = current.parent

    def search(self, iterations):
        total_combinations = len(list(itertools.product(*self.parameters.options)))
        if iterations < total_combinations:
            print(f"Warning: {iterations} iterations will not explore all possible {total_combinations} combinations.")

        if self.root is None:
            self.root = TunerNode(self.parameters.copy(), self.parameters.default_config.copy())

        for _ in tqdm(range(iterations)):
            root = self.root
            leaf = self.tree_policy(root, 1.0)
            score = self.evaluate(leaf)
            self.backup(leaf, score)

            self.n += 1
            if leaf not in self.terminal_nodes:
                self.terminal_nodes.append(leaf)

            if self.n % self.checkpoint_interval == 0:
                self.write_checkpoint()

        self.write_checkpoint()

    def get_best_parameters(self):
        if self.root is None:
            return None

        current: TunerNode = self.root
        while not current.is_terminal():
            current = current.best_child(0)
        return current.fixed_parameters


def create_tuner(player, params,
                 name=None, directory=DEFAULT_DIR,
                 opponent=MCTSPlayer, opponent_config=None,
                 checkpoint_interval=100
                 ) -> MCTSTuner:

    if opponent_config is None:
        opponent_config = {}
    return MCTSTuner(player, params, opponent, opponent_config, name=name, directory=directory, checkpoint_interval=checkpoint_interval)


def load_tuner(name, directory=DEFAULT_DIR, run_nr=None, checkpoint=None) -> MCTSTuner:
    project_dir = Path(config.ROOT_DIR) / directory / name

    if not os.path.exists(project_dir):
        raise ValueError("Could not find directory {}".format(project_dir))

    if run_nr is None:
        # get last_run from meta.json
        meta_file_path = project_dir / "meta.json"
        with open(meta_file_path, "r") as f:
            meta = json.load(f)
            run_nr = meta["run_nr"]

    run_dir = project_dir / "run_{}".format(run_nr)
    if checkpoint is None:
        cp_meta = run_dir / "cp.json"
        with open(cp_meta, "r") as f:
            cp_meta = json.load(f)
            checkpoint = cp_meta["cp_num"]

    cp_dir = run_dir / "checkpoint_{}".format(checkpoint)
    tuner_file = cp_dir / "tuner.dat"
    with open(tuner_file, "rb") as f:
        tuner = pickle.load(f)

    return tuner

if __name__ == "__main__":
    from bachelorarbeit.players.mcts import MCTSPlayer
    from bachelorarbeit.tools import get_range

    param = Parametrization()
    param.choice("exploration_constant", get_range(1.0, 9), default=1.0)

    tuner = create_tuner(MCTSPlayer, param, checkpoint_interval=20)
    tuner.search(500)

    # param = Parametrization()
    # opponent_config = {"exploration_constant": 1.0}
    #
    # param.choice("exploration_constant", [0.8, 0.9, 1.0, 1.1], default=1.0)
    # # param.xor(("alpha", [0.1, 0.5, 0.9]), ("b", [0.1, 0.01]), default=(("b", 0), ("alpha", None)))
    # # param.boolean("keep_tree", default=False)
    # tuner = MCTSTuner(MCTSPlayer, param, MCTSPlayer, opponent_config, name="Test")
    # tuner.search(5)
    # print(tuner.root.children)
    # print(tuner.get_best_parameters())
    #
    # tuner.write_checkpoint()

    # l = list(itertools.product(*param.options))
    # print(len(itertools.product(*param.options)))
    # print(l)
