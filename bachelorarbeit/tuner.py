import multiprocessing
import random
import itertools
from math import sqrt, log, ceil
from typing import List, Tuple, Any, Type
from tqdm import tqdm
import os
import pickle
import json
from datetime import datetime
from pathlib import Path

import config
from bachelorarbeit.games import Configuration, ConnectFour, Observation
from bachelorarbeit.selfplay import Arena
from bachelorarbeit.players.base_players import Player
from bachelorarbeit.players.mcts import MCTSPlayer

DEFAULT_DIR = "MCTSTuner"
VIRTUAL_LOSS = 4


def get_timestamp():
    return datetime.now().strftime("%Y/%m/%d-%H:%M:%S")


class TinyArena:
    def __init__(
            self,
            players: Tuple[Type[Player], Type[Player]],
            opponent_config: dict
    ):
        self.player_classes = players
        self.opponent_config = opponent_config
        self.player_config = None
        self.flip_players = False

    def instantiate_players(self) -> List[Player]:
        players = []
        for pclass, arg in zip(self.player_classes, (self.player_config, self.opponent_config)):
            if arg:
                players.append(pclass(**arg))
            else:
                players.append(pclass())

        if self.flip_players:
            players = players[::-1]
        return players

    def run_game(self):
        conf = Configuration()
        game = ConnectFour(columns=conf.columns, rows=conf.rows, inarow=conf.inarow, mark=1)
        players = self.instantiate_players()

        s = 0
        while not game.is_terminal():
            obs = Observation(board=game.board.copy(), mark=game.mark)
            active_player = s & 1
            m = players[active_player].get_move(obs, conf)
            # print(m)
            game.play_move(m)
            s += 1

        return game.get_reward(1), game.get_reward(2)

    def run(self, leaf: "TunerNode"):
        self.player_config = leaf.fixed_parameters

        results = [self.run_game()]
        self.flip_players = True
        results.append(self.run_game()[::-1])
        return results


class Parametrization:
    def __init__(self):
        self.options = []
        self.default_config = {}

    def add_default_option(self, name, value):
        self._check_key(name)
        self.default_config[name] = value

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

        self.virtual_losses = 0

    def q(self) -> float:
        if self.n == 0:
            return -1
        else:
            return (self.wins + 0.5 * self.draws) / sum([self.wins, self.losses, self.draws])

    def explo(self):
        if self.n == 0:
            return float("inf")
        else:
            return sqrt(2 * log(self.parent.n) / self.n)

    def is_expanded(self) -> bool:
        return self._expanded

    def is_terminal(self) -> bool:
        return self._terminal

    def add_virtual_loss(self):
        self.virtual_losses += 1
        self.losses += VIRTUAL_LOSS
        self.n += VIRTUAL_LOSS

    def remove_virtual_loss(self):
        if self.virtual_losses > 0:
            self.virtual_losses -= 1
            self.losses -= VIRTUAL_LOSS
            self.n -= VIRTUAL_LOSS

    def expand(self) -> "TunerNode":
        for name, val in self.options:
            fixed_params = {**self.fixed_parameters, name: val}
            self.children[(name, val)] = TunerNode(self.free_parameters.copy(), fixed_params, parent=self)
        self._expanded = True
        return random.choice(list(self.children.values()))

    def best_child(self, c: float) -> "TunerNode":
        return max(self.children.values(), key=lambda ch: ch.q() + c * ch.explo())

    def __repr__(self):
        return f"Node(n:{self.n}, q:{self.q()}, W/D/L: {self.wins}/{self.draws}/{self.losses}, vloss: {self.virtual_losses})"


class MCTSTuner:
    def __init__(self, player: Type[Player], parameters: Parametrization, opponent: Type[Player], opponent_config: dict,
                 name: str = None, directory: str = DEFAULT_DIR, checkpoint_interval: int = 100, exploration: float = 1.0):

        if len(parameters) == 0:
            raise ValueError("Parametrization has no options. Supply add least one tunable option with .choice(), "
                             ".boolean() or .xor()")

        self.player = player
        self.parameters = parameters
        self.opponent = opponent
        self.opponent_config = opponent_config

        self.root = None
        self.exploration = exploration

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
        return sorted(statistics, key=lambda d: -d["q"])

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
            "checkpoints": [],
            "opponent": {}
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
        timestamp: str = get_timestamp()
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
        timestamp: str = get_timestamp()
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
        cp_meta["opponent"] = self.opponent_config
        self._write_checkpoint_meta(cp_meta, run_dir)

    def _write_checkpoint_data(self, run_dir):
        cp_dir = run_dir / "checkpoint_{}".format(self.checkpoint_number)
        os.makedirs(cp_dir, exist_ok=True)
        if self.process_pool is not None:
            self.process_pool.close()
            self.process_pool = None

        # write pickled self to dir
        with open(cp_dir / "tuner.dat", "wb+") as f:
            pickle.dump(self, f)
        # write terminal_nodes to dir
        with open(cp_dir / "node_stats.json", "w+") as f:
            json.dump(self.collect_terminal_node_stats(), f)

    def write_checkpoint(self):
        # update meta_json next_run_number and last_run
        run_dir = self.project_dir / "run_{}".format(self.run_number)
        self._update_checkpoint_meta(run_dir)
        self._update_meta()
        self._write_checkpoint_data(run_dir)

    @staticmethod
    def tree_policy(root, c) -> TunerNode:
        current = root
        current.add_virtual_loss()
        while not current.is_terminal():
            if current.is_expanded():
                current: TunerNode = current.best_child(c)
            else:
                current = current.expand()
            current.add_virtual_loss()
        return current

    def evaluate_single_leaf(self, leaf) -> List[Tuple[float, float]]:
        num_processes = 7
        num_games = 10

        if self.process_pool is None:
            self.process_pool = multiprocessing.Pool(processes=num_processes, maxtasksperchild=20)

        players = (self.player, self.opponent)
        player_args = (leaf.fixed_parameters, self.opponent_config)
        arena = Arena(players=players, constructor_args=player_args, num_games=num_games, num_processes=num_processes)
        results = arena.run_game_mp(pool=self.process_pool, show_progress_bar=False, max_tasks=20)

        return results

    def evaluate_and_backup(self, leaves, total):
        if self.process_pool is None:
            self.process_pool = multiprocessing.Pool(config.NUM_PROCESSES, maxtasksperchild=10)

        leaves1, leaves2 = itertools.tee(leaves)

        players = (self.player, self.opponent)
        arena = TinyArena(players=players, opponent_config=self.opponent_config)
        pbar = tqdm(total=total, desc="Processing leaves", leave=False)
        for res, leaf in zip(self.process_pool.imap(arena.run, leaves1), leaves2):
            self.backup(leaf, res)
            pbar.update()
        pbar.close()

    def leaf_generator(self, root, chunk_size):
        for i in range(chunk_size):
            yield self.tree_policy(root, self.exploration)
        return

    def backup(self, leaf, score):
        if leaf not in self.terminal_nodes:
            self.terminal_nodes.append(leaf)

        self.n += 1  # keep track of number of leafs processed

        current: TunerNode = leaf
        win = score.count((1, -1))
        draw = score.count((0, 0))
        lose = score.count((-1, 1))

        while current is not None:
            current.remove_virtual_loss()
            current.n += 1
            current.wins += win
            current.draws += draw
            current.losses += lose
            current = current.parent

    def search(self, iterations):
        total_combinations = len(list(itertools.product(*self.parameters.options)))
        if (self.n + iterations) < total_combinations:
            print(f"Warning: {iterations} iterations will not explore all possible {total_combinations} combinations.")

        if self.root is None:
            self.root = TunerNode(self.parameters.copy(), self.parameters.default_config.copy())

        chunk_size = self.checkpoint_interval
        n_chunks = ceil(iterations / chunk_size)
        pbar = tqdm(total=n_chunks, desc="Processing chunks")
        processed = 0
        steps = 0
        while processed < iterations:
            chunk_size = min(chunk_size, iterations - processed)

            # the generator yields a new leaf each time next() is called on it
            leaves = self.leaf_generator(self.root, chunk_size)
            # these leaves are consumed by a pool of processes. A process grabs a leaf, evaluates it
            # and the result is then backed up the tree. Uses virtual loss so it becomes less likely
            # for two processes to evaluate the same leaf
            self.evaluate_and_backup(leaves, chunk_size)
            pbar.update()
            steps += 1
            processed += chunk_size
            self.write_checkpoint()
        pbar.close()

        # Runs n games in each leaf node in parallel
        # for _ in tqdm(range(iterations)):
        #     root = self.root
        #     leaf = self.tree_policy(root, 1.0)
        #     score = self.evaluate(leaf)
        #     self.backup(leaf, score)
        #
        #     self.n += 1
        #
        #     if self.n % self.checkpoint_interval == 0:
        #         self.write_checkpoint()

        self.write_checkpoint()

    def get_best_parameters(self):
        if self.root is None:
            return None

        current: TunerNode = self.root
        while not current.is_terminal():
            current = current.best_child(0)
        return current.fixed_parameters


def create_tuner(player: Type["Player"], parametrization: Parametrization,
                 name=None, directory=DEFAULT_DIR,
                 opponent=MCTSPlayer, opponent_config=None,
                 checkpoint_interval=100,
                 exploration: float = 1.0
                 ) -> MCTSTuner:

    if opponent_config is None:
        opponent_config = {}
    return MCTSTuner(player, parametrization,
                     opponent, opponent_config,
                     name=name, directory=directory,
                     checkpoint_interval=checkpoint_interval,
                     exploration=exploration)


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
        tuner: MCTSTuner = pickle.load(f)
    tuner.name = name

    return tuner

if __name__ == "__main__":
    from bachelorarbeit.players.mcts import MCTSPlayer
    from bachelorarbeit.tools import get_range

    # params = Parametrization()
    # params.choice("exploration_constant", get_range(1.0, 9), default=1.0)

    # tuner = create_tuner(MCTSPlayer, params, checkpoint_interval=1, opponent_config={"max_steps": 10})
    tuner = load_tuner("MCTSPlayer")
    tuner.search(10)
