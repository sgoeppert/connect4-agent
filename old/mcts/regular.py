import math
import random
import time
from old.games.connectfour import ConnectFour


class Node:
    def __init__(self, game_state, move=None, parent=None):
        self.game_state = game_state

        self.move = move
        self.parent = parent

        self.possible_actions = game_state.list_moves()

        self.children = []
        self.total_value = 0
        self.number_visits = 0

    def is_expanded(self):
        return len(self.possible_actions) == 0

    def Q(self):
        return self.total_value / self.number_visits

    def best_child(self, c_p):
        n_p = math.log(self.number_visits)
        return max(self.children, key=lambda c: c.Q() + c_p * math.sqrt(n_p/c.number_visits))

    def expand_random_child(self):
        m = random.choice(self.possible_actions)
        state = self.game_state.copy()
        state.play_move(m)
        ch = Node(game_state=state, move=m, parent=self)
        self.children.append(ch)
        self.possible_actions.remove(m)
        return ch

    def __repr__(self):
        return f"Node({self.move}, {self.number_visits})"


def backup(node, reward):
    node.number_visits += 1
    node.total_value += reward

    if node.parent is not None:
        backup(node.parent, -reward)


def evaluate_game_state(game_state):
    game = game_state.copy()
    scoring = game.get_other_player(game.get_current_player())
    while not game.is_terminal():
        game.play_move(random.choice(game.list_moves()))
    return game.get_reward(scoring)


def tree_search_one_step(root, c_p):
    current = root
    while current.is_expanded() and not current.game_state.is_terminal():
        current = current.best_child(c_p)
        # print(current)

    # # expand
    if not current.is_expanded():
        current = current.expand_random_child()

    # simulate
    score = evaluate_game_state(current.game_state)

    # backup
    backup(current, score)

def get_configurable_options():
    return ["steps", "cp"]

def agent(obs, conf, settings=None):
    if settings is None:
        settings = {
            "steps": 100,
            "cp": 1.0
        }
    game = ConnectFour(board=obs.board, columns=conf.columns, rows=conf.rows, inarow=conf.inarow, mark=obs.mark)
    root = Node(game_state=game)

    for _ in range(settings["steps"]):
        tree_search_one_step(root, settings["cp"])

    current = root.best_child(0.0)
    return current.move


def play_game_kaggle(obs, conf):
    start = time.time()

    game = ConnectFour(board=obs.board, columns=conf.columns, rows=conf.rows, inarow=conf.inarow, mark=obs.mark)
    root = Node(game_state=game)

    available_time = conf.timeout - 0.3
    while time.time() - start < available_time:
        tree_search_one_step(root, 1.0)

    current = root.best_child(0.0)
    return current.move

