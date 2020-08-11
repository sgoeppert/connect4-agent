import math
import random
import time
import numpy as np


class Node:
    def __init__(self, game_state, move=None, parent=None, move_name=None):
        self.game_state = game_state

        self.move = move
        self.parent = parent
        self.move_name = move_name

        self.possible_actions = game_state.list_moves()

        self.children = []
        self.total_value = 0
        self.number_visits = 0

        self.amaf = 0
        self.ma = 0

    def is_expanded(self):
        return len(self.possible_actions) == 0

    def Q(self):
        return self.total_value / max(1, self.number_visits)

    def QRave(self):
        return self.amaf / max(1, self.ma)

    def explo(self, log_parent_visits):
        if self.number_visits == 0:
            return 10
        return math.sqrt(log_parent_visits / max(1, self.number_visits))

    def best_child_rave(self, c_p, beta):
        log_p_visits = math.log(self.number_visits)
        return max(self.children, key=lambda c: (1 - beta) * c.Q() + beta * c.QRave() + c_p * c.explo(log_p_visits))

    # regular best_child
    # def best_child(self, c_p):
    #     n_p = 2 * math.log(self.number_visits)
    #     return max(self.children, key=lambda c: c.Q() + c_p * math.sqrt(n_p/max(1, c.number_visits)))

    # regular expansion
    # def expand_random_child(self):
    #     m = random.choice(self.possible_actions)
    #     state = self.game_state.copy()
    #     state.play_move(m)
    #     ch = Node(game_state=state, move=m, parent=self)
    #     self.children.append(ch)
    #     self.possible_actions.remove(m)
    #     return ch

    def expand_all_children(self, move_counts):
        for move in self.possible_actions:
            state = self.game_state.copy()
            state.play_move(move)
            move_name = 10 * (move_counts[move] * len(move_counts) + move) + self.game_state.get_current_player()
            self.children.append(Node(game_state=state, move=move, parent=self, move_name=move_name))
        self.possible_actions = []

    def __repr__(self):
        return f"Node({self.move}, {self.number_visits}, {self.total_value}, move_name={self.move_name}, amaf={self.amaf}, ma={self.ma})"


# regular backup
# def backup(node, reward):
#     node.number_visits += 1
#     node.total_value += reward
#
#     if node.parent is not None:
#         backup(node.parent, -reward)


def backup_rave(node, reward, moves):
    node.number_visits += 1
    node.total_value += reward

    for c in node.children:
        if c.move_name in moves:
            c.amaf += -reward
            c.ma += 1

    if node.parent is not None:
        backup_rave(node.parent, -reward, moves)


# regular evaluate_game_state
# def evaluate_game_state(game_state):
#     game = game_state.copy()
#     scoring = game.get_other_player(game.get_current_player())
#     while not game.is_terminal():
#         game.play_move(random.choice(game.list_moves()))
#     return game.get_reward(scoring)


def evaluate_game_state_rave(game_state, moves, move_counts):
    game = game_state.copy()
    scoring = game.get_other_player(game.get_current_player())
    while not game.is_terminal():
        m = random.choice(game.list_moves())
        move_name = 10 * (move_counts[m] * len(move_counts) + m) + game.get_current_player()
        move_counts[m] += 1
        moves.append(move_name)

        game.play_move(m)
    return game.get_reward(scoring)


def init_move_counts(board, rows, cols):
    board = np.flip(np.array(board).reshape(rows, cols), axis=0)
    move_counts = [0] * 7  # assume all columns are empty
    for c in range(cols):
        for r in range(rows):
            if board[r, c] != 0:
                move_counts[c] = r + 1
    return move_counts


def tree_search_one_step(root, c_p, beta):
    current = root

    action_space = root.game_state.get_action_space()
    gs = root.game_state
    board = gs.board
    move_counts = init_move_counts(board, gs.rows, gs.cols)
    moves = []

    while current.is_expanded() and not current.game_state.is_terminal():
        p = current.game_state.get_current_player()
        current = current.best_child_rave(c_p, beta)
        m = current.move
        move_name = 10 * (move_counts[m] * action_space + m) + p
        move_counts[m] += 1
        moves.append(move_name)

    # expand
    # create all children so we can update their rave values for an informed first selection
    if not current.is_expanded():
        current.expand_all_children(move_counts)

    # simulate
    score = evaluate_game_state_rave(current.game_state, moves, move_counts)

    # backup
    backup_rave(current, score, moves)


def get_configurable_options():
    return ["steps", "cp", "beta"]


def agent(obs, conf, settings=None):
    if settings is None:
        settings = {
            "steps": 100,
            "cp": 1.0,
            "beta": 0.8
        }
    game = ConnectFour(board=obs.board, columns=conf.columns, rows=conf.rows, inarow=conf.inarow, mark=obs.mark)
    root = Node(game_state=game)

    for _ in range(settings["steps"]):
        tree_search_one_step(root, settings["cp"], settings["beta"])

    current = root.best_child_rave(0.0, settings["beta"])
    return current.move


def play_game_kaggle(obs, conf):
    start = time.time()

    cp = 1.0
    beta = 0.8

    game = ConnectFour(board=obs.board, columns=conf.columns, rows=conf.rows, inarow=conf.inarow, mark=obs.mark)
    root = Node(game_state=game)

    available_time = conf.timeout - 0.3
    while time.time() - start < available_time:
        tree_search_one_step(root, cp, beta)

    current = root.best_child_rave(0.0, beta)
    return current.move


if __name__ == "__main__":
    from old.tools import play_game
    from old.mcts import agent as regular_agent
    from old.games.connectfour import ConnectFour

    results = []
    for _ in range(10):
        results.append(play_game(agent, regular_agent))
    for _ in range(10):
        results.append(play_game(regular_agent, agent)[::-1])

    print((1 + np.mean(results, axis=0)) / 2)

    # play_game(agent, agent)

    # g = ConnectFour()
    # root = Node(game_state=g)
    #
    # for _ in range(100):
    #     tree_search_one_step(root, 1.0)
    #
    # print(root.children)
    # print("Best:", root.best_child_rave(0.0))
