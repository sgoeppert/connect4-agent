import random
import math
import time
import gc

from typing import List, Tuple

K = 200
global_start = time.time()

def get_timer():
    last = time.time()
    def timer(label, reset=False):
        nonlocal last
        if reset:
            last = time.time()
            return
        now = time.time()
        elapsed = now - last
        print(f"{label} - {elapsed}")
        last = now
    return timer

class ConnectFour:
    def __init__(
            self,
            columns: int = 7,
            rows: int = 6,
            inarow: int = 4,
            mark: int = 1,
            board: List[int] = None
    ):
        if board is not None:
            self.board = board
        else:
            self.board = [0] * rows * columns
        self.cols = columns
        self.rows = rows
        self.mark = mark
        self.inarow = inarow
        self.stones_per_column = [0] * columns
        self.finished = False
        self.winner = None

    def is_terminal(self) -> bool:
        return self.finished

    def list_moves(self) -> List[int]:
        return [c for c in range(self.cols) if self.board[c] == 0]

    def get_current_player(self) -> int:
        return self.mark

    def get_other_player(self, player: int) -> int:
        return 3 - player

    def play_move(self, col: int) -> "ConnectFour":
        row = max([r for r in range(self.rows) if self.board[col + (r * self.cols)] == 0])
        self.board[col + (row * self.cols)] = self.mark
        self.stones_per_column[col] += 1

        if self.is_win(col, self.mark, has_played=True):
            self.finished = True
            self.winner = self.mark
        elif self.is_tie():
            self.finished = True
            self.winner = None

        self.mark = 3 - self.mark
        return self

    def is_win(self, column: int, mark: int, has_played: bool = False) -> bool:
        columns = self.cols
        rows = self.rows
        inarow = self.inarow - 1
        row = (
            min([r for r in range(rows) if self.board[column + (r * columns)] == mark])
            if has_played
            else max([r for r in range(rows) if self.board[column + (r * columns)] == 0])
        )

        def count(offset_row: int, offset_column: int) -> int:
            for i in range(1, inarow + 1):
                r = row + offset_row * i
                c = column + offset_column * i
                if (
                        r < 0
                        or r >= rows
                        or c < 0
                        or c >= columns
                        or self.board[c + (r * columns)] != mark
                ):
                    return i - 1
            return inarow

        return (
                count(1, 0) >= inarow  # vertical.
                or (count(0, 1) + count(0, -1)) >= inarow  # horizontal.
                or (count(-1, -1) + count(1, 1)) >= inarow  # top left diagonal.
                or (count(-1, 1) + count(1, -1)) >= inarow  # top right diagonal.
        )

    def is_tie(self) -> bool:
        return all(mark != 0 for mark in self.board)

    def get_reward(self, mark: int) -> int:
        if mark == self.winner:
            return 1
        elif self.winner is None:
            return 0
        else:
            return -1


    def get_move_name(self, column: int, played: bool = False) -> int:
        stones_in_col = self.stones_per_column[column]
        player = self.get_current_player()
        if played:
            stones_in_col -= 1
            player = self.get_other_player(player)
        return 10 * (stones_in_col * self.cols + column) + player

    def copy(self) -> "ConnectFour":
        cp = ConnectFour(board=self.board[:],
                         columns=self.cols,
                         rows=self.rows,
                         mark=self.mark,
                         inarow=self.inarow)
        cp.finished = self.finished
        cp.winner = self.winner
        cp.stones_per_column = self.stones_per_column[:]
        return cp


class Node:
    def __init__(self, game_state: ConnectFour, parent: "Node" = None):
        self.average_value = 0  # Q(v)
        self.total_value = 0  # V(v)
        self.number_visits = 0  # N(v)
        self.children = {}  # C(v), Kinder des Knotens

        self.parent = parent  # der direkte Elternknoten

        self.game_state = game_state  # der Spielzustand in diesem Knoten
        self.possible_moves = game_state.list_moves()  # Aktionen der noch nicht erforschten Kindknoten
        self.expanded = False  # ob der Knoten vollständig expandiert ist

    def Q(self) -> float:
        return self.average_value

    def best_child(self, C_p: float = 1.0) -> "Node":
        n_p = math.log(self.number_visits)

        def UCT(child: Node):
            """
            Berechnet den UCT Wert UCT = Q(v') + C_p * sqrt(ln(N(v))/N(v'))
            :param child: Knoten v'
            :return:
            """
            return child.Q() + C_p * math.sqrt(n_p / child.number_visits)

        _, c = max(self.children.items(), key=lambda entry: UCT(entry[1]))
        return c

    def increment_visit_and_add_reward(self, reward: float):
        self.number_visits += 1
        self.average_value += (reward - self.average_value) / self.number_visits

    def is_expanded(self) -> bool:
        return self.expanded

    def expand_one_child(self) -> "Node":
        node_class = type(self)

        move = random.choice(self.possible_moves)
        self.children[move] = node_class(game_state=self.game_state.copy().play_move(move), parent=self)
        self.possible_moves.remove(move)
        if len(self.possible_moves) == 0:
            self.expanded = True

        return self.children[move]

    def remove_parent(self, player):
        self.parent = None

    def __repr__(self):
        return f"Node(Q:{self.Q()}, N:{self.number_visits})"


class MCTSPlayer:
    name = "MCTSPlayer"

    def __init__(
            self,
            exploration_constant: float = 0.8,
            max_steps: int = 1000,
            keep_tree: bool = False,
            time_buffer_pct: float = 0.05
    ):
        self.replies = {}

        # UCT Exploration Konstante Cp
        self.exploration_constant = exploration_constant
        self.keep_tree = keep_tree  # ob der Baum zwischen Zügen erhalten bleibt
        self.root = None  # die Wurzel des Baumes

        # Limitiert die Ausführungszeit
        self.start_time = 0
        self.time_limit = 0
        self.time_buffer_pct = time_buffer_pct

        # Verbrauchte und maximale Schritte pro Zug
        self.max_steps = max_steps
        self.steps_taken = 0
        self.initialized = False

    def reset(self, conf = None, start_time = time.time()):
        self.steps_taken = 0
        self.start_time = start_time

        if not self.keep_tree:
            self.root = None

        if conf is not None:
            if self.initialized:
                limit = conf.actTimeout
            else:
                limit = conf.agentTimeout + conf.actTimeout

            if limit > 0:
                self.time_limit = limit * (1-self.time_buffer_pct)

    def has_resources(self) -> bool:
        if self.time_limit > 0:
            return time.time() - self.start_time < self.time_limit
        else:
            self.steps_taken += 1
            return self.steps_taken <= self.max_steps

    def evaluate_game_state(self, game_state: ConnectFour) -> float:
        game = game_state.copy()
        # Die Bewertung Geschieht aus Sicht des Spielers, der uns in diesen Zustand geführt hat, darum muss der Spieler
        # geholt werden, der zuletzt gezogen hat, nicht der der gerade an der Reihe ist.
        simulating_player = game.get_current_player()
        scoring = 3 - simulating_player

        last_move = None
        simulation_replies = {}

        while not game.is_terminal():
            move = None
            legal_moves = game.list_moves()
            if last_move is not None and last_move in self.replies:
                reply = self.replies[last_move]
                reply_move = (reply // 10) % game.cols
                mname = game.get_move_name(reply_move, played=False)
                if reply == mname:
                    move = reply_move

            if move not in legal_moves:
                move = random.choice(legal_moves)

            game.play_move(move)
            move_name = game.get_move_name(move, played=True)
            if last_move is not None:
                simulation_replies[last_move] = move_name
            
            last_move = move_name

        winner = game.winner
        if winner is not None:
            loser = 3 - winner
            for move, reply in simulation_replies.items():
                replying_to = move % 10
                if replying_to == loser:
                    self.replies[move] = reply


        return game.get_reward(scoring)

    def tree_policy(self, root: Node) -> Node:
        current = root
        while not current.game_state.is_terminal():
            if current.is_expanded():
                current = current.best_child(self.exploration_constant)
            else:
                return current.expand_one_child()
        return current

    def backup(self, node: Node, reward: float):
        current = node
        while current is not None:
            current.increment_visit_and_add_reward(reward)
            reward = -reward
            current = current.parent

    def best_move(self, node: Node) -> int:
        move, n = max(node.children.items(), key=lambda c: c[1].Q())
        return move

    def determine_opponent_move(self, new_board: List[int], old_board: List[int], columns: int = 7) -> int:
        i = 0
        for new_s, old_s in zip(new_board, old_board):
            if new_s != old_s:
                return i % columns
            i += 1
        return -1

    def _restore_root(self, observation, configuration):
        root = None
        # if we're keeping the tree and have a stored node, try to determine the opponents move and apply that move
        # the resulting node is out starting root-node
        if self.keep_tree and self.root is not None:
            root = self.root
            opp_move = self.determine_opponent_move(observation.board, self.root.game_state.board, configuration.columns)
            if opp_move in root.children:
                root = root.children[opp_move]
            else:
                root = None
        return root

    def _store_root(self, new_root):
        if self.keep_tree:
            self.root = new_root

    def init_root_node(self, root_game):
        return Node(root_game)

    def perform_search(self, root):
        steps = 0
        while self.has_resources():
            leaf = self.tree_policy(root)
            reward = self.evaluate_game_state(leaf.game_state)
            self.backup(leaf, reward)
            steps += 1
        print(steps)
        return self.best_move(root)

    def get_move(self, observation, conf, start_time = time.time()) -> int:
        _t = get_timer()
        _start = start_time
        self.reset(conf, _start)
        _t("reset done")
        # load the root if it was persisted
        root = self._restore_root(observation, conf)
        _t("root restored")
        # if no root could be determined, create a new tree from scratch
        if root is None:
            root_game = ConnectFour(columns=conf.columns, rows=conf.rows, inarow=conf.inarow,
                                    mark=observation.mark, board=observation.board)
            root = self.init_root_node(root_game)

        # run the search
        best = self.perform_search(root)
        _t("perform search")

        self._store_root(root.children[best])
        _t("store root")
        _time_taken = time.time() - _start
        print("Elapsed: ",_time_taken)

        return best

class RaveNode(Node):
    def __init__(self, *args, **kwargs):
        super(RaveNode, self).__init__(*args, **kwargs)
        self.rave_count = 0
        self.rave_score = 0
        self.rave_children: Dict[int, "RaveNode"] = {}

        self.beta = 1.0

    def QRave(self) -> float:
        return self.rave_score

    def exploration(self, parent_visits):
        return math.sqrt(parent_visits / self.number_visits)

    def best_child(self, C_p: float = 1.0) -> Tuple["RaveNode", int]:
        n_p = math.log(self.number_visits)

        def UCT_Rave(child: RaveNode):
            beta = child.beta
            return (1 - beta) * child.Q() + beta * child.QRave() \
                   + C_p * math.sqrt(n_p / child.number_visits)

        m, c = max(self.children.items(), key=lambda c: UCT_Rave(c[1]))
        return c, m

    def expand_one_child_rave(self, moves: List[int]) -> "RaveNode":
        nodeclass = type(self)
        move = random.choice(self.possible_moves)  # wähle zufälligen Zug
        child_state = self.game_state.copy().play_move(move)
        move_name = child_state.get_move_name(move, played=True)
        moves.append(move_name)
        self.children[move] = nodeclass(game_state=child_state, parent=self)
        self.rave_children[move_name] = self.children[move]
        self.possible_moves.remove(move)
        if len(self.possible_moves) == 0:
            self.expanded = True

        return self.children[move]

    def increment_rave_visit_and_add_reward(self, reward):
        self.rave_count += 1
        self.rave_score += (reward - self.rave_score) / self.rave_count

    def __repr__(self):
        return f"RaveNode(n: {self.number_visits}, v: {self.average_value}, rave_n: {self.rave_count}, rave_v: {self.rave_score})"


class RavePlayer(MCTSPlayer):
    name = "RavePlayer"

    def __init__(self, *args, **kwargs):
        super(RavePlayer, self).__init__(*args, **kwargs)

    def tree_policy_rave(self, root: "RaveNode", moves) -> "RaveNode":
        current = root

        while not current.game_state.is_terminal():
            if current.is_expanded():
                current, m = current.best_child(self.exploration_constant)
                # Hole den eindeutigen Namen des Spielzugs und merke ihn
                move_name = current.game_state.get_move_name(m, played=True)
                moves.append(move_name)
            else:
                return current.expand_one_child_rave(moves)

        return current

    def evaluate_game_state_rave(self, game_state: ConnectFour, moves: List[int]) -> float:
        game = game_state.copy()
        # Die Bewertung Geschieht aus Sicht des Spielers, der uns in diesen Zustand geführt hat, darum muss der Spieler
        # geholt werden, der zuletzt gezogen hat, nicht der der gerade an der Reihe ist.
        simulating_player = game.get_current_player()
        scoring = 3 - simulating_player

        last_move = None
        simulation_replies = {}

        while not game.is_terminal():
            move = None
            legal_moves = game.list_moves()
            if last_move is not None and last_move in self.replies:
                reply = self.replies[last_move]
                reply_move = (reply // 10) % game.cols
                mname = game.get_move_name(reply_move, played=False)
                if reply == mname:
                    move = reply_move

            if move not in legal_moves:
                move = random.choice(legal_moves)

            game.play_move(move)
            move_name = game.get_move_name(move, played=True)
            
            moves.append(move_name)

            if last_move is not None:
                simulation_replies[last_move] = move_name
            
            last_move = move_name

        winner = game.winner
        if winner is not None:
            loser = 3 - winner
            for move, reply in simulation_replies.items():
                replying_to = move % 10
                if replying_to == loser:
                    self.replies[move] = reply


        return game.get_reward(scoring)


    def backup_rave(self, node: RaveNode, reward: float, moves: List[int]):
        move_set = set(moves)
        current = node
        while current is not None:
            current.increment_visit_and_add_reward(reward)
            current.beta = K / (K + current.number_visits)
            # Wenn eines der Kinder dieses Knotens über einen in der Simulation gemachten Spielzug
            # erreicht werden könnte, aktualisiere seine Rave Statistik
            for mov, child in current.rave_children.items():
                if mov in move_set:
                    child.increment_rave_visit_and_add_reward(-reward)

            reward = -reward
            current = current.parent

    def perform_search(self, root):
        start = time.time()
        steps = 0
        while self.has_resources():
            moves = []  # die Liste aller gemachten Spielzüge in dieser Iteration
            leaf = self.tree_policy_rave(root, moves)
            reward = self.evaluate_game_state_rave(leaf.game_state, moves)
            self.backup_rave(leaf, reward, moves)
            steps += 1

        print(steps)
        return self.best_move(root)

    def init_root_node(self, root_game):
        return RaveNode(game_state=root_game)

    def best_move(self, node: RaveNode) -> int:
        n, move = node.best_child(C_p=0)
        # move, n = max(node.children.items(), key=lambda c: (1 - c[1].beta(0)) * c[1].Q() + c[1].beta(0) * c[1].QRave())
        return move



RPLAYER = None

def agent(observation, configuration):
    global RPLAYER, global_start

    main_timer = get_timer()

    enabled = gc.isenabled()
    gc.disable()
    start_time = time.time()

    if RPLAYER is None:
        start_time = global_start
        RPLAYER = RavePlayer(keep_tree=True, exploration_constant=0.25)
    main_timer("init")

    move = RPLAYER.get_move(observation, configuration, start_time)
    RPLAYER.initialized = True
    main_timer("get_move")
    if enabled:
        gc.enable()
    return move