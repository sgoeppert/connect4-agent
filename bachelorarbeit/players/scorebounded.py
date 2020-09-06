import math

from bachelorarbeit.players.mcts import Node, MCTSPlayer


class ScoreboundedNode(Node):
    def __init__(self,
                 cut_delta: float = 0.0,
                 cut_gamma: float = 0.0,
                 *args, **kwargs):
        super(ScoreboundedNode, self).__init__(*args, **kwargs)
        self.pess = -1
        self.opti = 1
        self.cut_delta = cut_delta
        self.cut_gamma = cut_gamma

        if self.parent:
            self.is_max_node = not self.parent.is_max_node
            self.cut_delta = self.parent.cut_delta
            self.cut_gamma = self.parent.cut_gamma
        else:
            self.is_max_node = True

    def best_child(self, C_p: float = 1.0) -> "ScoreboundedNode":
        n_p = math.log(self.number_visits)
        parent = self
        gamma = self.cut_gamma
        delta = self.cut_delta

        def score_func(c):
            if parent.is_max_node:
                return c.Q() + gamma * c.pess + delta * c.opti
            else:
                return c.Q() - delta * c.pess - gamma * c.opti

        children = list(self.children.values())
        # Entferne Kindknoten die das Ergebnis nicht verbessern können
        for c in children:
            if len(children) > 1:
                if self.is_max_node and c.opti <= self.pess:
                    children.remove(c)
                elif not self.is_max_node and c.pess >= self.opti:
                    children.remove(c)

        c = max(children, key=lambda c: score_func(c) + C_p * math.sqrt(n_p / c.number_visits))
        return c

    def min_pess_child(self):
        c_pess = [c.pess for c in self.children.values()]
        if not self.is_expanded():
            c_pess.append(-1)

        return min(c_pess)

    def max_opti_child(self):
        c_opti = [c.opti for c in self.children.values()]
        if not self.is_expanded():
            c_opti.append(1)

        return max(c_opti)

    def __repr__(self):
        maxnode = " Min"
        if self.is_max_node:
            maxnode = " Max"

        solved = ""
        if self.pess == self.opti:
            solved = " Solved"
        return f"SBNode(Q: {self.Q()}, N: {self.number_visits}, pess: {self.pess}, opti: {self.opti}{maxnode}{solved})"


class ScoreboundedPlayer(MCTSPlayer):
    name = "ScoreboundedPlayer"

    def __init__(self,
                 cut_delta: float = 0.0,
                 cut_gamma: float = 0.0,
                 *args, **kwargs):
        super(ScoreboundedPlayer, self).__init__(*args, **kwargs)
        self.cut_delta = cut_delta
        self.cut_gamma = cut_gamma

    def prop_pess(self, s: ScoreboundedNode):
        if s.parent:
            n = s.parent
            old_pess = n.pess
            if old_pess < s.pess:
                if n.is_max_node:
                    n.pess = s.pess
                    self.prop_pess(n)
                else:
                    n.pess = n.min_pess_child()
                    if old_pess > n.pess:
                        self.prop_pess(n)

    def prop_opti(self, s: ScoreboundedNode):
        if s.parent:
            n = s.parent
            old_opti = n.opti
            if old_opti > s.opti:
                if n.is_max_node:
                    n.opti = n.max_opti_child()
                    if old_opti > n.opti:
                        self.prop_opti(n)
                else:
                    n.opti = s.opti
                    self.prop_opti(n)

    def backup(self, node: ScoreboundedNode, reward: float):
        if node.game_state.is_terminal():
            bound_score = -reward if node.is_max_node else reward
            node.opti = bound_score
            node.pess = bound_score

            self.prop_pess(node)
            self.prop_opti(node)

        current = node
        while current is not None:
            current.number_visits += 1
            current.average_value += (reward - current.average_value) / current.number_visits
            reward = -reward

            current = current.parent

    def best_move(self, node: Node) -> int:
        move, n = max(node.children.items(), key=lambda c: c[1].Q() + c[1].opti + c[1].pess)
        return move

    def init_root_node(self, root_game):
        return ScoreboundedNode(game_state=root_game, cut_delta=self.cut_delta, cut_gamma=self.cut_gamma)

    def perform_search(self, root):
        while self.has_resources():
            leaf = self.tree_policy(root)
            reward = self.evaluate_game_state(leaf.game_state)
            self.backup(leaf, reward)

        return self.best_move(root)


if __name__ == "__main__":
    from bachelorarbeit.games import Observation, Configuration, ConnectFour
    from bachelorarbeit.tools import timer

    steps = 200
    p = ScoreboundedPlayer(max_steps=steps, exploration_constant=1.0)
    conf = Configuration()
    game = ConnectFour(columns=conf.columns, rows=conf.rows, inarow=conf.inarow, mark=1)
    obs = Observation(board=game.board.copy(), mark=game.mark)

    with timer(f"{steps} steps"):
        m = p.get_move(obs, conf)
    print(m)