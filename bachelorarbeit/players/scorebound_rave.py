from typing import List, Tuple
import math

from bachelorarbeit.players.rave import RaveNode, RavePlayer

class ScoreBoundedRaveNode(RaveNode):
    def __init__(self, delta: float = 0.0, gamma: float = 0.0, max_node: bool = True, *args, **kwargs):
        super(ScoreBoundedRaveNode, self).__init__(*args, **kwargs)
        self.pess = -1
        self.opti = 1
        self.delta = delta
        self.gamma = gamma
        self.is_max_node = max_node


    def best_child(self, C_p: float = 1.0, b: float = 0.0, alpha: float = None) -> Tuple["RaveNode", int]:
        n_p = math.log(self.number_visits)

        def rave_score(child: ScoreBoundedRaveNode):
            if alpha is not None:
                beta = alpha
            else:
                beta = child.beta(b)
            return (1 - beta) * child.Q() + beta * child.QRave()

        def scorebound_score(child: ScoreBoundedRaveNode):
            if self.is_max_node:
                return rave_score(child) + self.gamma * child.pess + self.delta * child.opti
            else:
                return rave_score(child) - self.delta * child.pess - self.gamma * child.opti

        def child_value(_, child):
            return scorebound_score(child) + C_p * math.sqrt(n_p / child.number_visits)

        children = self.children.copy()

        for move, _child in self.children.items():
            if len(children) > 1:
                if self.is_max_node and _child.opti <= self.pess:
                    del(children[move])
                elif not self.is_max_node and _child.pess >= self.opti:
                    del(children[move])

        m, c = max(children.items(), key=lambda ch: child_value(*ch))
        return c, m

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

    def expand_one_child_rave(self, moves: List[int]) -> "ScoreBoundedRaveNode":
        child = super(ScoreBoundedRaveNode, self).expand_one_child_rave(moves)
        child.is_max_node = not self.is_max_node
        return child

    def __repr__(self):
        maxnode = " Min"
        if self.is_max_node:
            maxnode = " Max"

        descr = f"SBRaveNode(Q: {self.Q()}, N: {self.number_visits}, Q_Rave: {self.rave_score}, N_Rave: {self.rave_count}," \
               f" pess: {self.pess}, opti: {self.opti}{maxnode})"

        return descr

class ScoreBoundedRavePlayer(RavePlayer):
    name = "ScoreBoundedRavePlayer"

    def __init__(self, delta: float = 0.0, gamma:float = 0.0, *args, **kwargs):
        super(ScoreBoundedRavePlayer, self).__init__(*args, **kwargs)
        self.delta = delta
        self.gamma = gamma

    def prop_pess(self, s: ScoreBoundedRaveNode):
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

    def prop_opti(self, s: ScoreBoundedRaveNode):
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


    def backup_rave(self, node: ScoreBoundedRaveNode, reward: float, moves: List[int]):

        if node.game_state.is_terminal():
            bound_score = -reward if node.is_max_node else reward
            node.opti = bound_score
            node.pess = bound_score

            self.prop_pess(node)
            self.prop_opti(node)

        move_set = set(moves)
        current = node
        while current is not None:
            current.number_visits += 1
            current.average_value += (reward - current.average_value) / current.number_visits

            # Wenn eines der Kinder dieses Knotens über einen in der Simulation gemachten Spielzug
            # erreicht werden könnte, aktualisiere seine Rave Statistik
            for mov, child in current.rave_children.items():
                if mov in move_set:
                    child.increment_rave_visit_and_add_reward(-reward)

            reward = -reward
            current = current.parent


    def init_root_node(self, root_game):
        return ScoreBoundedRaveNode(game_state=root_game, delta=self.delta, gamma=self.gamma, max_node=True)

    def perform_search(self, root):
        best = super(ScoreBoundedRavePlayer, self).perform_search(root)

        # print(root)
        # print("Children:", *list(root.children.items()), sep="\n\t")

        return best

if __name__ == "__main__":
    from bachelorarbeit.games import Observation, Configuration, ConnectFour
    from bachelorarbeit.tools import timer

    steps = 100
    pl = ScoreBoundedRavePlayer(max_steps=steps, exploration_constant=0.2, b=0.0, alpha=0.99)
    conf = Configuration()
    game = ConnectFour(columns=conf.columns, rows=conf.rows, inarow=conf.inarow, mark=1)
    obs = Observation(board=game.board.copy(), mark=game.mark)

    with timer(f"{steps} steps"):
        m = pl.get_move(obs, conf)
        print(m)
