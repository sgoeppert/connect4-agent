from typing import List
import math

from bachelorarbeit.players.rave import RaveNode, RavePlayer


class SarsaNode(RaveNode):
    def __init__(self, *args, **kwargs):
        super(SarsaNode, self).__init__(*args, **kwargs)
        # self.v = 0
        self.max_v = 0.01
        self.min_v = -0.01

    def normalizeQ(self):
        max_v = self.parent.max_v
        min_v = self.parent.min_v

        return (self.average_value - min_v) / (max_v - min_v)

    def Q(self):
        return self.normalizeQ()

    def __repr__(self):
        return f"SarsaNode(n: {self.number_visits}, Q: {self.Q()}, rave_n: {self.rave_count}, rave_v: {self.rave_score})"


class SarsaPlayer(RavePlayer):
    name = "SarsaPlayer"

    def __init__(self, gamma: float = 1.0, lamda: float = 1.0, *args, **kwargs):
        super(SarsaPlayer, self).__init__(*args, **kwargs)
        self.gamma = gamma
        self.lamda = lamda

    def backup_sarsa(self, node: SarsaNode, reward: float, moves: List[int], sim_steps: int):
        move_set = set(moves)

        delta_sum = math.pow(self.lamda * self.gamma, sim_steps - 1) * reward

        # print(f"Reward {reward} discounted {delta_sum}")

        current = node
        v_next = 0
        last_v = 0
        while current is not None:
            v_current = current.average_value
            # print("v_next", v_next, "v_current", v_current)
            delta = self.gamma * v_next - v_current
            delta_sum = self.lamda * self.gamma * delta_sum + delta

            # print("delta", delta)
            # print("delta_sum", delta_sum)

            current.number_visits += 1
            current.average_value += delta_sum / current.number_visits

            if last_v > current.max_v:
                current.max_v = last_v

            if last_v < current.min_v:
                current.min_v = last_v

            last_v = current.average_value
            # print(current)

            delta_sum = -delta_sum
            v_next = -v_current
            for mov, child in current.rave_children.items():
                if mov in move_set:
                    child.increment_rave_visit_and_add_reward(-reward)
                    # child.rave_count += 1
                    # child.rave_score += (-reward - child.rave_score) / child.rave_count
            reward = -reward
            current = current.parent

    def init_root_node(self, root_game):
        return SarsaNode(game_state=root_game)

    def perform_search(self, root):
        while self.has_resources():
            moves = []
            move_counts = SarsaPlayer.init_move_counts(root.game_state)

            leaf = self.tree_policy_rave(root, moves, move_counts)
            n_steps = len(moves)
            reward = self.evaluate_game_state_rave(leaf.game_state, moves, move_counts)
            sim_steps = len(moves) - n_steps

            self.backup_sarsa(leaf, reward, moves, sim_steps)

        return self.best_move(root)


if __name__ == "__main__":
    from bachelorarbeit.games import Observation, Configuration, ConnectFour
    from bachelorarbeit.tools import timer

    steps = 2000
    pl = SarsaPlayer(max_steps=steps, exploration_constant=0.9, b=0.01, gamma=1.0, lamda=0.98)
    conf = Configuration()
    game = ConnectFour(columns=conf.columns, rows=conf.rows, inarow=conf.inarow, mark=1)
    obs = Observation(board=game.board.copy(), mark=game.mark)

    with timer(f"{steps} steps"):
        m = pl.get_move(obs, conf)
        print(m)

    pl.max_steps = 200
    for i in range(15):
        game = ConnectFour(columns=conf.columns, rows=conf.rows, inarow=conf.inarow, mark=1)
        print("Testing game play")
        while not game.is_terminal():
            obs = Observation(board=game.board.copy(), mark=game.mark)
            m = pl.get_move(obs, conf)
            game.play_move(m)