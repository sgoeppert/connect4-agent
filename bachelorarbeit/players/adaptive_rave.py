from bachelorarbeit.games import Configuration, ConnectFour, Observation
from bachelorarbeit.players.mcts import MCTSPlayer
from bachelorarbeit.players.rave import RavePlayer
from bachelorarbeit.players.adaptive_playout import AdaptiveEvaluator


class AdaptiveRaveEvaluator(AdaptiveEvaluator):
    def __init__(self, forgetting=False, keep_replies=False):
        """
        :param forgetting:
        :param keep_replies:
        """
        super(AdaptiveRaveEvaluator, self).__init__(forgetting, keep_replies)
        self.moves = []

    def __call__(self, game_state: ConnectFour) -> float:
        game = game_state.copy()
        simulating_player = game.get_current_player()
        scoring = 3 - simulating_player

        last_move = None
        simulation_replies = {}

        while not game.is_terminal():
            move = self.get_next_move(game, last_move)
            game.play_move(move)

            move_name = game.get_move_name(move, played=True)
            self.moves.append(move_name)

            if last_move is not None:
                simulation_replies[last_move] = move_name

            last_move = move_name

        self.memorize(simulation_replies, game.winner)
        return game.get_reward(scoring)

    def reset(self):
        if not self.keep_replies:
            self.replies = {}


class AdaptiveRavePlayer(RavePlayer):
    name = "AdaptiveRavePlayer"

    def __init__(self, forgetting=False, keep_replies=False, exploration_constant=0.25, **kwargs):
        super(AdaptiveRavePlayer, self).__init__(exploration_constant=exploration_constant, **kwargs)
        self.evaluate = AdaptiveRaveEvaluator(forgetting, keep_replies)

    def reset(self, conf: Configuration = None):
        super(AdaptiveRavePlayer, self).reset(conf)
        self.evaluate.reset()


if __name__ == "__main__":
    from bachelorarbeit.selfplay import Arena
    from bachelorarbeit.tools import timer

    # steps = 4000
    # pl = AdaptiveRavePlayer(max_steps=steps, exploration_constant=0.4, alpha=0.5, keep_replies=True)
    # conf = Configuration()
    # game = ConnectFour(columns=conf.columns, rows=conf.rows, inarow=conf.inarow, mark=1)
    # obs = Observation(board=game.board.copy(), mark=game.mark)
    #
    # with timer(f"{steps} steps"):
    #     m = pl.get_move(obs, conf)
    #     print(m)
    #
    # rave_steps = 200
    # regular_steps = 1000
    #
    # arena = Arena(players=(AdaptiveRavePlayer, MCTSPlayer),
    #               constructor_args=(
    #                   {
    #                       "max_steps": rave_steps,
    #                       "exploration_constant": 0.4,
    #                       "keep_replies": True
    #                   },
    #                   {"max_steps": regular_steps, "exploration_constant": 1.0}),
    #               num_games=400,
    #               num_processes=8
    #               )
    # arena.run_game_mp(show_progress_bar=True)


    from bachelorarbeit.selfplay import Arena

    test_steps = 563
    regular_steps = 1000

    arena = Arena(players=(AdaptiveRavePlayer, MCTSPlayer),
                  constructor_args=(
                      {
                          "max_steps": test_steps,
                          "exploration_constant": 0.4,
                          "alpha": 0.5,
                          "keep_replies": True,
                      },
                      # {"max_steps": regular_steps, "exploration_constant": 0.4, "alpha": 0.5, "keep_replies": True}),
                      {
                          "max_steps": regular_steps
                      }),
                  num_games=500,
                  num_processes=8
                  )
    arena.run_game_mp(show_progress_bar=True)