from bachelorarbeit.players.adaptive_network_player import AdaptiveNetworkPlayer
from bachelorarbeit.players.adaptive_rave_network_player import AdaptiveRaveNetworkPlayer
from bachelorarbeit.players.mcts import MCTSPlayer
from bachelorarbeit.players.network_player import NetworkPlayer
from bachelorarbeit.players.rave import RavePlayer
from bachelorarbeit.players.scorebounded import ScoreboundedPlayer
from bachelorarbeit.players.transposition import TranspositionPlayer
from bachelorarbeit.players.adaptive_playout import AdaptivePlayoutPlayer
from bachelorarbeit.players.adaptive_rave import AdaptiveRavePlayer
from bachelorarbeit.tuner import create_tuner, load_tuner, Parametrization
from bachelorarbeit.tools import get_range

CHECKPOINT_INTERVAL = 250


if __name__ == "__main__":
    print("Tuning parameters")
    # OPPONENT_STEPS = 1000
    # print("Running Network Search")
    # nn_params = Parametrization()
    # nn_params.add_default_option("exploration_constant", 0.8)
    # nn_params.add_default_option("network_weight", 0.5)
    # nn_params.choice("max_steps", [300, 500, 800], default=300)
    # nn_params.choice("network_weight", [0.5, 0.8, 0.9], default=0.5)
    # tuner = create_tuner(NetworkPlayer, nn_params,
    #                      checkpoint_interval=250,
    #                      opponent_config={"max_steps": OPPONENT_STEPS},
    #                      exploration=2.0)  # explore options more evenly
    # tuner = load_tuner("NetworkPlayer")
    # tuner.search(2000)

    # print("Running Transpos Search")
    # transpos_params = Parametrization()
    # transpos_params.choice("exploration_constant", get_range(1.0, 5), default=1.0)
    # transpos_params.choice("uct_method", ["default", "UCT1", "UCT2", "UCT3"], default="default")
    # transpos_params.boolean("with_symmetry", default=False)
    # tuner = create_tuner(TranspositionPlayer, transpos_params, checkpoint_interval=CHECKPOINT_INTERVAL)
    # tuner = load_tuner("TranspositionPlayer")
    # tuner.exploration = 1.0
    # tuner.search(10000)

    # print("Running AdaptivePlayout Search")
    # adap_params = Parametrization()
    # adap_params.choice("exploration_constant", get_range(0.8, 5, step=0.1), default=0.8)
    # adap_params.boolean("keep_replies", default=False)
    # adap_params.boolean("forgetting", default=False)
    # tuner = create_tuner(AdaptivePlayoutPlayer, adap_params, checkpoint_interval=CHECKPOINT_INTERVAL)
    # tuner.search(10000)


    # print("Running Rave Search")
    # rave_params = Parametrization()
    # rave_params.choice("exploration_constant", get_range(0.4, 5, step=0.1), default=0.4)
    # rave_params.add_default_option("alpha", None)
    # # rave_params.choice("alpha", [None, 0.3, 0.5, 0.8], default=None)
    # # rave_params.xor(("k", [10, 100, 200]), ("alpha", [0.3, 0.5, 0.8]), default=(("k", 10), ("alpha", None)))
    # tuner = create_tuner(RavePlayer, rave_params, checkpoint_interval=CHECKPOINT_INTERVAL, exploration=2.5)
    # # tuner = load_tuner("RavePlayer")
    # tuner.search(2000)
    #
    # print("Running AdaptiveRave Search")
    # adarave_params = Parametrization()
    # adarave_params.choice("exploration_constant", get_range(0.4, 5, step=0.1), default=0.4)
    # adarave_params.add_default_option("alpha", None)
    # # adarave_params.choice("alpha", [None, 0.3, 0.5, 0.8], default=None)
    # # adarave_params.xor(("k", [10, 100, 200]), ("alpha", [0.3, 0.5, 0.8]), default=(("k", 10), ("alpha", None)))
    # adarave_params.add_default_option("keep_replies", False)
    # adarave_params.add_default_option("forgetting", False)
    # # adarave_params.boolean("keep_replies", default=False)
    # # adarave_params.boolean("forgetting", default=False)
    # tuner = create_tuner(AdaptiveRavePlayer, adarave_params, checkpoint_interval=CHECKPOINT_INTERVAL, exploration=2.0)
    # # tuner = load_tuner("AdaptiveRavePlayer")
    # tuner.search(2000)

    # print("Running MCTS Search")
    # mcts_params = Parametrization()
    # mcts_params.choice("exploration_constant", get_range(1.0, 9), default=1.0)
    # tuner = create_tuner(MCTSPlayer, mcts_params, checkpoint_interval=CHECKPOINT_INTERVAL)
    # tuner = load_tuner("MCTSPlayer")
    # tuner.search(4000)

    # print("Running Scorebounded Search")
    # scorebound_params = Parametrization()
    # scorebound_params.choice("exploration_constant", [0.9, 1.0], default=0.9)
    # scorebound_params.choice("cut_delta", [-0.2, 0.0, 0.2], default=0.0)
    # scorebound_params.choice("cut_gamma", [-0.2, 0.0, 0.2], default=0.0)
    # tuner = create_tuner(ScoreboundedPlayer, scorebound_params, checkpoint_interval=CHECKPOINT_INTERVAL)
    # tuner = load_tuner("ScoreboundedPlayer")
    # tuner.search(2000)

    # print("Running AdaptivePlayout Step Search")
    # adap_params = Parametrization()
    # adap_params.add_default_option("exploration_constant", 0.8)
    # adap_params.add_default_option("keep_replies", True)
    # adap_params.add_default_option("forgetting", False)
    # adap_params.choice("max_steps", [100, 200, 400, 600, 800], default=100)
    # tuner = create_tuner(AdaptivePlayoutPlayer, adap_params,
    #                      checkpoint_interval=CHECKPOINT_INTERVAL,
    #                      exploration=2.0)  # explore options more evenly
    # tuner.search(4000)


    print("Running AdaptivePlayout Search")
    adap_params = Parametrization()
    adap_params.choice("exploration_constant", get_range(1.0, 5), default=1.0)
    adap_params.add_default_option("keep_replies", False)
    adap_params.add_default_option("forgetting", False)
    tuner = create_tuner(AdaptivePlayoutPlayer, adap_params,
                         checkpoint_interval=CHECKPOINT_INTERVAL,
                         exploration=2.0,
                         name="NoKeepNoForgetLGR")
    tuner.search(2000)


    adap_params = Parametrization()
    adap_params.choice("exploration_constant", get_range(1.0, 5), default=1.0)
    adap_params.add_default_option("keep_replies", True)
    adap_params.add_default_option("forgetting", False)
    tuner = create_tuner(AdaptivePlayoutPlayer, adap_params,
                         checkpoint_interval=CHECKPOINT_INTERVAL,
                         exploration=2.0,
                         name="KeepNoForgetLGR")
    tuner.search(2000)


    adap_params = Parametrization()
    adap_params.choice("exploration_constant", get_range(1.0, 5), default=1.0)
    adap_params.add_default_option("keep_replies", True)
    adap_params.add_default_option("forgetting", True)
    tuner = create_tuner(AdaptivePlayoutPlayer, adap_params,
                         checkpoint_interval=CHECKPOINT_INTERVAL,
                         exploration=2.0,
                         name="KeepForgetLGR")
    tuner.search(2000)


    adap_params = Parametrization()
    adap_params.choice("exploration_constant", get_range(1.0, 5), default=1.0)
    adap_params.add_default_option("keep_replies", False)
    adap_params.add_default_option("forgetting", True)
    tuner = create_tuner(AdaptivePlayoutPlayer, adap_params,
                         checkpoint_interval=CHECKPOINT_INTERVAL,
                         exploration=2.0,
                         name="NoKeepForgetLGR")
    tuner.search(2000)


    # print("Running AdaptiveNetwork Search")
    # adanet_params = Parametrization()
    # adanet_params.choice("exploration_constant", [0.5, 0.8, 1.0], default=0.8)
    # adanet_params.choice("network_weight", [0.2, 0.5, 0.8], default=0.5)
    # adanet_params.add_default_option("keep_replies", True)
    # adanet_params.add_default_option("forgetting", False)
    # adanet_params.add_default_option("max_steps", 200)
    # # adanet_params.choice("max_steps", [200, 300, 500], default=200)
    # tuner = create_tuner(AdaptiveNetworkPlayer, adanet_params,
    #                      checkpoint_interval=250,
    #                      exploration=2.0,  # explore options more evenly
    #                      opponent_config={"max_steps": 1000})
    # tuner.search(2000)


    # print("Running AdaptiveRaveNetwork Step Search")
    # adanet_params = Parametrization()
    # adanet_params.choice("exploration_constant", get_range(0.3, 5), default=0.3)
    # adanet_params.xor(("k", [1, 10, 100]), ("alpha", [0.5, 0.8, 0.9]), default=(("k", 1), ("alpha", None)))
    # adanet_params.add_default_option("keep_replies", True)
    # adanet_params.add_default_option("forgetting", False)
    # adanet_params.add_default_option("max_steps", 200)
    # tuner = create_tuner(AdaptiveRaveNetworkPlayer, adanet_params,
    #                      checkpoint_interval=250,
    #                      opponent_config={"max_steps": 1000})
    # tuner.search(6000)
