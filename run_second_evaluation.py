from bachelorarbeit.mcts import MCTSPlayer
from bachelorarbeit.rave import RavePlayer
from bachelorarbeit.scorebounded import ScoreboundedPlayer
from bachelorarbeit.selfplay import Arena
from bachelorarbeit.tools import run_selfplay_experiment, run_move_evaluation_experiment
from bachelorarbeit.tools import evaluate_against_random, evaluate_against_flat_monte_carlo
from bachelorarbeit.tools import Table, dump_json
from bachelorarbeit.base_players import RandomPlayer, FlatMonteCarlo

import config

import math
from tqdm import tqdm
import itertools
import os
from pathlib import Path
from datetime import datetime

from bachelorarbeit.transposition import TranspositionPlayer

DEBUG = False

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
LOG_FILE = "second_eval_{}.log".format(timestamp)

if DEBUG:
    MAX_STEPS = 20
    NUM_GAMES = 5
    NUM_GAMES_DETAIL = 20
    MOVE_EVAL_REPEATS = 1
    MOVE_EVAL_STEPS = [25, 50]
    INITIAL_CP_VALUES = [1 / math.sqrt(2), 1.0]
    LOG_FILE = "debug_" + LOG_FILE

else:
    MAX_STEPS = 1000
    NUM_GAMES = 500
    NUM_GAMES_DETAIL = 500
    MOVE_EVAL_REPEATS = 10
    MOVE_EVAL_STEPS = [25, 50, 100, 200, 400, 800]
    INITIAL_CP_VALUES = [0.25, 0.5, 1 / math.sqrt(2), 0.85, 1.0, 1.1, 1.2, math.sqrt(2)]


def argmax(l):
    return max(range(len(l)), key=lambda i: l[i])


def get_range(center, num_vals=5, step=0.1):
    vals_left = num_vals // 2
    vals_right = num_vals - vals_left - 1

    values = [center - (i * step) for i in range(vals_left, 0, -1)]
    values += [center + (i * step) for i in range(vals_right + 1)]
    return list(map(lambda v: round(v, 4), values))


def write_log(message):
    print(message)
    fname = Path(config.ROOT_DIR) / LOG_FILE
    with open(fname, "a+") as f:
        f.write(message + "\n")


def explore_parameter_against_fixed_opponent(player, opponent, player_conf, opponent_conf,
                                             values, parameter="exploration_constant",
                                             variable_opponent=False):
    base_setting = player_conf
    write_log("Running explore_parameter for {} Iterations".format(len(values)))

    result_row = []
    for v in tqdm(values):
        settings = base_setting.copy()
        settings[parameter] = v

        comparison = opponent_conf.copy()
        if variable_opponent:
            comparison[parameter] = v

        res = run_selfplay_experiment(
            "",
            players=(player, opponent),
            constructor_args=(settings, comparison),
            num_games=NUM_GAMES
        )
        result_row.append(res["mean"][0])
    return result_row


def run_move_eval(player, base_setting):
    good_scores = []
    perfect_scores = []
    write_log("Running move_eval for {} Iterations".format(len(MOVE_EVAL_STEPS) * MOVE_EVAL_REPEATS))
    for step in tqdm(MOVE_EVAL_STEPS):
        settings = base_setting.copy()
        settings["max_steps"] = step
        _result = run_move_evaluation_experiment(
            "",
            player=player,
            player_config=settings,
            repeats=MOVE_EVAL_REPEATS
            )
        good_scores.append(_result["good_pct"])
        perfect_scores.append(_result["perfect_pct"])

    return good_scores, perfect_scores


def run_flat_mc_experiments():
    base_setting = {"max_steps": MAX_STEPS, "ucb_selection": True}
    opponent = {**base_setting, "exploration_constant": 1.0, "ucb_selection": False}

    """Compare Flat MC with UCB against Random Flat MC"""
    scores = explore_parameter_against_fixed_opponent(
        player=FlatMonteCarlo,
        opponent=FlatMonteCarlo,
        player_conf=base_setting,
        opponent_conf=opponent,
        values=INITIAL_CP_VALUES
    )
    tab = Table()
    tab.label = "flat-mc-ucb-against-random"
    tab.caption = f"Gewinnchance von Flat Monte Carlo mit UCB-Kindauswahl gegen zufällige Flat Monte Carlo Suche " \
                  f"über {NUM_GAMES} Spiele. Test von verschiedenen Parametern $C_p$. Beide Spieler haben {MAX_STEPS} " \
                  f"Schritte Bedenkzeit pro Zug."
    tab.add_row(scores, label="Sieg \\%")
    tab.set_full_col_header(INITIAL_CP_VALUES)
    tab.top_left = "$C_p$"
    tab.write_to_file("v2/1_flat_mc_ucb")

    write_log(f"Best score of Flat MC against Random Flat MC {max(scores)}")

    """Find Cp for Flat MC with UCB against C_p=1.0"""
    opponent = {**base_setting, "exploration_constant": 1.0}
    scores = explore_parameter_against_fixed_opponent(
        player=FlatMonteCarlo,
        opponent=FlatMonteCarlo,
        player_conf=base_setting,
        opponent_conf=opponent,
        values=INITIAL_CP_VALUES
    )
    tab = Table()
    tab.label = "flat-mc-cp-1"
    tab.caption = f"Vergleich der Spielstärke von Flat Monte Carlo in Abhängigkeit von $C_p$ bei {MAX_STEPS} Schritten " \
                  f"pro Spielzug über {NUM_GAMES} Spiele."
    tab.add_row(scores, label="Sieg \\%")
    tab.set_full_col_header(INITIAL_CP_VALUES)
    tab.top_left = "$C_p$"
    tab.write_to_file("v2/2_flat_mc_cp_1")

    best_initial_cp = INITIAL_CP_VALUES[argmax(scores)]
    write_log(f"Best initial Cp for Flat Monte Carlo with UCB {best_initial_cp} with score {max(scores)}")

    """Refine Cp"""
    opponent = {**base_setting, "exploration_constant": best_initial_cp}
    refined_cp_values = get_range(best_initial_cp, 5, 0.05)
    scores = explore_parameter_against_fixed_opponent(
        player=FlatMonteCarlo,
        opponent=FlatMonteCarlo,
        player_conf=base_setting,
        opponent_conf=opponent,
        values=refined_cp_values
    )
    tab = Table()
    tab.label = "flat-mc-cp-2"
    tab.caption = f"Verfeinerung des Parameters $C_p$ um den zuvor gefundenen Wert $C_p={best_initial_cp:.3f}$. " \
                  f"Test über {NUM_GAMES} Spiele mit {MAX_STEPS} Schritten pro Zug."
    tab.add_row(scores, label="Sieg \\%")
    tab.set_full_col_header(refined_cp_values)
    tab.top_left = "$C_p$"
    tab.write_to_file("v2/2_flat_mc_cp_2")

    if max(scores) > 0.5:
        FLAT_MC_CP = refined_cp_values[argmax(scores)]
        write_log(f"Cp value was improved. Using {FLAT_MC_CP}")
    else:
        FLAT_MC_CP = best_initial_cp
        write_log(f"Cp value did not improve. Using {FLAT_MC_CP}")

    """Get move score"""
    setting = {**base_setting, "exploration_constant": FLAT_MC_CP}
    good, perfect = run_move_eval(FlatMonteCarlo, setting)
    tab = Table()
    tab.label = "flat-mc-move-eval"
    tab.caption = f"Gute und Perfekte Zugauswahl des Flat Monte Carlo Spielers im Datensatz mit 1000 Spielpositionen " \
                  f"mit einem Parameter $C_p={FLAT_MC_CP}$. Es wurden {MOVE_EVAL_REPEATS} " \
                  f"Wiederholungen durchgeführt."

    tab.top_left = "Steps"
    tab.set_full_col_header(MOVE_EVAL_STEPS)
    tab.add_row(good, label="Gut \\%")
    tab.add_row(perfect, label="Perfekt \\%")
    tab.write_to_file("v2/3_flat_mc_move_score")
    write_log(f"Flat Monte Carlo move scores:\n{tab.print()}")

    return {**base_setting, "exploration_constant": FLAT_MC_CP}

def run_mcts_experiments(flat_mc_settings):

    base_setting = {"max_steps": MAX_STEPS, "exploration_constant": 1.0}

    """Compare MCTS Cp against flat MC"""
    scores = explore_parameter_against_fixed_opponent(
        player=MCTSPlayer,
        opponent=FlatMonteCarlo,
        player_conf=base_setting,
        opponent_conf=flat_mc_settings,
        values=INITIAL_CP_VALUES
    )
    tab = Table()
    tab.label = "mcts-against-flat-mc"
    tab.caption = f"Gewinnchance von Monte Carlo Tree Search Flat Monte Carlo Suche mit UCB Kindauswahl" \
                  f"über {NUM_GAMES} Spiele. Test von verschiedenen Parametern $C_p$. Beide Spieler haben {MAX_STEPS} " \
                  f"Schritte Bedenkzeit pro Zug."
    tab.add_row(scores, label="Sieg \\%")
    tab.set_full_col_header(INITIAL_CP_VALUES)
    tab.top_left = "$C_p$"
    tab.write_to_file("v2/4_mcts_vs_flat_mc")

    write_log(f"Best score of MCTS against Flat MC {max(scores)}")

    """Find Cp for MCTS through selfplay"""
    opponent = {**base_setting, "exploration_constant": 1.0}

    scores = explore_parameter_against_fixed_opponent(
        player=MCTSPlayer,
        opponent=MCTSPlayer,
        player_conf=base_setting,
        opponent_conf=opponent,
        values=INITIAL_CP_VALUES
    )
    tab = Table()
    tab.label = "mcts-cp-1"
    tab.caption = f"Gewinnchance der Monte Carlo Tree Search mit verschiedenen Parametern $C_p$ " \
                  f"gegen MCTS mit $C_p=1.0$ über {NUM_GAMES} Spiele."

    tab.top_left = "$C_p$"
    tab.set_full_col_header(INITIAL_CP_VALUES)
    tab.add_row(scores, label="Sieg \\%")
    tab.write_to_file("v2/5_mcts_cp_1")

    best_initial_cp = INITIAL_CP_VALUES[argmax(scores)]
    write_log(f"Best initial Cp for MCTS {best_initial_cp} with score {max(scores)}")

    """Find better Cp"""
    opponent = {**base_setting, "exploration_constant": best_initial_cp}
    refined_cp_values = get_range(best_initial_cp, 5, 0.05)

    scores = explore_parameter_against_fixed_opponent(
        player=MCTSPlayer,
        opponent=MCTSPlayer,
        player_conf=base_setting,
        opponent_conf=opponent,
        values=refined_cp_values
    )
    tab = Table()
    tab.label = "mcts-cp-2"
    tab.caption = f"Verfeinerung des Parameters $C_p$ für MCTS um den zuvor gefundenen Wert $C_p={best_initial_cp:.3f}$. " \
                  f"Test über {NUM_GAMES} Spiele mit {MAX_STEPS} Schritten pro Zug."

    tab.top_left = "$C_p$"
    tab.set_full_col_header(refined_cp_values)
    tab.add_row(scores, label="Sieg \\%")
    tab.write_to_file("v2/5_mcts_cp_2")

    if max(scores) > 0.5:
        MCTS_CP = refined_cp_values[argmax(scores)]
        write_log(f"MCTS Cp value was improved. Using {MCTS_CP}")
    else:
        MCTS_CP = best_initial_cp
        write_log(f"MCTS Cp value did not improve. Using {MCTS_CP}")

    """Compare with tree and without for different amount of steps"""
    opponent = {**base_setting, "exploration_constant": MCTS_CP}
    setting = {**base_setting, "exploration_constant": MCTS_CP, "keep_tree": True}
    step_vals = [100, 200, 400, 800]
    scores = explore_parameter_against_fixed_opponent(
        player=MCTSPlayer,
        opponent=MCTSPlayer,
        player_conf=setting,
        opponent_conf=opponent,
        values=step_vals,
        parameter="max_steps",
        variable_opponent=True
    )
    tab = Table()
    tab.label = "mcts-keep-tree"
    tab.caption = f"Gewinnchance für unterschiedliche Anzahl Schritte eines MCTS Spielers, der den Baum zwischen Zügen " \
                  f"behält gegen einen Spieler der in jedem Zug einen neuen Baum erstellt. {NUM_GAMES} Spiele."
    tab.add_row(scores, label="Sieg \\%")
    tab.top_left = "Steps"
    tab.set_full_col_header(step_vals)
    tab.write_to_file("v2/6_mcts_tree")

    """Run MCTS move evaluation"""
    setting = {**base_setting, "exploration_constant": MCTS_CP}

    good, perfect = run_move_eval(MCTSPlayer, setting)
    tab = Table()
    tab.label = "mcts-move-eval"
    tab.caption = f"Gute und Perfekte Zugauswahl des MCTS Spielers im Datensatz mit 1000 Spielpositionen " \
                  f"mit einem Parameter $C_p={MCTS_CP}$. Es wurden {MOVE_EVAL_REPEATS} Wiederholungen durchgeführt."

    tab.top_left = "Steps"
    tab.set_full_col_header(MOVE_EVAL_STEPS)
    tab.add_row(good, label="Gut \\%")
    tab.add_row(perfect, label="Perfekt \\%")
    tab.write_to_file("v2/7_mcts_move_score")
    write_log(f"MCTS move scores:\n{tab.print()}")

    return setting


def run_transpos_experiments(flat_mc_settings, mcts_config):
    base_setting = {"max_steps": MAX_STEPS, "exploration_constant": 1.0}

    """Compare Transpos Cp against flat MC"""
    scores = explore_parameter_against_fixed_opponent(
        player=TranspositionPlayer,
        opponent=FlatMonteCarlo,
        player_conf=base_setting,
        opponent_conf=flat_mc_settings,
        values=INITIAL_CP_VALUES
    )
    tab = Table()
    tab.label = "transpos-against-flat-mc"
    tab.caption = f"Gewinnchance von MCTS mit Tranpositionen gegen Flat Monte Carlo Suche mit UCB Kindauswahl" \
                  f"über {NUM_GAMES} Spiele. Test von verschiedenen Parametern $C_p$. Beide Spieler haben {MAX_STEPS} " \
                  f"Schritte Bedenkzeit pro Zug."
    tab.add_row(scores, label="Sieg \\%")
    tab.set_full_col_header(INITIAL_CP_VALUES)
    tab.top_left = "$C_p$"
    tab.write_to_file("v2/8_transpos_vs_flat_mc")

    write_log(f"Best score of Transpos against Flat MC {max(scores)}")

    """Find Cp for Transpos through selfplay against MCTS"""
    scores = explore_parameter_against_fixed_opponent(
        player=TranspositionPlayer,
        opponent=MCTSPlayer,
        player_conf=base_setting,
        opponent_conf=mcts_config,
        values=INITIAL_CP_VALUES
    )
    tab = Table()
    tab.label = "transpos-cp-1"
    tab.caption = f"Gewinnchance der MCTS mit Transpositionen mit verschiedenen Parametern $C_p$ " \
                  f"gegen MCTS mit $C_p={mcts_config['exploration_constant']}$ über {NUM_GAMES} Spiele."

    tab.top_left = "$C_p$"
    tab.set_full_col_header(INITIAL_CP_VALUES)
    tab.add_row(scores, label="Sieg \\%")
    tab.write_to_file("v2/9_transpos_cp_1")

    best_initial_cp = INITIAL_CP_VALUES[argmax(scores)]
    write_log(f"Best initial Cp for Transpos {best_initial_cp} with score {max(scores)}")

    """Find better Cp"""
    refined_cp_values = get_range(best_initial_cp, 5, 0.05)

    scores = explore_parameter_against_fixed_opponent(
        player=TranspositionPlayer,
        opponent=MCTSPlayer,
        player_conf=base_setting,
        opponent_conf=mcts_config,
        values=refined_cp_values
    )
    tab = Table()
    tab.label = "transpos-cp-2"
    tab.caption = f"Verfeinerung des Parameters $C_p$ für MCTS mit Transpositionen um den zuvor gefundenen Wert " \
                  f"$C_p={best_initial_cp:.3f}$. " \
                  f"Test über {NUM_GAMES} Spiele mit {MAX_STEPS} Schritten pro Zug."

    tab.top_left = "$C_p$"
    tab.set_full_col_header(refined_cp_values)
    tab.add_row(scores, label="Sieg \\%")
    tab.write_to_file("v2/9_transpos_cp_2")

    if max(scores) > 0.5:
        TRANSPOS_CP = refined_cp_values[argmax(scores)]
        write_log(f"Transpos Cp value was improved. Using {TRANSPOS_CP}")
    else:
        TRANSPOS_CP = best_initial_cp
        write_log(f"Transpos Cp value did not improve. Using {TRANSPOS_CP}")

    """Compare different uct methods against regular MCTS for a few Cp vals"""
    cp_vals = get_range(TRANSPOS_CP, 7, 0.1)
    uct_methods = ["UCT1", "UCT2", "UCT3", "default"]

    tab = Table()
    tab.top_left = "$C_p$"
    tab.set_full_col_header(cp_vals)

    best_score = 0
    best_method = None
    best_cp = 0

    for meth in uct_methods:
        setting = base_setting.copy()
        setting["uct_method"] = meth
        scores = explore_parameter_against_fixed_opponent(
            player=TranspositionPlayer,
            opponent=MCTSPlayer,
            player_conf=setting,
            opponent_conf=mcts_config,
            values=cp_vals
        )
        tab.add_row(scores, label=meth)
        if max(scores) > best_score:
            best_score = max(scores)
            best_method = meth
            best_cp = cp_vals[argmax(scores)]

    tab.label = "transpos-methods"
    tab.caption = f"Veränderung der Spielstärke mit unterschiedlichen Auswahlmethoden der Kindauswahl für verschiedene " \
                  f"Parameter $C_p$ gegen den optimierten MCTS Spieler. {NUM_GAMES} Spiele mit {MAX_STEPS} Schritten " \
                  f"pro Zug."
    tab.write_to_file("v2/10_transpos_methods")

    write_log(f"Transpos scores with different selection methods:\n{tab.print()}")
    write_log(f"Best selection method: {best_method}")

    """Run Transpos move evaluation"""

    tab = Table()
    for meth in uct_methods:
        setting = {**base_setting, "exploration_constant": TRANSPOS_CP, "uct_method": meth}
        good, perfect = run_move_eval(TranspositionPlayer, setting)

        good = ["gut"] + good
        perfect = ["perfekt"] + perfect

        tab.add_row(good, label=meth)
        tab.add_row(perfect, label=meth)

    tab.set_full_col_header(["gut/perfekt"] + MOVE_EVAL_STEPS)
    tab.top_left = "UCT-Methode"

    tab.label = "transpos-move-eval"
    tab.caption = f"Gute und Perfekte Zugauswahl des MCTS Spielers mit Transpositionen im Datensatz mit 1000 " \
                  f"Spielpositionen mit einem Parameter $C_p={TRANSPOS_CP}$. Es wurden {MOVE_EVAL_REPEATS} " \
                  f"Wiederholungen durchgeführt."

    tab.write_to_file("v2/11_transpos_move_score")
    write_log(f"Transpos move scores:\n{tab.print()}")

    transpos_conf = {**base_setting, "uct_method": best_method, "exploration_constant": best_cp}
    return transpos_conf


def run_score_bounded_experiments(flat_mc_settings, mcts_config):
    base_setting = {"max_steps": MAX_STEPS, "exploration_constant": 1.0}

    """Compare Score bounded Cp against flat MC"""
    scores = explore_parameter_against_fixed_opponent(
        player=ScoreboundedPlayer,
        opponent=FlatMonteCarlo,
        player_conf=base_setting,
        opponent_conf=flat_mc_settings,
        values=INITIAL_CP_VALUES
    )
    tab = Table()
    tab.label = "score-bounded-against-flat-mc"
    tab.caption = f"Gewinnchance von Score Bounded MCTS gegen Flat Monte Carlo Suche mit UCB Kindauswahl" \
                  f"über {NUM_GAMES} Spiele. Test von verschiedenen Parametern $C_p$. Beide Spieler haben {MAX_STEPS} " \
                  f"Schritte Bedenkzeit pro Zug."
    tab.add_row(scores, label="Sieg \\%")
    tab.set_full_col_header(INITIAL_CP_VALUES)
    tab.top_left = "$C_p$"
    tab.write_to_file("v2/12_score_bounded_vs_flat_mc")

    write_log(f"Best score of Score Bounded MCTS against Flat MC {max(scores)}")

    """Find Cp for Score bounded through selfplay against MCTS"""
    scores = explore_parameter_against_fixed_opponent(
        player=ScoreboundedPlayer,
        opponent=MCTSPlayer,
        player_conf=base_setting,
        opponent_conf=mcts_config,
        values=INITIAL_CP_VALUES
    )
    tab = Table()
    tab.label = "score-bounded-cp-1"
    tab.caption = f"Gewinnchance des Score Bounded MCTS mit verschiedenen Parametern $C_p$ " \
                  f"gegen MCTS mit $C_p={mcts_config['exploration_constant']}$ über {NUM_GAMES} Spiele."

    tab.top_left = "$C_p$"
    tab.set_full_col_header(INITIAL_CP_VALUES)
    tab.add_row(scores, label="Sieg \\%")
    tab.write_to_file("v2/13_score_bounded_cp_1")

    best_initial_cp = INITIAL_CP_VALUES[argmax(scores)]
    write_log(f"Best initial Cp for Score Bounded {best_initial_cp} with score {max(scores)}")

    """Find better Cp"""
    refined_cp_values = get_range(best_initial_cp, 5, 0.05)

    scores = explore_parameter_against_fixed_opponent(
        player=ScoreboundedPlayer,
        opponent=MCTSPlayer,
        player_conf=base_setting,
        opponent_conf=mcts_config,
        values=refined_cp_values
    )
    tab = Table()
    tab.label = "score-bounded-cp-2"
    tab.caption = f"Verfeinerung des Parameters $C_p$ für Score Bounded MCTS um den zuvor gefundenen Wert " \
                  f"$C_p={best_initial_cp:.3f}$. " \
                  f"Test über {NUM_GAMES} Spiele mit {MAX_STEPS} Schritten pro Zug."

    tab.top_left = "$C_p$"
    tab.set_full_col_header(refined_cp_values)
    tab.add_row(scores, label="Sieg \\%")
    tab.write_to_file("v2/13_score_bounded_cp_2")

    if max(scores) > 0.5:
        SCORE_BOUNDED_CP = refined_cp_values[argmax(scores)]
        write_log(f"Score bounded Cp value was improved. Using {SCORE_BOUNDED_CP}")
    else:
        SCORE_BOUNDED_CP = best_initial_cp
        write_log(f"Score bounded Cp value did not improve. Using {SCORE_BOUNDED_CP}")

    """Find optimal parameters for cut_delta and cut_gamma"""
    val_range = [-0.2, -0.1, 0, 0.1, 0.2]

    tab = Table()
    tab.top_left = "cut\\_gamma/cut\\_delta"
    tab.set_full_col_header(val_range)

    best_score = 0
    best_params = None

    for cut_gamma in val_range:
        setting = {**base_setting, "exploration_constant": SCORE_BOUNDED_CP, "cut_gamma": cut_gamma}
        scores = explore_parameter_against_fixed_opponent(
            player=ScoreboundedPlayer,
            opponent=MCTSPlayer,
            player_conf=setting,
            opponent_conf=mcts_config,
            values=val_range,
            parameter="cut_delta"
        )
        tab.add_row(scores, label=cut_gamma)
        if max(scores) > best_score:
            best_score = max(scores)
            delta = val_range[argmax(scores)]
            best_params = {"cut_gamma": cut_gamma, "cut_delta": delta}

    tab.label = "score-bounded-best-params"
    tab.caption = f"Vergleich der Auswirkung der Parameter \\verb|cut\\_gamma| " \
                  f"und \\verb|cut\\_delta| in Score Bounded MCTS auf die Spielstärke " \
                  f"gegen optimierten MCTS Spieler. {NUM_GAMES} Spiele mit {MAX_STEPS} Schritten pro Zug."

    tab.write_to_file("v2/14_scorebounded_best_params")

    if best_score > 0.5:
        write_log("Optimized Score Bounded MCTS is better than MCTS. Optimal configuration {} with score {}"
                  .format(best_params, best_score))
    else:
        write_log("Score Bounded MCTS is worse than regular MCTS. Best parameters Conf {}".format(best_params))

    """Run Score bounded move score evaluation"""
    setting = {**base_setting, "exploration_constant": SCORE_BOUNDED_CP, **best_params}

    good, perfect = run_move_eval(ScoreboundedPlayer, setting)
    tab = Table()
    tab.label = "score-bounded-move-eval"
    tab.caption = f"Gute und Perfekte Zugauswahl des Score Bounded MCTS Spielers im " \
                  f"Datensatz mit 1000 Spielpositionen mit einem Parameter $C_p={SCORE_BOUNDED_CP}$ " \
                  f"und den Parametern \\verb|cut\\_delta={best_params['cut_delta']}| und " \
                  f"\\verb|cut\\_gamma={best_params['cut_gamma']}." \
                  f" Es wurden {MOVE_EVAL_REPEATS} Wiederholungen durchgeführt."

    tab.top_left = "Steps"
    tab.set_full_col_header(MOVE_EVAL_STEPS)
    tab.add_row(good, label="Gut \\%")
    tab.add_row(perfect, label="Perfekt \\%")
    tab.write_to_file("v2/15_score_bounded_move_score")
    write_log(f"Score bounded move scores:\n{tab.print()}")

    return setting


def run_rave_experiments(flat_mc_settings, mcts_config):
    base_setting = {"max_steps": MAX_STEPS, "exploration_constant": 1.0}

    """Compare Rave Cp against flat MC"""
    scores = explore_parameter_against_fixed_opponent(
        player=RavePlayer,
        opponent=FlatMonteCarlo,
        player_conf=base_setting,
        opponent_conf=flat_mc_settings,
        values=INITIAL_CP_VALUES
    )
    tab = Table()
    tab.label = "rave-against-flat-mc"
    tab.caption = f"Gewinnchance von RAVE MCTS gegen Flat Monte Carlo Suche mit UCB Kindauswahl" \
                  f"über {NUM_GAMES} Spiele. Test von verschiedenen Parametern $C_p$. Beide Spieler haben {MAX_STEPS} " \
                  f"Schritte Bedenkzeit pro Zug."
    tab.add_row(scores, label="Sieg \\%")
    tab.set_full_col_header(INITIAL_CP_VALUES)
    tab.top_left = "$C_p$"
    tab.write_to_file("v2/16_rave_vs_flat_mc")

    write_log(f"Best score of RAVE MCTS against Flat MC {max(scores)}")

    """Find Cp for RAVE through selfplay against MCTS"""
    scores = explore_parameter_against_fixed_opponent(
        player=RavePlayer,
        opponent=MCTSPlayer,
        player_conf=base_setting,
        opponent_conf=mcts_config,
        values=INITIAL_CP_VALUES
    )
    tab = Table()
    tab.label = "rave-cp-1"
    tab.caption = f"Gewinnchance des RAVE MCTS Spielers mit verschiedenen Parametern $C_p$ " \
                  f"gegen MCTS mit $C_p={mcts_config['exploration_constant']}$ über {NUM_GAMES} Spiele."

    tab.top_left = "$C_p$"
    tab.set_full_col_header(INITIAL_CP_VALUES)
    tab.add_row(scores, label="Sieg \\%")
    tab.write_to_file("v2/17_rave_cp_1")

    best_initial_cp = INITIAL_CP_VALUES[argmax(scores)]
    write_log(f"Best initial Cp for RAVE {best_initial_cp} with score {max(scores)}")

    """Find better Cp"""
    refined_cp_values = get_range(best_initial_cp, 5, 0.05)

    scores = explore_parameter_against_fixed_opponent(
        player=RavePlayer,
        opponent=MCTSPlayer,
        player_conf=base_setting,
        opponent_conf=mcts_config,
        values=refined_cp_values
    )
    tab = Table()
    tab.label = "rave-cp-2"
    tab.caption = f"Verfeinerung des Parameters $C_p$ für RAVE MCTS um den zuvor gefundenen Wert " \
                  f"$C_p={best_initial_cp:.3f}$. " \
                  f"Test über {NUM_GAMES} Spiele mit {MAX_STEPS} Schritten pro Zug."

    tab.top_left = "$C_p$"
    tab.set_full_col_header(refined_cp_values)
    tab.add_row(scores, label="Sieg \\%")
    tab.write_to_file("v2/17_rave_cp_2")

    if max(scores) > 0.5:
        RAVE_CP = refined_cp_values[argmax(scores)]
        write_log(f"RAVE Cp value was improved. Using {RAVE_CP}")
    else:
        RAVE_CP = best_initial_cp
        write_log(f"RAVE Cp value did not improve. Using {RAVE_CP}")

    """Find optimal parameter b"""
    val_range = [0, 0.001, 0.01, 0.1]

    tab = Table()
    tab.top_left = "$b$"
    tab.set_full_col_header(val_range)

    best_score = 0
    best_param = None

    setting = {**base_setting, "exploration_constant": RAVE_CP}
    scores = explore_parameter_against_fixed_opponent(
        player=RavePlayer,
        opponent=MCTSPlayer,
        player_conf=setting,
        opponent_conf=mcts_config,
        values=val_range,
        parameter="b"
    )
    tab.add_row(scores, label="Sieg \\%")
    if max(scores) > best_score:
        best_score = max(scores)
        best_param = val_range[argmax(scores)]

    tab.label = "rave-best-param"
    tab.caption = f"Vergleich der Auswirkung des Parameters $b$ in RAVE MCTS auf die Spielstärke " \
                  f"gegen optimierten MCTS Spieler. {NUM_GAMES} Spiele mit {MAX_STEPS} Schritten pro Zug."

    tab.write_to_file("v2/18_rave_best_param")

    if best_score > 0.5:
        write_log("Optimized RAVE MCTS is better than MCTS. Optimal configuration {} with score {}"
                  .format(best_param, best_score))
    else:
        write_log("RAVE MCTS is worse than regular MCTS. Best parameters Conf {}".format(best_param))

    """Run RAVE move score evaluation"""
    setting = {**base_setting, "exploration_constant": RAVE_CP, "b": best_param}

    good, perfect = run_move_eval(RavePlayer, setting)
    tab = Table()
    tab.label = "rave-move-eval"
    tab.caption = f"Gute und Perfekte Zugauswahl des RAVE MCTS Spielers im " \
                  f"Datensatz mit 1000 Spielpositionen mit einem Parameter $C_p={RAVE_CP}$ " \
                  f"und dem Parameter $b={best_param}." \
                  f" Es wurden {MOVE_EVAL_REPEATS} Wiederholungen durchgeführt."

    tab.top_left = "Steps"
    tab.set_full_col_header(MOVE_EVAL_STEPS)
    tab.add_row(good, label="Gut \\%")
    tab.add_row(perfect, label="Perfekt \\%")
    tab.write_to_file("v2/19_rave_move_score")
    write_log(f"RAVE move scores:\n{tab.print()}")

    return setting


if __name__ == "__main__":
    write_log("Starting...")
    # flat_mc_config = run_flat_mc_experiments()
    # mcts_config = run_mcts_experiments(flat_mc_config)
    #
    # transpos_config = run_transpos_experiments(flat_mc_config, mcts_config)
    #
    # score_bounded_config = run_score_bounded_experiments(flat_mc_config, mcts_config)
    flat_mc_config = {"max_steps": 1000, "ucb_selection": True, "exploration_constant": 1.2}
    mcts_config = {"max_steps": 1000, "exploration_constant": 0.85}
    rave_config = run_rave_experiments(flat_mc_config, mcts_config)

    # configurations = {
    #     "flat": flat_mc_config,
    #     "mcts": mcts_config,
    #     "tranpos": transpos_config,
    #     "scorebound": score_bounded_config,
    #     "rave": rave_config
    # }

    # dump_json("eval_results/configurations_{}.json", configurations)