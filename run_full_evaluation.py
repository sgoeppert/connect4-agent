from bachelorarbeit.mcts import MCTSPlayer
from bachelorarbeit.rave import RavePlayer
from bachelorarbeit.scorebounded import ScoreboundedPlayer
from bachelorarbeit.selfplay import Arena
from bachelorarbeit.tools import run_selfplay_experiment, run_move_evaluation_experiment
from bachelorarbeit.tools import evaluate_against_random, evaluate_against_flat_monte_carlo
from bachelorarbeit.tools import Table
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
LOG_FILE = "full_eval_{}.log".format(timestamp)

if DEBUG:
    MAX_STEPS = 20
    NUM_GAMES = 5
    NUM_GAMES_DETAIL = 20
    MOVE_EVAL_REPEATS = 1
    INITIAL_CP_VALUES = [0.1, 1 / math.sqrt(2)]
    LOG_FILE = "debug_" + LOG_FILE
else:
    MAX_STEPS = 750
    NUM_GAMES = 300
    NUM_GAMES_DETAIL = 500
    MOVE_EVAL_REPEATS = 10
    INITIAL_CP_VALUES = [0.25, 0.5, 1 / math.sqrt(2), 1.0, 1.2, math.sqrt(2)]


def get_range(center, num_vals=5, step=0.1):
    vals_left = num_vals // 2
    vals_right = num_vals - vals_left - 1

    values = [center - (i * step) for i in range(vals_left, 0, -1)]
    values += [center + (i * step) for i in range(vals_right + 1)]
    return list(map(lambda v: round(v, 4), values))


def write_log(message):
    fname = Path(config.ROOT_DIR) / LOG_FILE
    with open(fname, "a+") as f:
        f.write(message + "\n")


def test_random_player():
    print("test_random_player")
    results = run_selfplay_experiment("Random vs Random",
                                      players=(RandomPlayer, RandomPlayer),
                                      num_games=NUM_GAMES_DETAIL)
    write_log(f"Random selfplay result {results['mean']}")


def test_flat_mc_vs_random():
    print("test_flat_mc_vs_random")
    results = evaluate_against_random(
        player=FlatMonteCarlo,
        constructor_args={"max_steps": MAX_STEPS, "ucb_selection": True, "exploration_constant": 1.0},
        num_games=20
    )

    return results["mean"]


def test_flat_mc_ucb_parameters_against_flat_mc(cp_values):
    print("test_flat_mc_ucb_parameters_against_flat_mc")
    base_setting = {"max_steps": MAX_STEPS, "ucb_selection": True}
    comparison = {"max_steps": MAX_STEPS, "ucb_selection": False}

    best_score = 0
    best_configuration = None

    tab = Table()
    tab.caption = "Flache Monte-Carlo-Suche mit UCB gegen ohne. Vergleich der Performance in Abhängigkeit von $C_p$."
    tab.label = "flat-mc-1"
    tab.top_left = "$C_p$"
    tab.set_row_label(0, "Sieg \%")

    all_results = []
    for cp in tqdm(cp_values):
        settings = {**base_setting, "exploration_constant": cp}
        _result = run_selfplay_experiment(
            title="UCB MC vs Flat MC",
            players=(FlatMonteCarlo, FlatMonteCarlo),
            constructor_args=(settings, comparison),
            num_games=NUM_GAMES
        )
        all_results.append(_result)
        score = _result["mean"][0]
        tab.add_column([score], head=cp)

        if score > best_score:
            best_score = score
            best_configuration = settings

    return tab, best_configuration


def flat_mc_ucb_exploration(opponent_config: dict, cp_values: list):
    print("flat_mc_ucb_exploration - cp: {}".format(cp_values))
    base_setting = {"max_steps": MAX_STEPS, "ucb_selection": True}
    comparison = opponent_config

    best_score = 0
    best_configuration = None

    tab = Table()
    tab.top_left = "$C_p$"
    tab.set_row_label(0, "Sieg \%")

    all_results = []
    for cp in tqdm(cp_values):
        settings = {**base_setting, "exploration_constant": cp}
        _result = run_selfplay_experiment(
            title="UCB Selfplay",
            players=(FlatMonteCarlo, FlatMonteCarlo),
            constructor_args=(settings, comparison),
            num_games=NUM_GAMES
        )
        all_results.append(_result)
        score = _result["mean"][0]
        tab.add_column([score], head=cp)

        if score > best_score:
            best_score = score
            best_configuration = settings

    return tab, best_configuration


def random_move_evaluation():
    print("random_move_evaluation")
    res = run_move_evaluation_experiment(
        title="Random move eval",
        player=RandomPlayer,
        repeats=MOVE_EVAL_REPEATS
    )

    return [res["good_pct"], res["perfect_pct"]]


def flat_ucb_move_evaluation(cp_values: list):
    print("flat_ucb_move_evaluation - Cp: {}".format(cp_values))
    base_setting = {"max_steps": MAX_STEPS, "ucb_selection": True}
    best_score = 0
    best_configuration = None

    tab = Table()
    tab.top_left = "$C_p$"
    tab.set_row_label(0, "Gut \%")
    tab.set_row_label(1, "Perfekt \%")

    all_results = []
    for cp in tqdm(cp_values):
        settings = {**base_setting, "exploration_constant": cp}
        _result = run_move_evaluation_experiment(
            title="UCB Move Evaluation",
            player=FlatMonteCarlo,
            player_config=settings,
            repeats=MOVE_EVAL_REPEATS
        )
        all_results.append(_result)
        good = _result["good_pct"]
        perfect = _result["perfect_pct"]
        tab.add_column([good, perfect], head=cp)

        if perfect > best_score:
            best_score = perfect
            best_configuration = settings

    return tab, best_configuration


def mcts_against_flat_mc(opponent_config: dict, cp_values: list):
    print("mcts_against_flat_mc")
    base_setting = {"max_steps": MAX_STEPS, "keep_tree": False}
    comparison = opponent_config

    best_score = 0
    best_configuration = None

    tab = Table()
    tab.top_left = "$C_p$"
    tab.set_row_label(0, "\\verb|keep_tree=False|")

    print("keep_tree=false")
    all_results = []
    for cp in tqdm(cp_values):
        settings = {**base_setting, "exploration_constant": cp}
        _result = run_selfplay_experiment(
            title="MCTS vs FlatMC",
            players=(MCTSPlayer, FlatMonteCarlo),
            constructor_args=(settings, comparison),
            num_games=NUM_GAMES
        )
        all_results.append(_result)
        score = _result["mean"][0]
        tab.add_column([score], head=cp)

        if score > best_score:
            best_score = score
            best_configuration = settings

    print("keep_tree=true")
    base_setting["keep_tree"] = True
    keep_tree_scores = []
    for cp in tqdm(cp_values):
        settings = {**base_setting, "exploration_constant": cp}
        _result = run_selfplay_experiment(
            title="MCTS vs FlatMC",
            players=(MCTSPlayer, FlatMonteCarlo),
            constructor_args=(settings, comparison),
            num_games=NUM_GAMES
        )
        all_results.append(_result)
        score = _result["mean"][0]
        keep_tree_scores.append(score)

        if score > best_score:
            best_score = score
            best_configuration = settings

    tab.add_row(keep_tree_scores, label="\\verb|keep_tree=True|")

    return tab, best_configuration


def mcts_move_evaluation(cp_values: list):
    print("mcts_move_evaluation - Cp: {}".format(cp_values))
    base_setting = {"max_steps": MAX_STEPS}

    best_score = 0
    best_configuration = None

    tab = Table()
    tab.top_left = "$C_p$"
    tab.set_row_label(0, "Gut \%")
    tab.set_row_label(1, "Perfekt \%")

    all_results = []
    for cp in tqdm(cp_values):
        settings = {**base_setting, "exploration_constant": cp}
        _result = run_move_evaluation_experiment(
            title="MCTS Move Evaluation",
            player=MCTSPlayer,
            player_config=settings,
            repeats=MOVE_EVAL_REPEATS
        )
        all_results.append(_result)
        good = _result["good_pct"]
        perfect = _result["perfect_pct"]
        tab.add_column([good, perfect], head=cp)

        if perfect > best_score:
            best_score = perfect
            best_configuration = settings

    return tab, best_configuration


def transposition_against_flat_mc(opponent_config: dict, cp_values: list):
    print("transposition_against_flat_mc")
    base_setting = {"max_steps": MAX_STEPS, "keep_tree": False, "uct_method": "UCT1"}
    comparison = opponent_config

    best_score = 0
    best_configuration = None

    tab = Table()
    tab.top_left = "$C_p$"
    tab.set_row_label(0, "\\verb|UCT1|")

    print("UCT1")
    all_results = []
    for cp in tqdm(cp_values):
        settings = {**base_setting, "exploration_constant": cp}
        _result = run_selfplay_experiment(
            title="",
            players=(TranspositionPlayer, FlatMonteCarlo),
            constructor_args=(settings, comparison),
            num_games=NUM_GAMES
        )
        all_results.append(_result)
        score = _result["mean"][0]
        tab.add_column([score], head=cp)

        if score > best_score:
            best_score = score
            best_configuration = settings

    print("UCT2")
    base_setting["uct_method"] = "UCT2"
    uct2_scores = []
    for cp in tqdm(cp_values):
        settings = {**base_setting, "exploration_constant": cp}
        _result = run_selfplay_experiment(
            title="",
            players=(TranspositionPlayer, FlatMonteCarlo),
            constructor_args=(settings, comparison),
            num_games=NUM_GAMES
        )
        all_results.append(_result)
        score = _result["mean"][0]
        uct2_scores.append(score)

        if score > best_score:
            best_score = score
            best_configuration = settings

    tab.add_row(uct2_scores, label="\\verb|UCT2|")

    print("default")
    base_setting["uct_method"] = "default"
    default_scores = []
    for cp in tqdm(cp_values):
        settings = {**base_setting, "exploration_constant": cp}
        _result = run_selfplay_experiment(
            title="",
            players=(TranspositionPlayer, FlatMonteCarlo),
            constructor_args=(settings, comparison),
            num_games=NUM_GAMES
        )
        all_results.append(_result)
        score = _result["mean"][0]
        default_scores.append(score)

        if score > best_score:
            best_score = score
            best_configuration = settings

    tab.add_row(default_scores, label="\\verb|default|")

    return tab, best_configuration


def transposition_move_evaluation(cp_values: list):
    print("transposition_move_evaluation - Cp: {}".format(cp_values))
    base_setting = {"max_steps": MAX_STEPS}

    best_score = 0
    best_configuration = None

    tab = Table()
    tab.top_left = "$C_p$"
    tab.set_full_col_header(["gut/perfekt"] + cp_values)

    uct_methods = ["UCT1", "UCT2", "default"]

    for uct in tqdm(uct_methods):
        print("Eval ", uct)
        goods = ["Gut \%"]
        perfects = ["Perfekt \%"]
        for cp in cp_values:
            settings = {**base_setting, "exploration_constant": cp, "uct_method": uct}
            _result = run_move_evaluation_experiment(
                title="",
                player=TranspositionPlayer,
                player_config=settings,
                repeats=MOVE_EVAL_REPEATS,
                show_progress_bar=False
            )
            perfect = _result["perfect_pct"]
            goods.append(_result["good_pct"])
            perfects.append(perfect)

            if perfect > best_score:
                best_score = perfect
                best_configuration = settings
        tab.add_row(goods, label=uct)
        tab.add_row(perfects, label=uct)

    return tab, best_configuration


def test_best_transpos_against_best_mcts(tp_conf, mcts_conf):
    print("test_best_transpos_against_best_mcts")
    tab = Table()
    tab.set_full_col_header(["Ergebnis"])
    uct_methods = ["UCT1", "UCT2", "default"]

    best_score = 0
    best_configuration = None
    for uct in tqdm(uct_methods):
        settings = tp_conf.copy()
        settings["uct_method"] = uct
        res = run_selfplay_experiment(
            "",
            players=(TranspositionPlayer, MCTSPlayer),
            constructor_args=(settings, mcts_conf),
            num_games=NUM_GAMES
        )
        score = res["mean"][0]
        tab.add_row([score], label=uct)
        if score > best_score:
            best_score = score
            best_configuration = settings

    return tab, best_configuration, best_score


def scorebounded_against_flat_mc(opponent_config: dict, cp_values: list):
    print("scorebounded_against_flat_mc")
    base_setting = {"max_steps": MAX_STEPS, "keep_tree": False}
    comparison = opponent_config

    best_score = 0
    best_configuration = None

    tab = Table()
    tab.top_left = "$C_p$"
    tab.set_row_label(0, "Sieg \%")

    all_results = []
    for cp in tqdm(cp_values):
        settings = {**base_setting, "exploration_constant": cp}
        _result = run_selfplay_experiment(
            title="",
            players=(ScoreboundedPlayer, FlatMonteCarlo),
            constructor_args=(settings, comparison),
            num_games=NUM_GAMES
        )
        all_results.append(_result)
        score = _result["mean"][0]
        tab.add_column([score], head=cp)

        if score > best_score:
            best_score = score
            best_configuration = settings

    return tab, best_configuration


def scorebounded_move_evaluation(cp_values: list):
    print("scorebounded_move_evaluation")
    base_setting = {"max_steps": MAX_STEPS}

    best_score = 0
    best_configuration = None

    tab = Table()
    tab.top_left = "$C_p$"
    tab.set_row_label(0, "Gut \%")
    tab.set_row_label(1, "Perfekt \%")

    all_results = []
    for cp in tqdm(cp_values):
        settings = {**base_setting, "exploration_constant": cp}
        _result = run_move_evaluation_experiment(
            title="",
            player=ScoreboundedPlayer,
            player_config=settings,
            repeats=MOVE_EVAL_REPEATS
        )
        all_results.append(_result)
        good = _result["good_pct"]
        perfect = _result["perfect_pct"]
        tab.add_column([good, perfect], head=cp)

        if perfect > best_score:
            best_score = perfect
            best_configuration = settings

    return tab, best_configuration


def scorebounded_find_best_parameters_against_mcts(cp: float, value_range: list, opponent: dict):
    print("scorebounded_find_best_parameters")
    base_setting = {"max_steps": MAX_STEPS, "exploration_constant": cp}

    tab = Table()
    tab.top_left = "gamma/delta"
    tab.set_full_col_header(value_range)

    best_score = 0
    best_configuration = None

    pbar = tqdm(total=len(value_range) ** 2)

    for gamma in value_range:
        row = []
        for delta in value_range:
            settings = {**base_setting, "cut_delta": delta, "cut_gamma": gamma}
            res = run_selfplay_experiment(
                "",
                players=(ScoreboundedPlayer, MCTSPlayer),
                constructor_args=(settings, opponent),
                num_games=NUM_GAMES
            )
            score = res["mean"][0]
            row.append(score)
            pbar.update()
            if score > best_score:
                best_score = score
                best_configuration = settings
        tab.add_row(row, label=gamma)
    pbar.close()

    return tab, best_configuration, best_score


def rave_against_flat_mc(opponent_config: dict, cp_values: list):
    print("rave_against_flat_mc")
    base_setting = {"max_steps": MAX_STEPS, "keep_tree": False}
    comparison = opponent_config

    best_score = 0
    best_configuration = None

    tab = Table()
    tab.top_left = "$C_p$"
    tab.set_row_label(0, "Sieg \%")

    all_results = []
    for cp in tqdm(cp_values):
        settings = {**base_setting, "exploration_constant": cp}
        _result = run_selfplay_experiment(
            title="",
            players=(RavePlayer, FlatMonteCarlo),
            constructor_args=(settings, comparison),
            num_games=NUM_GAMES
        )
        all_results.append(_result)
        score = _result["mean"][0]
        tab.add_column([score], head=cp)

        if score > best_score:
            best_score = score
            best_configuration = settings

    return tab, best_configuration


def rave_move_evaluation(cp_values: list):
    print("rave_move_evaluation")
    base_setting = {"max_steps": MAX_STEPS}

    best_score = 0
    best_configuration = None

    tab = Table()
    tab.top_left = "$C_p$"
    tab.set_row_label(0, "Gut \%")
    tab.set_row_label(1, "Perfekt \%")

    all_results = []
    for cp in tqdm(cp_values):
        settings = {**base_setting, "exploration_constant": cp}
        _result = run_move_evaluation_experiment(
            title="",
            player=RavePlayer,
            player_config=settings,
            repeats=MOVE_EVAL_REPEATS
        )
        all_results.append(_result)
        good = _result["good_pct"]
        perfect = _result["perfect_pct"]
        tab.add_column([good, perfect], head=cp)

        if perfect > best_score:
            best_score = perfect
            best_configuration = settings

    return tab, best_configuration


def rave_find_best_parameter_against_mcts(cp: float, value_range: list, opponent: dict):
    print("rave_find_best_parameter_against_mcts")
    base_setting = {"max_steps": MAX_STEPS, "exploration_constant": cp}

    tab = Table()
    tab.top_left = "$b$"

    best_score = 0
    best_configuration = None

    for b in tqdm(value_range):
        settings = {**base_setting, "b": b}
        res = run_selfplay_experiment(
            "",
            players=(RavePlayer, MCTSPlayer),
            constructor_args=(settings, opponent),
            num_games=NUM_GAMES
        )
        score = res["mean"][0]
        tab.add_column([score], head=b)
        if score > best_score:
            best_score = score
            best_configuration = settings

    return tab, best_configuration, best_score


if __name__ == "__main__":
    executed_steps = 0
    print("===== BASELINE =====")
    """
    Allgemeiner Test mit random player
    """
    test_random_player()
    executed_steps += 1
    print(executed_steps, "/21")

    """
    Teste of flat monte carlo funktioniert
    """
    result = test_flat_mc_vs_random()
    executed_steps += 1
    print(executed_steps, "/21")
    if result[0] > 0.5:
        write_log("Flat Monte Carlo is better than random")
    else:
        write_log("Flat Monte Carlo is worse than random")

    """
    Bestimme optimalen Parameter C_p für FlatMC mit UCB gegen FlatMC ohne UCB
    """
    t, ucb_conf = test_flat_mc_ucb_parameters_against_flat_mc(INITIAL_CP_VALUES)
    t.write_to_file("1_flat_mc_ucb_vs_flat_mc")
    write_log("Best initial C_p for FlatMC UCB is {}".format(ucb_conf["exploration_constant"]))
    executed_steps += 1
    print(executed_steps, "/21")

    """
    Bestimme optimalen Parameter C_p für FlatMC mit UCB gegen Spieler mit oben bestimmtem Parameter C_p
    """
    t, best_conf = flat_mc_ucb_exploration(ucb_conf, INITIAL_CP_VALUES)
    t.caption = "Optimierung von $C_p$ durch Selfplay gegen den zuvor bestimmten Parameter $C_p={:.3f}$" \
        .format(ucb_conf["exploration_constant"])
    t.label = "flat-mc-2"
    t.write_to_file("2_flat_mc_ucb_selfplay_1")
    write_log("Best C_p for FlatMC UCB is {} after first selfplay".format(best_conf["exploration_constant"]))
    executed_steps += 1
    print(executed_steps, "/21")

    """
    Optimiere den Parameter auf 0.05 genau durch nochmaliges Selfplay
    """
    best_cp = best_conf["exploration_constant"]
    flat_ucb_detail_cp = get_range(best_cp, num_vals=5, step=0.05)
    t, best_conf_ucb = flat_mc_ucb_exploration(ucb_conf, flat_ucb_detail_cp)

    t.caption = "Verfeinerung der vorherigen Ergebnisse gegen $C_p={:.3f}$" \
        .format(ucb_conf["exploration_constant"])
    t.label = "flat-mc-3"
    t.write_to_file("2_flat_mc_ucb_selfplay_2")

    best_flat_ucb_cp = best_conf_ucb["exploration_constant"]
    write_log("Refined C_p for FlatMC UCB is {}".format(best_flat_ucb_cp))
    executed_steps += 1
    print(executed_steps, "/21")

    """
    Prüfe wie gut der zufällige Spieler den korrekten Spielzug findet
    """
    random_scores = random_move_evaluation()
    """
    Prüfe wie gut FlatMC den korrekten Spielzug findet
    """
    tab, best_conf_move = flat_ucb_move_evaluation(flat_ucb_detail_cp)
    tab.insert_column(0, random_scores, head="RandomPlayer")
    tab.label = "move-eval-base"
    tab.caption = "Prozentsatz der guten und perfekten Züge im Datensatz mit 750 Spielpositionen für den zufälligen " \
                  "und den Flat Monte Carlo Spieler. Jede Evaluation wurde 10 mal wiederholt und der Durchschnitt der " \
                  "Ergebnisse gebildet. Der Flat Monte Carlo Spieler benutzt die UCB-Formel zur Kindauswahl mit der " \
                  "Verschiedenen $C_p$ und hat 750 Iterationen Bedenkzeit pro Zug."
    tab.write_to_file("3_move_eval_baseline")

    write_log("First move evaluation done")
    executed_steps += 1
    print(executed_steps, "/21")
    # best_conf_ucb = {"max_steps": 100, "ucb_selection": True, "exploration_constant": 1.0}


    print("===== NORMALE MCTS =====")

    """
    Vergleiche normale MCTS mit FlatMC
    """
    tab, best_conf = mcts_against_flat_mc(best_conf_ucb, INITIAL_CP_VALUES)
    tab.label = "mcts-flat-mc-1"
    tab.caption = "Gewinnchance der MCTS gegen FlatMC über 300 Spiele für verschiedene Werte von $C_p$."
    tab.write_to_file("4_mcts_vs_flat_mc_1")
    executed_steps += 1
    print(executed_steps, "/21")
    """
    Verfeinere C_p gegen FlatMC
    """
    mcts_best_cp = best_conf["exploration_constant"]
    mcts_detail_cp = get_range(mcts_best_cp, num_vals=5, step=0.05)

    tab, best_mcts_conf = mcts_against_flat_mc(best_conf_ucb, mcts_detail_cp)
    tab.label = "mcts-flat-mc-2"
    tab.caption = "Optimierung des Parameters $C_p$ mit einer Auflösung" \
                  " vom 0.05 um den zuvor bestimmten Wert {:.4f} gegen FlatMC. 300 Spiele.".format(mcts_best_cp)
    tab.write_to_file("4_mcts_vs_flat_mc_2")
    mcts_best_cp = best_mcts_conf["exploration_constant"]
    executed_steps += 1
    print(executed_steps, "/21")

    """
    Prüfe wie gut MCTS den korrekten Spielzug findet
    """
    tab, best_mcts_conf_move = mcts_move_evaluation(mcts_detail_cp)
    tab.label = "mcts-move-eval"
    tab.caption = "Prozentsatz der guten und perfekten Züge im Datensatz mit 750 Spielpositionen für die " \
                  "normale Monte-Carlo-Baumsuche. Jede Evaluation wurde 10 mal wiederholt und der " \
                  "Durchschnitt der Ergebnisse gebildet."
    tab.write_to_file("5_mcts_move_eval")
    executed_steps += 1
    print(executed_steps, "/21")

    if mcts_best_cp != best_mcts_conf_move["exploration_constant"]:
        write_log(f"Best Cp for MCTS in selfplay {mcts_best_cp} differs from Cp "
                  f"for Move Eval {best_mcts_conf_move['exploration_constant']}")
    else:
        write_log(f"Best Cp for MCTS in selfplay and Move Eval are the same: {mcts_best_cp}")


    print("===== TRANSPOSITON MCTS =====")
    """
    Vergleiche MCTS mit Transpositionen mit FlatMC
    """
    tab, transpos_conf = transposition_against_flat_mc(best_conf_ucb, INITIAL_CP_VALUES)
    tab.label = "transpos-flat-mc-1"
    tab.caption = "Gewinnchance von MCTS mit Transpositionen gegen FlatMC über " \
                  "300 Spiele für verschiedene Parameter $C_p$."
    tab.write_to_file("6_transpos_flat_mc_1")
    executed_steps += 1
    print(executed_steps, "/21")

    """
    Verfeinere Transpositionen C_p gegen FlatMC
    """
    transpos_best_cp = transpos_conf["exploration_constant"]
    transpos_detail_cp = get_range(transpos_best_cp, num_vals=5, step=0.05)

    tab, best_transpos_conf = transposition_against_flat_mc(best_conf_ucb, transpos_detail_cp)
    tab.label = "transpos-flat-mc-2"
    tab.caption = "Optimierung des Parameters $C_p$ mit einer Auflösung" \
                  " vom 0.05 um den zuvor bestimmten Wert {:.4f} gegen FlatMC. 300 Spiele.".format(transpos_best_cp)
    tab.write_to_file("6_transpos_flat_mc_2")
    transpos_best_cp = best_transpos_conf["exploration_constant"]

    executed_steps += 1
    print(executed_steps, "/21")

    """
    Prüfe wie gut MCTS mit Transpositionen den korrekten Spielzug findet
    """
    tab, best_transpos_move_conf = transposition_move_evaluation(transpos_detail_cp)
    tab.label = "tranpos-move-eval"
    tab.caption = "Anteil guter und perfekter Züge der MCTS mit Transpositionen für verschiedene Parameter $C_p$."

    tab.write_to_file("7_transpos_move_eval")
    executed_steps += 1
    print(executed_steps, "/21")

    transpos_best_move_cp = best_transpos_move_conf["exploration_constant"]
    if transpos_best_move_cp != transpos_best_cp:
        write_log(f"Best Cp for Transpositions vs FlatMC {transpos_best_cp} differs"
                  f"from best Cp in Move Eval {transpos_best_move_cp}")
    else:
        write_log(f"Best Cp for Transpositions is {transpos_best_cp}")

    """
    Vergleiche MCTS mit Transpositionen gegen MCTS ohne
    """
    tab, best_transpos_conf, score = test_best_transpos_against_best_mcts(best_transpos_conf, best_mcts_conf)
    tab.label = "transpos-mcts"
    tab.label = "Vergleich von MCTS mit Transpositionen gegen den optimierten MCTS Spieler. " \
                "Beide Spieler haben {} Schritte pro Spielzug Bedenkzeit, der MCTS Spieler " \
                "spielt mit $C_p={}$ und der Transpositions-Spieler mit $C_p={}$" \
        .format(MAX_STEPS, mcts_best_cp, transpos_best_cp)
    tab.write_to_file("8_transpos_vs_mcts")
    executed_steps += 1
    print(executed_steps, "/21")

    if score > 0.5:
        write_log("MCTS with Transpositions is better than regular MCTS. "
                  "Best player had configuration {} and won with {} score."
                  .format(best_transpos_conf, score))
    else:
        write_log("Transpositions don't seem to improve play against MCTS. Conf {}".format(best_transpos_conf))

    print("===== SCORE BOUNDED MCTS =====")
    """
    Vergleiche Score Bounded mit FlatMC
    """
    tab, scorebound_conf = scorebounded_against_flat_mc(best_conf_ucb, INITIAL_CP_VALUES)
    tab.label = "scorebounded-flat-mc-1"
    tab.caption = "Gewinnchance von Score Bounded MCTS gegen FlatMC über " \
                  "300 Spiele für verschiedene Parameter $C_p$."
    tab.write_to_file("9_scorebound_flat_mc_1")
    executed_steps += 1
    print(executed_steps, "/21")

    """
    Verfeinere Score Bounded C_p gegen FlatMC
    """
    scorebound_best_cp = scorebound_conf["exploration_constant"]
    scorebound_detail_cp = get_range(scorebound_best_cp, num_vals=5, step=0.05)

    tab, best_scorebound_conf = scorebounded_against_flat_mc(best_conf_ucb, scorebound_detail_cp)
    tab.label = "scorebound-flat-mc-2"
    tab.caption = "Optimierung des Parameters $C_p$ mit einer Auflösung" \
                  " vom 0.05 um den zuvor bestimmten Wert {:.4f} gegen FlatMC. 300 Spiele.".format(scorebound_best_cp)
    tab.write_to_file("9_scorebound_flat_mc_2")

    scorebound_best_cp = best_scorebound_conf["exploration_constant"]
    executed_steps += 1
    print(executed_steps, "/21")
    """
    Prüfe wie gut Score Bounded MCTS den korrekten Spielzug findet
    """
    tab, best_scorebound_move_conf = scorebounded_move_evaluation(scorebound_detail_cp)
    tab.label = "scorebound-move-eval"
    tab.caption = "Anteil guter und perfekter Züge der Score Bounded MCTS für verschiedene Parameter $C_p$."

    tab.write_to_file("10_scorebound_move_eval")
    executed_steps += 1
    print(executed_steps, "/21")

    scorebound_best_move_cp = best_scorebound_move_conf["exploration_constant"]
    if scorebound_best_move_cp != scorebound_best_cp:
        write_log(f"Best Cp for Score Bounded vs FlatMC {scorebound_best_cp} differs"
                  f"from best Cp in Move Eval {scorebound_best_move_cp}")
    else:
        write_log(f"Best Cp for Score Bounded is {scorebound_best_cp}")

    """
    Finde optimale Parameter gamma und delta für Score Bounded MCTS
    """
    temp = scorebounded_find_best_parameters_against_mcts(scorebound_best_cp,
                                                       value_range=[-0.2, -0.1, 0, 0.1, 0.2],
                                                       opponent=best_mcts_conf)
    tab, best_scorebound_conf, best_score = temp
    tab.label = "scorebound-best-params"
    tab.caption = f"Vergleich der Auswirkung der Parameter \\verb|cut\_gamma| " \
                  f"und \\verb|cut\_delta| in Score Bounded MCTS auf die Spielstärke gegen optimierten MCTS Spieler. {NUM_GAMES}"
    tab.write_to_file("11_scorebounded_best_params")
    if best_score > 0.5:
        write_log("Optimized Score Bounded MCTS is better than MCTS. Optimal configuration {} with score {}"
                  .format(best_scorebound_conf, best_score))
    else:
        write_log("Score Bounded MCTS is worse than regular MCTS. Conf {}".format(best_scorebound_conf))
    executed_steps += 1
    print(executed_steps, "/21")


    print("===== RAVE MCTS =====")
    """
    Vergleiche Rave MCTS mit FlatMC
    """
    tab, rave_conf = rave_against_flat_mc(best_conf_ucb, INITIAL_CP_VALUES)
    tab.label = "rave-flat-mc-1"
    tab.caption = "Suche nach optimalem Parameter $C_p$ für Rave MCTS gegen Flat Monte Carlo."
    tab.write_to_file("12_rave_vs_flat_mc_1")
    executed_steps += 1
    print(executed_steps, "/21")

    """
    Verfeinere Rave C_p gegen FlatMC
    """
    rave_best_cp = rave_conf["exploration_constant"]
    rave_detail_cp = get_range(rave_best_cp, 5, 0.05)

    tab, rave_best_conf = rave_against_flat_mc(best_conf_ucb, rave_detail_cp)
    tab.label = "rave-flat-mc-2"
    tab.caption = "Suche nach optimalem Parameter $C_p$ für Rave MCTS gegen Flat Monte Carlo."
    tab.write_to_file("12_rave_vs_flat_mc_2")
    executed_steps += 1
    print(executed_steps, "/21")

    """
    Prüfe wie gut Rave MCTS den korrekten Spielzug findet
    """
    tab, best_rave_move_conf = rave_move_evaluation(rave_detail_cp)
    tab.label = "rave-move-eval"
    tab.caption = "Anteil guter und perfekter Züge der Rave MCTS für verschiedene Parameter $C_p$."
    tab.write_to_file("13_rave_move_eval")
    executed_steps += 1
    print(executed_steps, "/21")

    rave_best_move_cp = best_rave_move_conf["exploration_constant"]
    if rave_best_move_cp != rave_best_cp:
        write_log(f"Best Cp for Rave vs FlatMC {rave_best_cp} differs"
                  f"from best Cp in Move Eval {rave_best_move_cp}")
    else:
        write_log(f"Best Cp for Rave is {rave_best_cp}")

    """
    Finde optimale Parameter b für Rave MCTS
    """
    tab, best_rave_conf, best_score = rave_find_best_parameter_against_mcts(
        rave_best_cp,
        [1, 0.1, 0.01, 0.001, 0.0001],
        opponent=best_mcts_conf
    )

    tab.label = "rave-best-param"
    tab.caption = "Vergleich der Spielstärke gegen normale MCTS in Abhängigkeit vom Parameter $b$."
    tab.write_to_file("14_rave_best_param")

    if best_score > 0.5:
        write_log("Rave MCTS is better than MCTS. Optimal configuration {} with score {}."
                  .format(best_rave_conf, best_score))
    else:
        write_log("Rave MCTS is worse than regular MCTS. Conf {}".format(best_rave_conf))
    executed_steps += 1
    print(executed_steps, "/21")

    print("DONE")
    