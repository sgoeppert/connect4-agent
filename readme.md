# Entwicklung eines Agenten für VierGewinnt mit Monte-Carlo-Baumsuche und neuronalen Netzen

Dieses Repository enthält den Programmcode, der im Laufe meiner Bachelorarbeit entwickelt wurde, sowie die dabei gesammelten Daten. 

## Installation
Die Agenten wurden mit Python 3.8 entwickelt und getestet. Für die Implementierung der neuronalen Netze wurde Tensorflow 2 mit Keras verwendet. 

Es wird empfohlen ein Python `venv` zu verwenden um die Abhängigkeiten zu installieren.

Erstellung der virtuellen Umgebung (erzeugt einen Ordner `venv` im aktuellen Verzeichnis):
```
python3 -m venv venv
``` 

und danach Aktivierung der Umgebung auf Linux mit
```
source ./venv/bin/activate
```
oder auf Windows
```
.\venv\bin\activate.bat
```

Nachdem das venv aktiviert ist, könnend die Abhängigkeiten mit pip installiert werden:
```
python3 -m pip install -r requirements.txt
```

Dies lädt alle benötigten Module herunter und installiert sie. Für die Verwendung der neuronalen Netze wird außerdem empfohlen [Nvidia CUDA zu installieren](https://www.tensorflow.org/install/gpu).

## Inhalt
Für diese Arbeit wurden mehrere Agenten auf Basis der Monte-Carlo-Baumsuche entwickelt. Folgende Agenten wurden entwickelt:

 * Monte-Carlo-Baumsuche `bachelorarbeit.players.mcts.MCTSPlayer`
 * Verbesserung durch Transpositionen `bachelorarbeit.players.transposition.TranspositionPlayer`
 * All Moves as First Heuristik und RAVE `bachelorarbeit.players.rave.RavePlayer`
 * Score Bounded MCTS `bachelorarbeit.players.scorebounded.ScoreboundedPlayer`
 * Last-Good-Reply-Policy `bachelorarbeit.players.adaptive_player.AdaptivePlayoutPlayer`
 * Kombination mit neuronalen Netzen `bachelorarbeit.players.network_player.NetworkPlayer`

Sowie verschieden Kombinationen der oben beschriebenen Spieler.

Für die Auswertung der Spieler wurden verschiedene Hilfsprogramme in `bachelorarbeit.selfplay` und `bachelorarbeit.tools` entwickelt, die unter anderem ein paralleles Ausführen einer großen Anzahl von Spielen erlauben.

Die folgenden Skripte können im Hauptverzeichnis gestartet werden.  
 * `python3 -m run_tuner` startet den `MCTSTuner`, welcher automatisch vordefinierte Parametrisierungen untersucht und nach den optimalen Parametern sucht. Die Ergebnisse des Tuners befinden sich im Verzeichnis `/MCTSTuner`.
   * Jeder Durchlauf legt ein neues `run_` Verzeichnis an und schreibt mehrere Checkpoints in diesem Verzeichnis.
   * Jeder Checkpoint enthält eine Datei `node_stats.json` mit der Auswertung der Parametrisierungen
 * `python3 -m test_scaling` vergleicht eine Vielzahl von Agenten bei gleicher Ausführungszeit miteinander. Die Ergebnisse werden nach `/scaling_test` geschrieben.
 * `python3 -m test_model` vergleicht die trainierten neuronalen Netze in Kombination mit der MCTS und schreibt Ergebnisse nach `/network_test`
 * `python3 -m generate_selfplay_training_data` lässt den MCTSPlayer gegen sich selbst spielen und schreibt die dabei erzeugten Spieldaten nach `/memory`
 * `python3 -m train_all_models_autokeras` verarbeitet die Trainingsdaten und benutzt AutoKeras um neuronale Netze zu trainieren. Die finalen Modelle werden in `/best_models` gespeichert. Während des Trainings werden Log-Daten für TensorBoard im Verzeichnis `/tensorboard_log_autokeras`  erstellt und die Modell-Checkpoints werden in `/auto_models` gespeichert. 
   * Der Trainingsfortschritt kann mit Tensorboard `tensorboard --logdir tensorboard_log_autokeras` überwacht werden
   