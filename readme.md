# Entwicklung eines Agenten für VierGewinnt mit Monte-Carlo-Baumsuche und neuronalen Netzen

Dieses Repository enthält den Programmcode, der im Laufe meiner Bachelorarbeit entwickelt wurde, sowie die dabei gesammelten Daten. 

## Installation
Die Agenten wurden mit Python 3.8 entwickelt und getestet. Für die Implementierung der neuronalen Netze wurde Tensorflow 2 mit Keras verwendet. 

Es wird empfohlen ein Python `venv` zu verwenden um die Abhängigkeiten zu installieren.

Erstellung der virtuellen Umgebung (erzeugt einen Ordner `venv` im aktuellen Verzeichnis):
```
python3 -m venv venv
``` 

und danach Aktivierung der Umgebung mit (Linux)
```
source ./venv/bin/activate
```
oder für Windows
```
.\venv\bin\activate.bat
```

Nachdem das venv aktiviert ist, könnend die Abhängigkeiten mit pip installiert werden:
```
python3 -m pip install -r requirements.txt
```

Dies lädt alle benötigten Module herunter und installiert sie. Für die Verwendung der neuronalen Netze wird außerdem empfohlen, [Nvidia CUDA zu installieren](https://www.tensorflow.org/install/gpu).

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

Die Vergleichsprogramme werden über Skripte im Hauptverzeichnis gestartet.  
`run_tuner.py` startet den `MCTSTuner`, welcher automatisch vordefinierte Parametrisierungen untersucht und nach den optimalen Parametern sucht. Die Ergebnisse des Tuners befinden sich im Verzeichnis `/MCTSTuner`.
`test_scaling.py` vergleicht eine Vielzahl von Agenten bei gleicher Ausführungszeit miteinander.