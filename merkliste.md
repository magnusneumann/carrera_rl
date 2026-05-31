
Ein Notebook dokumentiert den Arbeits/Denk/Forschungsprozess und dient als Interaktive Dokumentation mit welcher der Entscheidungsprozess und die Simulation zum mit machen vorgeführt wird.
Das Auto brauch eine Trägheit und auch eine Art Grip/Höchstgeschwindigkeit, ggf hardgecoded einen Schleudervorgang, bei dem das Fahrzeug ausbricht in die aktuelle Lenkrichtung und die letzte Trajektorie beibehällt.
Das Auto sollte etwas Trägheit haben und weiter rollen auch wenn der Gasbefehl grade nicht aktiv ist.

Notebook ablauf:
Ziel erklären
Umgebung mit Auto
Physik Grip, Trägheit, Lenken mit Geschwindigkeit andere Wirkung?
Steuerungbauen, WASD
Pi, die Policy definieren nicht auf Rand/ weiß fahren, nicht rückwärts, vorwärts gut, bei besserer(höherer) oder gleicher rundenzeit reward. je höher v desto höher reward, wenige wechsel zwischen gas bremse und lenkung reward (smooth fahren)
Virtuelle Kamera/CV damit man checken kann wo und wohin das auto ist/will?
Modell konfigurieren
Training configurieren
training starten
weitere trainings/arichtekturen/modelle konfigurieren,
funktion/schliefe die alle trainings/architekturen lernen lässt und testet/vergleicht


Ich will Modell-frei PPO
Ich will Modell-based probieren
Ich will Imitation/Demonstration Learning probieren

Virtuelle Kamera Bild - Reward funktion wie?
Was für ein Netz ist aktuell hinten dran. Muss das größer werden bei mehr Input?
Agenten weiter trainieren

Mit Bilddaten lernen
TrainingsFactory
