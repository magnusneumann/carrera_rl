

## Doku Arbeit:

Kurzfassung könnte weiter oben angeordnet sein, dann würde der Abschnitt nicht auf die nächste  Seite umbrechen

Nomenklatur könnte auch weniger platz über der überschrift vertragen, dann würden die letzten beiden einträge der nomenklatur nicht auf eine neue seite umbrechen

Seite 8 formel 2.3 hier wiederspricht sich etwas. laut dem satz vor der formel muss st alles enthalten was für die zukunft wichtig ist - unter der formel wird aber gesagt die aktuelle informations darf nichts über die folgeposition verraten.
das ist ein widerspruch.
Die Formel zum replay buffer erhält in der pdf version das wort fertig - das wundert mich etwas.

auf manchen seiten fehlen die Seitenzahlen - zb Seite 17 ist aber glaube ich bei latex so gewollt - kann dann auch so bleiben

Das habe ich anders erinnerung: Der genauere Blick relativiert jedoch,wasdabei übertragen wird. Vom vortrainiertenCNN
werden nur die drei Faltungsschichten kopiert; die beiden Linear-Schichten – also genau der
Teil, der aus Bildmerkmalen Gas, Bremse und Lenkung macht – werden verworfen. Übertragen
wird damit Sehfähigkeit, nicht Fahrwissen. Von den elf Runden des Lidar-Experten kommt beim
Vision-Agenten nichts an.
Ich meine ich kenne Tensorboard grafiken die zeigen dass der reward zu beginn des trainings sehr hoch ist - also konnte das vortrainierte netz kurz fahren - war aber overfittet - dewegen hat es durch das lenen aus unbekannten aktionen alles verlernt.





















# Was soll die Doku vermitteln?

Die Dokumentation soll erklären, woran gearbeitet wurde, wie recherchiert wurde und welche Ergebnisse die Recherche lieferte. Sie soll zeigen, wie das Repository zu verstehen ist, warum die Projektstruktur und der 2-Stufen-Ansatz gewählt wurden, was erprobt und bewiesen wurde, welche Erkenntnisse gewonnen wurden und wie gut das Gesamtziel damit erreicht werden konnte.

# Die Dokumentation soll umfassen (LaTeX-Vorlage)

- Deckblatt
- Selbstständigkeitserklärung
- Abstract und Zusammenfassung
- Nach dem Inhaltsverzeichnis: Tabellen-/Abbildungsverzeichnis, Quellenverzeichnis, Anhang

# Inhaltsverzeichnis

## Kapitel 1: Einleitung

#### Motivation

Physical AI mit Carrera Hybrid (visuelle Fahrzeugsteuerung durch Deckenkamera anstelle von Robotern), Interesse an Reinforcement Learning, begrenzte Rechenressourcen (Notwendigkeit von 2D-Prototyping) und der Trackmania KI Rekord von https://www.youtube.com/@yoshtm.

#### Projektstruktur

- Phase 1: 2D-Python-Prototyping & Methodenevaluation:
  Aufbau einer leichtgewichtigen, hochperformanten 2D-Simulationsumgebung (`Carrera2DEnv`), um fundamentale Fragen zu Beobachtungsräumen, Netzwerkarchitekturen, Lernalgorithmen (PPO vs. SAC), Belohnungsfunktionen und Datenvorverarbeitungsmethoden isoliert und zeiteffizient zu untersuchen.
- Phase 2: Transfer in die fotorealistische 3D-Simulation:
  Übertragung der im 2D-Prototyping validierten Best Practices in die fotorealistische 3D-Simulation (NVIDIA Isaac Sim).
- Ausblick auf Phase 3: Sim-to-Real-Transfer:
  Erforschung des Sim-to-Real-Gaps auf der realen Carrera-Hybrid-Hardware (Agenten auf die echte Bahn bringen).

#### Ziele dieser Arbeit

Schnelles 2D-Prototyping, um Konzepte und Probleme mit geringer Rechenlast frühzeitig abzuwägen; Reward Engineering für stabiles Fahren; physikalisch präzises 3D-Simulationstraining mit der entwickelten Pipeline als Basis für einen Sim-to-Real-Transfer in einem Folgeprojekt.
Einblick in die Projektstruktur: Warum Gymnasium? Wo liegen Modelle und Agenten? Wie wird trainiert? Vorstellung des interaktiven Notebooks, in dem die Arbeit nachvollzogen werden kann (Definition des Environments, physikalische Modellierung, Anpassung des Rewards) – all das soll kompakt aufgezeigt werden (ggf. mit Verweis auf eine begleitende `README.md`).

## Kapitel 2: Grundwissen / Begriffserklärungen

#### Theoretische Grundlagen

Verweis auf das empfohlene Einführungsvideo (https://www.youtube.com/watch?v=VnpRp7ZglfA) als anschaulichen Einstieg in das Thema, da ein einzelnes Grundlagenkapitel kaum das gesamte Feld des RL lückenlos abbilden kann.
Die in der Dokumentation verwendeten Konzepte und Begriffe werden hier präzise definiert (Erklärungen aus dem Standardwerk *Reinforcement Learning: An Introduction* von Sutton & Barto zitieren):

- MDP und POMDP ausführlich erläutern
- PPO und SAC: Funktionsweise, Definition einer Policy, On-Policy vs. Off-Policy, modellbasierte vs. modellfreie Verfahren
- Imitation Learning, Behavioral Cloning, Learning from Demonstration, Covariate Shift
- Privilegiertes Lernen (Privileged Learning / Learning by Cheating)
- Frame-Stacking, hybrider Input (Fusion verschiedener Datenformate, z. B. Bilddaten und Skalarparameter)
- Replay-Buffer, Pretraining, Online- vs. Offline-Training

#### State of the Art (SOTA)

Aktueller Stand der Forschung im Bereich RL für RC-Cars, Slotcars und Racing Agents:

- **DIRL (Cai et al. \cite{cai2021vision}):** Beweist die Überlegenheit von hybridem Imitation- und Reinforcement-Learning für reale 1:20-RC-Fahrzeuge.
- **Gran Turismo Sophy (Wurman et al., Nature 2022 \cite{wurman2022outracing}):** Erreicht übermenschliche Rundenzeiten mittels SAC, asymmetrischen Critic-Architekturen und optimierten Streckenfortschritts-Belohnungen.
- **VISTA (Amini et al. \cite{amini2020learning}):** Zeigt, dass das gezielte Erleben und Trainieren von OOD-Rettungsmanövern (Out-of-Distribution) für einen robusten Sim-to-Real-Transfer essenziell ist.
- Verweis auf den Trackmania KI Rekord https://www.youtube.com/@yoshtm

## Kapitel 3: Hauptteil

### Phase 1: 2D-Prototyping

#### Idee

Pragmatischer industrieller Entwicklungsansatz: Was ist bereits vorhanden und was lässt sich sofort beschaffen oder erstellen?
Das Auto (als Grafik) soll auf der Strecke (als Grafik) per WASD-Tastatur steuerbar sein – später soll der Agent dieselbe Steuerschnittstelle nutzen.

- **Strecke:** Screenshot des Streckenlayouts aus der offiziellen Carrera-Hybrid-App; Maßstabsübertragung anhand der realen Abmessungen der physischen Bahn.
- **Fahrzeug & Maße:** Freigestellte Vogelperspektive des Fahrzeugs aus Produktbildern; geometrische Maße und Gewicht direkt am echten Fahrzeug gemessen.
- **Physikalische Randbedingungen:** Höchstgeschwindigkeit aus Messungen in YouTube-Videos entnommen und über den Maßstab skaliert.
- **Beschleunigungsvermögen:** Erste Schätzung nach Augenmaß in einem interaktiven WASD-Demo-Modell im Notebook.
- **Fahrphysik:** Einspurmodell (Bicycle Model), das Befehle für Gas, Bremse und Lenkung verarbeitet und daraus Beschleunigung, Geschwindigkeit und Gierrate berechnet.
  So kann das Prototyping ohne lange Vorlaufzeit starten. Die Spezifikationen werden abschließend in einer Übersichtstabelle zusammengefasst.

#### Umsetzung

Arbeiten im Jupyter Notebook mit modular ausgelagerten Skripten für Environment, Reward, Training und Utilities.
Da die Streckenabmessungen in Pixeln und Metern bekannt sind, lässt sich das Fahrzeugbild über den Maßstab exakt skalieren. Mit Unterstützung generativer Coding-Werkzeuge entsteht eine Notebook-Zelle mit interaktiver WASD-Steuerung auf dem Streckenhintergrund.

Vorbereitung für autonome Agenten:

- Der Agent benötigt Informationen über Fahrtrichtung und Streckenbegrenzungen.
- Der Agent benötigt eine virtuelle Kamera (globale Deckenansicht und fahrzeugfester Ego-Crop).
- Der Agent benötigt Informationen über Streckenfortschritt und Ziellinie.
  Mittels OpenCV-Farberkennung werden die Streckenränder und die Mittellinie extrahiert und als `.npy`-Arrays gespeichert (`tracklimits_cv.py` für die Ränder, Generierung der Mittellinie im Notebook). Die Start-/Ziellinie wird manuell per Cursor über die Start-/Endpixel abgelesen und fest hinterlegt.

Entwurf der Belohnungsfunktion:

- Erste Version bestraft Wandkollisionen und Rückwärtsfahren hart.
- Schnelleres Fahren steigert den Reward quadratisch.
- Korrektes Überfahren der Ziellinie bringt hohe Bonuspunkte; Stillstand führt zu Minuspunkten pro Zeitschritt.
- Überprüfung der Belohnungsfunktion im manuellen WASD-Fenster.
- Aktuelle Reward Funktion erklären

Erstes einfaches Training (Proof of Concept):

- Schnellste Methode: Raycasts. Virtuelle Laserstrahlen schneiden die Streckenbegrenzungen und liefern Distanzwerte. Zu Beginn werden 2 Strahlen im Winkel von $\pm 30^\circ$ eingesetzt.
- Ein erster Agent lernt so bereits nach nur 50.000 Frames einfaches Fahren auf der Strecke.
- **Erkenntnis:** Das Setup, der Reward, die Streckenbegrenzungen, die Basis-Physik, die Steuerung und die Trainingspipeline funktionieren grundlegend. Von dieser soliden Basis aus können Hypothesen verlässlich getestet werden.

Nächster Schritt: Systematische Erprobung verschiedener Eingabekonzepte:

- **Vision:** Unterteilt in Single-Frame (statisches Einzelbild) und Stacked-Frames (aktuelles Bild zusammen mit zwei Bildern aus der jüngsten Vergangenheit).
- Beide Varianten werden sowohl als **globale Draufsicht** auf die gesamte Strecke als auch als **fahrzeugfester Ego-Crop** (Fahrzeug bleibt im Bild zentriert und nach oben ausgerichtet; Strecke bewegt sich relativ zum Auto) getestet.
- **Hybrider Input:** Kombination der visuellen Ansätze mit Skalarparametern (aktuelle Geschwindigkeit und aktueller Lenkwinkel), da diese bei Single-Frame-Eingaben nicht implizit geschätzt werden können.

Reihenfolge der Tests:

1. Single-Frame Ego-Crop (`02_Vision_SingleFrame`, Zellen 29–30)
2. Single-Frame Global (`02_Vision_SingleFrameGlobal`, Zellen 31–32)
3. Hybrid / Multi-Input (`03_Hybrid_MultiInput`, Zellen 33–36)
4. Frame-Stacking (`04_Vision_FrameStack_3` & `07_Vision_Global_3Stack`, Zellen 37–40)

Physikalische Überarbeitung auf gemessene Realdynamik (`measured_v2` vs. `force_drag_v1`):
  Nachträgliche Präzisierung des Fahrzeugmodells anhand von Videoanalysen realer Carrera-Hybrid-Fahrzeuge. Ziel war eine realitätsnahe Annäherung an das spätere 3D-Verhalten, um sicherzustellen, dass der Proof of Concept nicht durch eine unphysikalisch übermotorisierte Modellierung entwertet wird.

Das Subpixel-Wahrnehmungsproblem:
  Unter der realistischen Physik konvergierten die visuellen Agenten zunächst überhaupt nicht mehr. Ursache ist eine optische Wahrnehmungsschwelle: Bei der reduzierten Bildauflösung ($166 \times 100\text{ px}$) bewegt sich das Fahrzeug aus dem Stand heraus über viele Zeitschritte hinweg um weniger als einen Pixel pro Frame (erst ab ca. $0.64\text{ m/s}$ ist eine Positionsveränderung sichtbar). Da der Agent bei realistischer Trägheit über zahlreiche Frames hinweg kontinuierlich Vollgas geben müsste, bevor im 3-Frame-Stack überhaupt ein optischer Unterschied erkennbar wird, ist er beim Anfahren blind für die eigene Beschleunigung – während die Belohnungsfunktion ihn zeitgleich für zu geringe Geschwindigkeit abstraft.

Technische Lösungsansätze gegen die Wahrnehmungsschwelle:

- *Temporales Spreizen des Framestapels:* Abtastung von $t, t-4, t-8$ (anstelle von $t, t-1, t-2$). Dies senkt die Wahrnehmungsschwelle rechnerisch ohne zusätzlichen Rechenaufwand von $0.64\text{ m/s}$ auf $0.16\text{ m/s}$.
- *Auflösungserhöhung:* Anhebung der Bildauflösung auf $250 \times 150\text{ Pixel}$.

Reward Engineering – Evolution zu `v3_rundenzeit`:

- *Beseitigung des Ratchet-Bugs (Sperrklinke):* Eine relative Bestrafung gegenüber früheren Zufalls-Bestzeiten entwertete Folgerunden mit netto $-250$ Punkten und wurde entfernt.
- *Korrektur des Oszillations-Bugs (`w_integral`):* Eine Strafe gegen hochfrequentes Lenkzappeln bestrafte fälschlicherweise auch das kontinuierliche, saubere Halten des Lenkwinkels in Kurven mit bis zu $-625$ Punkten.

Systematischer Algorithmenvergleich (PPO vs. SAC):

- *PPO (On-Policy):* Zeigte sich auf Bilddaten als extrem sample-ineffizient (benötigte rund 90 Millionen Schritte), da gesammelte Rollout-Daten nach jedem Gradientenupdate verworfen werden.
- *SAC (Off-Policy):* Lernt dank des Replay-Buffers drastisch schneller, da gerenderte Frames mehrfach wiederverwendet werden; neigt bei Bilddaten jedoch zu Instabilitäten durch Critic-Divergenz (beobachtet im Einbruch von Lauf `1432`).

Software- und Buffer-Optimierungen:

- *Speicheroptimierter Replay-Buffer (`framestapel_buffer.py`):* Speichert lediglich Einzelbilder und setzt die 3er-Stacks erst on-the-fly beim Mini-Batch-Sampling zusammen; senkt den RAM-Bedarf von 23 GB auf 15 GB.
- *Korrektur der Zeitlimit-Behandlung (`handle_timeout_termination=True`):* Verhindert, dass das Erreichen des 2000-Step-Limits fälschlicherweise als katastrophaler Terminalzustand gelernt wird.
- *Headless-Rendering via OpenCV:* Umstellung von Pygame auf reines OpenCV/NumPy-Rendering, um das Windows-Fenster-Handle-Limit bei parallelen Trainingsumgebungen zu umgehen. Zugleich Aufdeckung des Antialiasing-Artefakts: Eine Kantenabweichung von lediglich $0.05\,\%$ im Auto-Sprite reichte aus, um die Policy im Pygame-Fenster lahmzulegen.

#### Erkenntnisse (Gut / Schlecht)

- Erste Agenten zeigten teils starkes Oszillieren: Lenkung und Gas sprangen binär zwischen $-1$ und $+1$. Für den Realtransfer unbrauchbar, da reale Servos träger sind und vorausschauender gelenkt werden muss.
- Reifenhaftung (Grip) wurde anfangs vernachlässigt und musste nachgebessert werden (untersteuerndes Verhalten).
- Einige Input-Varianten konvergieren nur sehr schwer (Singleframe).
- Bei vielen Parametern und zufälliger Gewichtsinitialisierung scheitert das Lernen oft am "Henne-Ei-Problem": Da der Agent zu Beginn nicht fahren kann, sieht er fast nur Crashs und kann nicht lernen, welche visuellen Merkmale für das Fahren entscheidend sind.
- Ein vortrainiertes Netz hilft: Trotz des Risikos von Catastrophic Forgetting konvergiert der Agent mit Pretraining (Learning from Demonstration anhand des stochastischen Lidar-Experten) deutlich zuverlässiger als ohne.
- Der Agent lernt Geschwindigkeit und Beschleunigung aus Bildern nicht so schnell implizit, wie anfängliche Recherchen vermuten ließen.
- Ziel ist ein Agent, der menschliche Rundenzeiten schlägt und eine vergleichbare Übersicht wie ein menschlicher Fahrer hat (Blick auf die gesamte Bahn). Deshalb wird der global funktionierende Ansatz weiter verfeinert.

#### Fallen und Fehler

- Die Belohnungsfunktion ist hochgradig sensibel: Änderungen führen oft zu Rückschritten, da neue Strafen oder Boni unvorhergesehene, gegenläufige Effekte auslösen können.
- "Angst vor der Ziellinie": Visuelle Agenten mieden die Ziellinie, weil sie zuvor gelernt hatten, dass dunkler Asphalt sicher ist und farbliche Abweichungen Begrenzungen darstellen – die karierte Markierung wurde als Hindernis missverstanden.
- Eine feste Startposition führte zum Einprägen lokaler Minima. Ein Variieren der Startpositionen und das Starten mit Anfangsgeschwindigkeit hätten die Konvergenz beschleunigen können.

### Phase 2: 3D-Deployment (von Claude auszufüllen)

#### Ideen und Konzepte

#### Umsetzung

#### Erkenntnis Gut Schlecht

#### Fallen und Fehler

### Erkenntnisse – Lessons und Best Practices

## Kapitel 4: Ergebnisse

2D Untersuchungen haben eine Pipeline augedeckt um Komplexere Agenten zu trainieren. Aktuellste Pipeline erklären.
Die Pipeline ist in der 3D Simulation anwendbar ODER die Pipeline versagt bei der 3D Simulation.

## Kapitel 5: Fazit (Phase 1 und 2) & Ausblick (Phase 3)

Sim to Real Gap - Agenten der schon fahren kann mit Augmentation bombadieren - damit dieser auch bei harten Störungen fahren kann.

---

## 2. Fehlende Belege & Nachträgliche Experimente (Re-Trainings / Re-Evals)

Folgende empirische Belege und Tests müssen noch durchgeführt bzw. aus vorhandenen Logs visualisiert werden:

- [ ] **Task 2.1: Pretraining-Loss-Kurve des CNNs***Ziel:* Grafischer Nachweis, wie der MSE-Loss über 20 Epochen auf dem Demonstrationsdatensatz fällt und ab wann Überanpassung (Overfitting) einsetzt.
- [ ] **Task 2.2: Vergleichsplots der TensorBoard-Kurven (Critic-Loss & Entropy)***Ziel:* Gegenüberstellung von Lauf `1432` (Critic-Divergenz) vs. Lauf `1748` (stabiler Lidar-SAC) vs. Lauf `2210` (Vision-SAC).
- [ ] **Task 2.3: Auswertung des gespreizten Framestapels ($t, t-4, t-8$)***Ziel:* Experimenteller Beweis, dass die Absenkung der Wahrnehmungsschwelle von $0.64\text{ m/s}$ auf $0.16\text{ m/s}$ das Anfahren und Kurvenfahren des Vision-Agenten unter `measured_v2` ermöglicht.
- [ ] **Task 2.4: PPO-Abschlusslauf mit `lr = 5e-5` + Backbone***Ziel:* Schließen der letzten offenen Vergleichszelle im PPO-Versuchsplan (Vergleich zu SAC bei gleichem Zeitbudget).
- [ ] **Task 2.5: Erzeugung hochauflösender Abbildungen für den Bericht**
  - [ ] Abbildung Streckenmasken (OpenCV Konturen, Splines, Start/Ziel).
  - [ ] Abbildung Fahrzeugperspektiven: Globale Ansicht ($166\times 100$, 4 Graustufen) vs. Ego-Crop vs. Telemetrieleiste.
  - [ ] Abbildung Trajektorienvergleich: Ideallinie des Lidar-Experten vs. verrauschte Rollouts vs. Schüler-Trajektorie.

---

## 3. Strukturierte Fragen an Claude / Kollegen mit Zugang zu weiteren Modellen & Logs

> **Kontext:** Auf dem Zweitrechner befinden sich historische Checkpoints, ungefilterte TensorBoard-Logs und ggf. 3D-Isaac-Sim-Szenen, die wegen Dateigrößenbeschränkungen nicht vollständig im aktuellen Git-Stand liegen.

### Block A: Datensatz & Behavioral Cloning Pretraining

1. **Genaue Datenpunkte:** Wie viele Frames/Übergänge umfasst der finale Demonstrationsdatensatz für das Pretraining (z.B. Datensatz `20260814_193640`) genau?
2. **Noise Injection Parameter:** Welcher Rauschwert $\sigma$ (Standardabweichung auf Lenkung und Gas) wurde bei den Aufzeichnungsfahrten des Lidar-Experten verwendet?
3. **Pretraining Performance:** Welcher finale Validierungs-MSE wurde auf den 3 Aktionen (Lenkung, Gas, Bremse) erreicht, und wurden die Convolutional-Layer während des anschließenden SAC-Trainings eingefroren (frozen backbone) oder end-to-end mittrainiert?

### Block B: TensorBoard-Verläufe & Detailwerte historischer Läufe

4. **PPO-Läufe (`1601`, `2233`, `2328`):** Liegen die genauen Kurven für `rollout/ep_rew_mean` und `train/explained_variance` für den 90M-Schritte-Lauf `2328` vor?
5. **Kamerarezepte (`2147`, `2210`, `2331`):** Gibt es Aufzeichnungen darüber, ob bei `2210` (5M Steps) zwischenzeitlich Teiltrajektorien von mehr als einer halben Runde gefahren wurden, oder ob das Fahrzeug an denselben Kurvenscheitelpunkten verunfallte?
6. **Lauf `1432` Checkpoint:** Ist der Checkpoint bei genau Schritt 2.0M (Höhepunkt mit 3828 Punkten) separat gesichert, um ihn für Videoaufnahmen/Abbildungen rendern zu können?

### Block C: Übertragung in die 3D-Simulation (NVIDIA Isaac Sim / Isaac Lab)

7. **Stand des 3D-Setups:** Existiert bereits ein USD-Asset der Carrera-Hybrid-Strecke und des Fahrzeugs in Isaac Sim?
8. **Physik-Parameter in 3D:** Welche PhysX-Reibwerte ($\mu_{\text{static}}, \mu_{\text{dynamic}}$) und Reifenmodelle wurden für die 3D-Simulation hinterlegt?
9. **Kameramodell:** Wird in Isaac Sim eine Deckenkamera (Top-Down Orthografisch/Perspektivisch) oder eine fahrzeugfeste Onboard-Kamera simuliert, und welche Render-Pipeline (RTX Real-Time vs. Tiled Rendering) wird für die Observation-Frames verwendet?
10. **Erste Trainingsergebnisse in 3D:** Wurde der Lidar-Experte oder das vortrainierte Vision-CNN bereits in Isaac Lab getestet, und welche Konvergenzzeiten zeigen sich im Vergleich zur 2D-Python-Umgebung?

---
