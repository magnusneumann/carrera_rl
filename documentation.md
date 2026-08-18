# Projektdokumentation: Autonome Fahrzeugsteuerung für Carrera Hybrid mittels Reinforcement Learning und Privilegierter Expertendestillation

**Projekt:** Prototyping und Evaluierung von Deep Reinforcement Learning (DRL) Architekturen für ein autonomes Miniatur-Rennfahrzeug (Carrera Hybrid) in einer 2D-Simulationsumgebung  
**Autoren / Mitwirkende:** Magnus Neumann, Felix Faass  
**Status:** 2D-Prototyping, Methodenevaluation & Vorbereitung für Sim-to-Real / NVIDIA Isaac Sim  
**Repository:** [carrera_rl](file:///home/mneumann/Desktop/EigeneDateien/FE%20Carrera%20RL/carrera_rl)  

---

## Inhaltsverzeichnis
1. [Management Summary & Motivation](#1-management-summary--motivation)
2. [Wissenschaftliche Grundlagen & Stand der Technik (State of the Art)](#2-wissenschaftliche-grundlagen--stand-der-technik-state-of-the-art)
   - 2.1 MDP vs. POMDP und die Rolle des Frame Stackings
   - 2.2 Imitation Learning, Behavioral Cloning und das Problem des Covariate Shift
   - 2.3 Privilegiertes Lernen: Teacher-Student Distillation (*Learning by Cheating*)
   - 2.4 Relevante Arbeiten und Literatur-Blueprints
3. [Systemarchitektur & 2D-Simulationsumgebung (`Carrera2DEnv`)](#3-systemarchitektur--2d-simulationsumgebung-carrera2denv)
   - 3.1 Streckengeometrie & Computer Vision Pipeline (`tracklimits_cv.py`)
   - 3.2 Physikalische Modellierung: Von `force_drag_v1` zu realitätsnahem `measured_v2`
   - 3.3 Beobachtungsräume (Observation Spaces): Raycasts vs. Crop-View vs. Global-View
   - 3.4 Aktionsraum & Lenkdynamik
4. [Reward Engineering & Belohnungsdesign](#4-reward-engineering--belohnungsdesign)
   - 4.1 Mathematische Formulierung der Belohnungsfunktion
   - 4.2 Kritische Fallen & Lerneffekte: Sperrklinke, tote Zonen und Oszillationsstrafen
5. [Der 4-Phasen-Trainingsansatz (Cross-Modal Distillation)](#5-der-4-phasen-trainingsansatz-cross-modal-distillation)
   - 5.1 Phase 1: Bootstrapping des Lehrers (Lidar-Experte via PPO / SAC)
   - 5.2 Phase 2: Datensammlung mit Action Noise Injection (Lernen von Recovery-Manövern)
   - 5.3 Phase 3: Visuelles Pre-Training (Behavioral Cloning / CNN-Warmstart)
   - 5.4 Phase 4: End-to-End Fine-Tuning mittels Soft Actor-Critic (SAC)
6. [Empirische Ergebnisse & Chronologie der Trainingsläufe](#6-empirische-ergebnisse--chronologie-der-trainingsläufe)
   - 6.1 Zeitleiste der Experimente und Paradigmenwechsel
   - 6.2 Quantitative Vergleichstabelle aller Hauptläufe
   - 6.3 Der Kontrollversuch (Lauf `1748`): Beweis für das Wahrnehmungsproblem
7. [Tiefgehende RL-Erkenntnisse & Systemanalysen (Engineering Insights)](#7-tiefgehende-rl-erkenntnisse--systemanalysen-engineering-insights)
   - 7.1 Das Subpixel-Dilemma bei $166 \times 100$ Pixeln
   - 7.2 Lösungsansätze: Temporale Spreizung vs. Erhöhte Auflösung
   - 7.3 Replay-Buffer-Architektur (`framestapel_buffer.py`) & Speicheroptimierung
   - 7.4 Korrekte Behandlung von Episoden-Timeouts (`handle_timeout_termination`)
   - 7.5 Tensorboard-Diagnose: Critic-Divergenz vs. stochastische Ausreißer
   - 7.6 Stolpersteine in Messung, Multi-Env-Skalierung und Rendering
8. [Codebase-Architektur & Workflow im Notebook](#8-codebase-architektur--workflow-im-notebook)
   - 8.1 Struktur des Repositories
   - 8.2 Leitfaden durch das [Notebook_prototyping.ipynb](file:///home/mneumann/Desktop/EigeneDateien/FE%20Carrera%20RL/carrera_rl/Notebook_prototyping.ipynb)
9. [Fazit & Roadmap für den Transfer in NVIDIA Isaac Sim](#9-fazit--roadmap-für-den-transfer-in-nvidia-isaac-sim)

---

## 1. Management Summary & Motivation

Das Ziel dieses Projekts ist die Entwicklung und systematische Erforschung einer hochgradig sample-effizienten Reinforcement-Learning-Pipeline zur autonomen Steuerung eines Slotcar-Rennfahrzeugs (**Carrera Hybrid**, Maßstab ca. 1:43) am physikalischen Limit.

```
+---------------------------------------------------------------------------------------------------+
|                                  GESAMTMETHODIK IM ÜBERBLICK                                      |
|                                                                                                   |
|  [ 1. LEHRER (Raycasts/Lidar) ]                                                                   |
|      Niedrigdimensional, schnell konvergierend (Minuten) via PPO/SAC                             |
|                           |                                                                       |
|                           v (Noise Injection: Fahren mit induzierten Fehlern)                    |
|  [ 2. DATENSATZ ] ------------------------------------------------------------------+             |
|      3-Frame-Stacks + Korrekturausgaben des Lehrers                                 |             |
|                           |                                                         |             |
|                           v (Behavioral Cloning / Supervised Pretraining)           |             |
|  [ 3. SCHÜLER (Vision CNN) ]                                                        |             |
|      Erlernt robuste Merkmalsextraktion & Spurhaltung ohne Trial-and-Error          |             |
|                           |                                                         |             |
|                           v (Online RL Fine-Tuning via Soft Actor-Critic / SAC)     |             |
|  [ 4. OPTIMIERTE POLICY ] <---------------------------------------------------------+             |
|      Pixelgenaue Ideallinie & Rundenzeitoptimierung                                               |
+---------------------------------------------------------------------------------------------------+
```

### Kernproblem
Das Training eines visuellen End-to-End-Agenten (Eingabe: Kamerabild $\to$ Ausgabe: kontinuierliche Lenk- und Gasbefehle) direkt in komplexen 3D-Simulatoren (wie NVIDIA Isaac Sim / Isaac Lab) ist extrem rechenintensiv, sample-ineffizient und scheitert häufig an unstrukturiertem Random Walk in hochdimensionalen Beobachtungsräumen.

### Lösungsansatz
Zur Vermeidung teurer Trial-and-Error-Iterationen in 3D wurde eine leichtgewichtige, physikalisch validierte **2D-Simulationsumgebung in Python** aufgebaut. Darin wird das Paradigma der **Privilegierten Expertendestillation (Cross-Modal Imitation Learning / Teacher-Student Learning)** umgesetzt:
1. Ein **privilegierter Lehrer-Agent** wird auf einfachen, niedrigdimensionalen Distanzdaten (Raycasts/Lidar) in wenigen Minuten trainiert.
2. Der Lehrer generiert unter **Rausch-Injektion (Noise Injection)** Demonstrationsdaten, wodurch gezielt **Rettungsmanöver (Recovery Behaviors)** aufgezeichnet werden, um den berüchtigten **Covariate Shift** zu eliminieren.
3. Ein **visuelles Schüler-Netzwerk** (CNN mit 3-Frame-Stack) wird per Supervised Learning auf diesen Daten vorinitialisiert.
4. Das vortrainierte CNN wird abschließend über **Soft Actor-Critic (SAC)** online auf Rundenzeiten optimiert.

---

## 2. Wissenschaftliche Grundlagen & Stand der Technik (State of the Art)

### 2.1 MDP vs. POMDP und die Rolle des Frame Stackings
Im Standard-Reinforcement-Learning wird die Umwelt als **Markov-Entscheidungsprozess (MDP)** modelliert, formalisiert als Tupel $(\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$. Die Markov-Eigenschaft verlangt:
$$P(s_{t+1} \mid s_t, a_t, s_{t-1}, a_{t-1}, \dots) = P(s_{t+1} \mid s_t, a_t)$$
Erhält der Agent lediglich ein einzelnes statisches 2D-Kamerabild $o_t \in \mathbb{R}^{H \times W}$, kann er Geschwindigkeit, Beschleunigung, Schlupf und Gierrate des Fahrzeugs nicht bestimmen. Das Problem wird zu einem **Partiell Beobachtbaren MDP (POMDP)**.

**Lösung über Frame Stacking:**  
Durch das Stapeln von $k$ aufeinanderfolgenden Bildern (hier $k=3$):
$$s_t = [o_t, o_{t-1}, o_{t-2}]$$
bilden zeitliche Pixelverschiebungen (optischer Fluss) Ableitungen nach der Zeit $\frac{\partial I}{\partial t}$ ab. Das neuronale Netz kann so Geschwindigkeits- und Drehratenvektoren intern rekonstruieren und den POMDP näherungsweise in einen MDP überführen.

---

### 2.2 Imitation Learning, Behavioral Cloning und das Problem des Covariate Shift
Beim reinen **Behavioral Cloning (BC)** wird die Policy $\pi_\theta(a \mid s)$ als Regressionsproblem auf einem Datensatz $\mathcal{D} = \{(s_i, a_i^*)\}$ trainiert:
$$\mathcal{L}_{\text{BC}}(\theta) = \mathbb{E}_{(s, a^*) \sim \mathcal{D}} \left[ \| \pi_\theta(s) - a^* \|^2 \right]$$
**Das fundamentale Versagen von naivem BC (Covariate Shift / Distribution Shift):**
* Ein perfekter Experte bewegt sich ausschließlich auf der Ideallinie $\mathcal{S}_{\text{ideal}} \subset \mathcal{S}$.
* Trifft der Schüler im Autonomiebetrieb eine minimale Fehlentscheidung $\epsilon$, driftet er in Zustände $s' \notin \mathcal{S}_{\text{ideal}}$ ab (Out-of-Distribution, OOD).
* Da der Datensatz keine Demonstrationen für Rettungsaktionen aus $s'$ enthält, steigen die Fehler exponentiell mit dem Zeithorizont $T$ ($\mathcal{O}(T^2)$ nach Ross & Bagnell, 2010), was zum unvermeidlichen Streckenabflug führt.

**Gegenmaßnahme im Projekt: Rausch-Injektion (Noise Injection):**  
Dem Lehrer-Agenten wird während der Datensammlung ein stochastischer Störterm $\xi_t \sim \mathcal{N}(0, \sigma^2)$ auf die Aktion beaufschlagt. Das Fahrzeug gerät ins Schlingern, und der Lehrer muss aktiv gegensteuern. Der Schüler lernt dadurch explizit die Abfang-Dynamik:
$$\text{Zustand: } s_t (\text{verdriftet}) \implies \text{Target: } a_t^* (\text{starke Gegenlenkung / Bremsen})$$

---

### 2.3 Privilegiertes Lernen: Teacher-Student Distillation (*Learning by Cheating*)
Das Konzept des privilegierten Lernens (Chen et al., CoRL 2019) trennt das Problem in zwei orthogonale Teilaufgaben:
1. **Entscheidungsfindung (Control/Policy Problem):** Lernen, wie man fährt, unter Verfügbarkeit perfekter, rauschfreier Sensoren (Raycasts, exakte Pose, Eigengeschwindigkeit).
2. **Wahrnehmungstransfer (Perception Problem):** Abbilden von Rohpixeln auf die vom Experten vorgegebenen Aktionen.

```
                              [ PRIVILEGIERTE INFORMATIONEN ]
                               (Exakte Abstände / Raycasts)
                                             |
                                             v
                           +-----------------------------------+
                           |        Lehrer-Agent (Oracle)      |
                           |  (Kompaktes MLP, lernt in Min.)   |
                           +-----------------------------------+
                                             |
                                             v
+------------------------+             (Aktion a_t)             +-----------------------+
|  Kamerabild-Sequenz    |                   |                  |  Visueller Schüler    |
|   (3-Frame-Stack)      | ================= v ===============> | (CNN Feature-Encoder) |
+------------------------+         [ DISTILLATION / BC ]        +-----------------------+
```

---

### 2.4 Relevante Arbeiten und Literatur-Blueprints
Die Konzeption stützt sich auf folgende Schlüsselpublikationen aus dem Projekt-Fundus:

1. **Cai et al. (DIRL): *Vision-Based Autonomous Car Racing Using Deep Imitative Reinforcement Learning***  
   *Blueprint:* Kombination aus Imitation Learning zur Initialisierung und modellfreiem RL zur Feinoptimierung auf realen 1:20 RC-Cars.
2. **Chen et al.: *Learning by Cheating* (CoRL 2019)**  
   *Theoretischer Nachweis:* Zweistufige Teacher-Student-Pipelines schlagen direktes End-to-End-RL aus Bildern in Konvergenzgeschwindigkeit und Generalisierung um Größenordnungen.
3. **Wurman et al. (Sony AI): *Outracing champion Gran Turismo drivers with deep reinforcement learning* (Nature 2022)**  
   *SOTA-Beleg:* Weltklasse-Rennleistungen durch Soft Actor-Critic (SAC), Fortschritts-Rewards entlang der Streckenmittellinie und asymmetrische Trainingsstrukturen.
4. **Amini et al. (MIT): *Learning Robust Control Policies for End-to-End Autonomous Driving From Data-Driven Simulation* (VISTA)**  
   *Synthese:* Notwendigkeit der Erfassung von OOD-Trajektorien zur Überbrückung des Sim-to-Real-Gaps.

---

## 3. Systemarchitektur & 2D-Simulationsumgebung (`Carrera2DEnv`)

Die Simulationsumgebung wurde in [src/envs/carrera_2d_env.py](file:///home/mneumann/Desktop/EigeneDateien/FE%20Carrera%20RL/carrera_rl/src/envs/carrera_2d_env.py) auf Basis des Gymnasium-Standards implementiert.

```
                      +-------------------------------------------------+
                      |                 Carrera2DEnv                    |
                      +-------------------------------------------------+
                               |                               |
                +--------------+--------------+                |
                |                             |                |
                v                             v                v
      +--------------------+        +--------------------+   +-------------------+
      |   TrackProcessor   |        |   Physik-Engine    |   |  VirtualCamera    |
      |   (OpenCV/Spline)  |        |  (measured_v2)     |   |  (Render/Stack)   |
      +--------------------+        +--------------------+   +-------------------+
```

### 3.1 Streckengeometrie & Computer Vision Pipeline (`tracklimits_cv.py`)
Die reale Carrera-Hybrid-Rennstrecke besitzt standardisierte Dimensionen:
* **Fahrbahnbreite:** 250 mm
* **Randstreifen (Curbs/Begrenzungen):** 21 mm (Rot und Blau markiert)
* **Fahrzeugdimensionen:** Länge 98 mm, Breite 39 mm, Radstand: Heck bis Hinterachse 23 mm, Heck bis Vorderachse 71 mm.

In [src/utils/tracklimits_cv.py](file:///home/mneumann/Desktop/EigeneDateien/FE%20Carrera%20RL/carrera_rl/src/utils/tracklimits_cv.py) extrahiert OpenCV die Fahrbahngeometrie:
* **HSV-Filterung:** Farbmasken isolieren die roten Außen- und blauen Innenbegrenzungen.
* **Flächenschwerpunkte & Splines:** Die Innen- und Außenkonturen werden über Nearest-Neighbor-Projektionen verbunden; die interpolierten Mittelpunkte bilden die geglättete **Mittellinie**.
* **Start-/Ziellinie:** Definiert als geometrische Bounding-Box (`(462, 55)` bis `(468, 114)`).

---

### 3.2 Physikalische Modellierung: Von `force_drag_v1` zu realitätsnahem `measured_v2`

Im Projektverlauf zeigte sich, dass die Genauigkeit des physikalischen Modells über Gelingen oder Scheitern der visuellen Wahrnehmung entscheidet.

| Parameter / Eigenschaft | Frühes Modell (`force_drag_v1`) | Gemessenes Modell (`measured_v2`) | Wissenschaftliche Konsequenz |
|---|---|---|---|
| **Beschleunigung $a_{\max}$** | $12.5\text{ m/s}^2$ | $\approx 0.6 - 1.2\text{ m/s}^2$ | Altes Modell beschleunigte **~20-fach zu stark**. |
| **Verzögerung (Bremsen)** | $14.4\text{ m/s}^2$ | $\approx 1.5\text{ m/s}^2$ | Altes Modell bremste fast instantan. |
| **Maximalgeschwindigkeit $v_{\max}$** | $5.0 - 12.0\text{ m/s}$ | $1.6 - 1.9\text{ m/s}$ | Reale Carrera-Fahrzeuge fahren maximal ~1.9 m/s. |
| **Rückwärtsfahren** | Erlaubt ($v < 0$) | Verhindert ($v \ge 0$) | Realitätsabgleich: Carrera blockiert Rückwärtsgang. |
| **Lenkänderungsrate $\Delta\delta_{\max}$** | $1.0\text{ /Step}$ (Sprunghaft) | $0.5\text{ /Step}$ | Verhinderung von unphysikalischem Teleport-Lenken. |
| **Pixelversatz / Frame ($166\times 100$)** | $8 - 19\text{ Pixel/Frame}$ | **$\le 3\text{ Pixel/Frame}$ (im Start: $< 1\text{ px}$)** | **Entdeckung des Subpixel-Wahrnehmungsproblems!** |

> [!IMPORTANT]
> Das alte Modell `force_drag_v1` maskierte fundamentale Wahrnehmungsprobleme: Weil das Auto unrealistisch schnell schoss, erzeugte selbst eine geringe Auflösung signifikante Frame-zu-Frame-Pixelverschiebungen. Erst der Wechsel auf `measured_v2` legte die reale Herausforderung offen.

---

### 3.3 Beobachtungsräume (Observation Spaces)

Die Umgebung unterstützt modular konfigurierbare Beobachtungstypen:
1. **`obs_type="lidar"` (Privilegiert):**
   * Vektor $\in \mathbb{R}^4$: $[v / v_{\max}, d_{\text{links}}, d_{\text{mitte}}, d_{\text{rechts}}]$
   * 3 virtuelle Raycast-Distanzsensoren berechnen die euklidische Schnittdistanz zu den Streckenbegrenzungen.
2. **`obs_type="single_frame"`:**
   * Ein Graustufenbild ($166 \times 100 \times 1$) in 4 quantisierten Helligkeitsstufen (Ego-Crop oder Globale Ansicht).
3. **`obs_type="stacked"` (Standard für visuelles RL):**
   * $3$ gestapelte Graustufenbilder ($166 \times 100 \times 3$, bzw. höher aufgelöst $250 \times 150 \times 3$).
   * Globale Ansicht bildet die gesamte Strecke ab; das Fahrzeug ist als orientierter Sprite sichtbar.
4. **`obs_type="hybrid"`:**
   * Bildstapel fusioniert mit dem aktuellen Aktions- und Geschwindigkeitsvektor.

---

### 3.4 Aktionsraum & Lenkdynamik
Kontinuierlicher Aktionsraum $\mathcal{A} \in [-1, 1]^2$:
* **Aktion $a[0] \in [-1, 1]$ (Lenkung):** Skaliert auf maximalen Radeinschlag. Die tatsächliche Lenkwinkeländerung pro Zeitschritt wird über `max_steer_change = 0.5` dynamisch begrenzt (Tiefpass-Charakteristik des Servos).
* **Aktion $a[1] \in [-1, 1]$ (Gas / Bremse):** Positive Werte steuern die Motorkraft; negative Werte aktivieren die elektrische Bremse.

---

## 4. Reward Engineering & Belohnungsdesign

Die Gestaltung der Belohnungsfunktion $\mathcal{R}(s, a, s')$ war einer der sensibelsten Entwicklungsschritte. Im finalen Stand `v3_rundenzeit` in [src/utils/reward_func.py](file:///home/mneumann/Desktop/EigeneDateien/FE%20Carrera%20RL/carrera_rl/src/utils/reward_func.py) gilt folgende mathematische Formulierung:

### 4.1 Mathematische Formulierung
$$r_t = r_{\text{progress}} + r_{\text{speed}} + r_{\text{step}} + r_{\text{lap}} - r_{\text{collision}} - r_{\text{wall\_penalty}} - r_{\text{steer\_penalty}}$$

Wesentliche Komponenten:
1. **Streckenfortschritt (Progress):**
   $$r_{\text{progress}} = w_{\text{prog}} \cdot \Delta s_{\text{mittellinie}} \quad (\text{nur in korrekter Fahrtrichtung})$$
2. **Geschwindigkeit & Schleichstrafe:**
   $$r_{\text{speed}} = w_v \cdot \frac{v}{v_{\max}} - w_{\text{slow}} \cdot \max\left(0, 1 - \frac{v}{v_{\text{threshold}}}\right)$$
3. **Wandberührung & Crash:**
   $$r_{\text{collision}} = \begin{cases} C_{\text{crash}} & \text{falls kollidiert (Episode terminiert)} \\ 0 & \text{sonst} \end{cases}$$
4. **Rundenbonus:**
   $$r_{\text{lap}} = B_{\text{lap}} + \max(0, T_{\text{target}} - T_{\text{lap}}) \cdot w_{\text{time}}$$

---

### 4.2 Kritische Fallen & Lerneffekte

```
+---------------------------------------------------------------------------------------------------+
|                                  REWARD-FALLEN & DESIGN-ITERATIONEN                               |
+---------------------------------------------------------------------------------------------------+
|  1. DIE SPERRKLINKE (Ratchet Bug):                                                                |
|     Alter Reward belohnte jede neue Bestzeit, bestrafte aber langsamere Runden drastisch.         |
|     Folge: Eine einzige schnelle Zufallsrunde machte alle Folge-Runden netto negativ (-250).      |
|     Lösung: Entfernung der relativen Rundenzeithistorie in v3.                                    |
|                                                                                                   |
|  2. DIE TOTE ZONE:                                                                                |
|     Harte Stufenfunktion bei v < 0.6 m/s führte zu Gradienten-Klippen.                            |
|     Lösung: Ersetzung durch kontinuierliches Gefälle.                                             |
|                                                                                                   |
|  3. STRAFEN AUF GUTES VERHALTEN (w_integral = 0.08):                                              |
|     Der Oszillationsschutz bestrafte jede kontinuierliche Lenkbewegung in Kurven.                 |
|     Folge: Der perfekte Lidar-Experte zahlte -625 Punkte Strafe über 2000 Frames.                 |
|     Lösung: w_integral wurde auf 0.0 gesetzt.                                                     |
+---------------------------------------------------------------------------------------------------+
```

---

## 5. Der 4-Phasen-Trainingsansatz (Cross-Modal Distillation)

Das Zusammenspiel der Komponenten wird in [Notebook_prototyping.ipynb](file:///home/mneumann/Desktop/EigeneDateien/FE%20Carrera%20RL/carrera_rl/Notebook_prototyping.ipynb) realisiert:

```
[ Phase 1: Lidar-Lehrer ]  -->  [ Phase 2: Noisy Rollouts ]  -->  [ Phase 3: BC Pretrain ]  -->  [ Phase 4: SAC Fine-Tuning ]
   (PPO auf Raycasts)              (Recovery Trajektorien)            (CNN Feature-Warmstart)          (High-Performance RL)
```

### 5.1 Phase 1: Bootstrapping des Lehrers (Lidar-Experte)
* **Algorithmus:** PPO / SAC mit MLP-Policy (`net_arch=[64, 64]`).
* **Input:** $4$ Dimensionen (3 Raycasts + Eigengeschwindigkeit).
* **Performance:** Erreicht nach nur **50.000 Schritten (< 15 Minuten)** volle 2000 Frames Episodenlänge und fährt über 11 fehlerfreie Runden am Stück mit $\bar{v} = 1.22\text{ m/s}$.

### 5.2 Phase 2: Datensammlung mit Action Noise Injection
* Der Lidar-Experte steuert das Fahrzeug in der Simulation.
* Auf die Expertenaktion $a_t^*$ wird Gaußsches Rauschen addiert:
  $$\tilde{a}_t = \text{clip}(a_t^* + \mathcal{N}(0, \sigma^2), -1, 1)$$
* **Essenz:** Weicht das Fahrzeug durch $\tilde{a}_t$ von der Ideallinie ab, berechnet der Lidar-Experte im nächsten Zeitschritt die korrigierende Rettungsaktion $a_{t+1}^*$.
* **Speicherung:** Pro Zeitschritt werden der 3-Frame-Stack $s_t \in \mathbb{R}^{3 \times 100 \times 166}$ und die unverrauschte Soll-Aktion $a_t^*$ in einem PyTorch-Tensor-Datensatz gespeichert ($12.000$ bis $50.000$ Samples).

### 5.3 Phase 3: Visuelles Pre-Training (Behavioral Cloning)
* Ein Convolutional Neural Network (Nature-CNN-Architektur oder benutzerdefiniertes 3-Layer-CNN mit Flatten- und Dense-Layers) wird trainiert:
  $$\min_\theta \frac{1}{|\mathcal{D}|} \sum_{(s, a^*) \in \mathcal{D}} \| f_\theta(s) - a^* \|^2$$
* **Ergebnis:** Das Modell erreicht nach 20 Epochen einen Mean Squared Error (MSE) von $< 0.0037$ und erlernt robuste visuelle Filter für Fahrbahnränder und Fahrzeugorientierung.

### 5.4 Phase 4: End-to-End Fine-Tuning mittels Soft Actor-Critic (SAC)
* Die vortrainierten CNN-Gewichte werden in die SAC-Policy (`Actor` und `Critic`) geladen.
* **SAC-Mechanik:**
  * Off-Policy Actor-Critic mit Replay-Buffer.
  * Maximierung des Maximum-Entropy-Ziels:
    $$J(\pi) = \sum_{t=0}^T \mathbb{E}_{(s_t, a_t)} \left[ r(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot \mid s_t)) \right]$$
  * Verhindert verfrühte Konvergenz auf suboptimale Bremsstrategien und nutzt Erfahrungswerte im Buffer mit hoher Sample-Effizienz.

---

## 6. Empirische Ergebnisse & Chronologie der Trainingsläufe

### 6.1 Zeitleiste der Experimente und Paradigmenwechsel
Die Chronologie (detailliert in [zeitleiste.md](file:///home/mneumann/Desktop/EigeneDateien/FE%20Carrera%20RL/carrera_rl/zeitleiste.md)) teilt sich in methodisch getrennte Phasen:

```
[ Phase 1: force_drag_v1 ]  --->  [ Phase 2: measured_v2 ]  --->  [ Phase 3: gamma 0.999 ]  --->  [ Phase 4: Kontrollversuch ]
      (Alte Physik)                    (Reale Physik)                    (Kollaps)                     (Lidar-SAC Beweis)
```

1. **Phase 1 (Juni bis 24.07. – Altes Modell `force_drag_v1`):**  
   Fahrzeug 20-fach übermotorisiert. Lauf `1432` erzielte mit 3828.2 Punkten und 2000 Frames scheinbar hervorragende Ergebnisse, stürzte jedoch im weiteren Trainingsverlauf ab.
2. **Phase 2 (28.–29.07. – Gemessene Physik `measured_v2` & Reward v3):**  
   Scharfer Schnitt. Das Auto verhält sich nun realitätsgetreu. Die Bild-Agenten (`2147`, `2210`) erreichen keine stabilen Runden mehr.
3. **Phase 3 (31.07. – $\gamma$-Experiment, Lauf `2331`):**  
   Erhöhung von $\gamma = 0.99 \to 0.999$ bei unveränderter Lernrate führte zur Explosion der Q-Werte und zum sofortigen Kollaps der Policy nach 250k Steps.
4. **Phase 4 (14.08. – Der Kontrollversuch `1748`):**  
   Isolierter Test: SAC unter `measured_v2` und Reward v3, jedoch auf Lidar-Sensoren. **Voller Erfolg (6116.4 Punkte, 11 Runden, 2000 Frames)**.

---

### 6.2 Quantitative Vergleichstabelle aller Hauptläufe

Alle Modelle wurden unter einheitlichen Testbedingungen im aktuellen Evaluationsmodul ([src/eval/vergleich.py](file:///home/mneumann/Desktop/EigeneDateien/FE%20Carrera%20RL/carrera_rl/src/eval/vergleich.py)) neu vermessen:

| Modell / Run | Algorithmus | Beobachtung | Physikmodell | Total Steps | Reward (Eval) | EpLen (Frames) | Runden | Ø-Tempo | Status / Befund |
|---|---|---|---|---|---|---|---|---|---|
| **Lidar-Experte** | PPO | Raycasts (4D) | `measured_v2` | 500 k | **6401.7** | **2000** (Limit) | **11** | $1.22\text{ m/s}$ | Souveräne Referenz |
| **Lidar SAC (`1748`)** | SAC | Raycasts (4D) | `measured_v2` | 500 k | **6116.4** | **2000** (Limit) | **11** | $1.19\text{ m/s}$ | **Kontrollversuch bestanden** |
| **Vision SAC `1432`** | SAC | 3-Stack ($166\times 100$) | `force_drag_v1` | 2.5 M | **3828.2** | **2000** | 3 | $1.27\text{ m/s}$ | Bestwert alte Physik; divergiert später |
| **Vision SAC `2210`** | SAC | 3-Stack ($166\times 100$) | `measured_v2` | 5.0 M | $-30.3$ | 256 | 0 | $0.81\text{ m/s}$ | Scheitert an Wahrnehmungsschwelle |
| **Vision SAC `2147`** | SAC | 3-Stack ($166\times 100$) | `measured_v2` | 2.5 M | $-266.5$ | 174 | 0 | $0.62\text{ m/s}$ | Frühzeitiger Crash |
| **Vision SAC `2331`** | SAC | 3-Stack ($166\times 100$) | `measured_v2` | 1.5 M | $-349.8$ | 33 | 0 | $0.12\text{ m/s}$ | Kollaps durch $\gamma = 0.999$ |
| **Vision PPO `2328`** | PPO | 3-Stack ($166\times 100$) | `force_drag_v1` | 90 M | $-3.9$ | 76 | 0 | $0.45\text{ m/s}$ | Sample-Ineffizienz von On-Policy PPO |

---

### 6.3 Der Kontrollversuch (Lauf `1748`): Beweis für das Wahrnehmungsproblem

Der Kontrollversuch vom 14.08. lieferte die entscheidende wissenschaftliche Erkenntnis des Projekts:
* **Frage:** Scheitern die visuellen Agenten an algorithmischer Instabilität von SAC, an der Trägheit von `measured_v2` oder an der Belohnungsfunktion?
* **Experiment:** Alle Hyperparameter, Netzwerktiefen, Buffer-Einstellungen und physikalischen Randbedingungen blieben exakt identisch – lediglich die Beobachtung wurde von $166 \times 100$ Pixeln auf $4$ Lidar-Werte umgestellt.
* **Ergebnis:** Der Agent meisterte die Strecke in unter **54 Minuten Trainingszeit**.
* **Wissenschaftlicher Schluss:** Algorithmus, Physik, Reward, Puffergrößen und Diskontierungsfaktoren sind vollkommen intakt. **Der Engpass liegt ausschließlich in der Bildauflösung und der zeitlichen Abtastung der visuellen Sensorik.**

---

## 7. Tiefgehende RL-Erkenntnisse & Systemanalysen (Engineering Insights)

Ausführlich dokumentiert in [rl_erkenntnisse.md](file:///home/mneumann/Desktop/EigeneDateien/FE%20Carrera%20RL/carrera_rl/rl_erkenntnisse.md):

### 7.1 Das Subpixel-Dilemma bei $166 \times 100$ Pixeln

```
+---------------------------------------------------------------------------------------------------+
|                                     DAS SUBPIXEL-PROBLEM                                          |
|                                                                                                   |
|  Reale Streckenbreite: 1.80 m  -->  Kamerabild: 166 Pixel Breite  ===>  1 Pixel ≈ 10.8 mm        |
|  Simulationsschritt: dt = 0.0167 s (60 Hz)                                                        |
|                                                                                                   |
|  Mindestgeschwindigkeit für 1 Pixel Versatz / Step:                                              |
|      v_min = 0.0108 m / 0.0167 s = 0.64 m/s                                                       |
|                                                                                                   |
|  Konsequenz im Anfahrbereich (v < 0.64 m/s):                                                      |
|      Frame(t) == Frame(t-1) == Frame(t-2)  ==>  Kein optischer Fluss!                             |
|      Der Agent ist "blind" für seine eigene Bewegung und wird zeitgleich vom Reward bestraft.     |
+---------------------------------------------------------------------------------------------------+
```

---

### 7.2 Lösungsansätze: Temporale Spreizung vs. Erhöhte Auflösung

Zwei architektonische Hebel heben die Subpixel-Blockade auf:

```
Hebel 1: Temporale Spreizung (Frame Striding)
    Statt [t, t-1, t-2] speichere [t, t-4, t-8]
    -> 4-facher zeitlicher Hebel!
    -> Wahrnehmungsschwelle sinkt von 0.64 m/s auf 0.16 m/s
    -> Kosten: 0% Rechenzeit, 0% Extra-Speicher! (Favorit)

Hebel 2: Erhöhte Bildauflösung (250 x 150 Pixel)
    -> 1 Pixel entspricht nur noch 7.2 mm
    -> Wahrnehmungsschwelle sinkt von 0.64 m/s auf 0.42 m/s
    -> Kosten: +89% Rechenzeit je Update, Buffer wächst von 15.5 auf 34.9 GB
```

---

### 7.3 Replay-Buffer-Architektur (`framestapel_buffer.py`) & Speicheroptimierung

Standard-Implementierungen von `VecFrameStack` legen für jeden Übergang $(s_t, a_t, r_t, s_{t+1})$ alle 3 gestapelten Bilder redundant ab. Bei 500.000 Schritten führte dies zu untragbaren Speicherlasten (> 23 GB RAM).

In [src/utils/framestapel_buffer.py](file:///home/mneumann/Desktop/EigeneDateien/FE%20Carrera%20RL/carrera_rl/src/utils/framestapel_buffer.py) wurde ein spezialisierter Ringpuffer entwickelt:
* **Speicherung:** Pro Zeitschritt wird nur das **aktuelle Einzelbild** (1 Kanal, uint8) im Ring abgelegt.
* **On-the-Fly Assembly:** Beim Ziehen eines Batches (`sample()`) werden die Indizes $i, i-1, i-2$ dynamisch zu einem 3-Kanal-Tensor assembliert.
* **Episodengrenzen-Schutz:** Überschreitet ein Index eine Episodengrenze (`done=True`), wird das älteste Bild der neuen Episode dupliziert, um Kontaminationen aus Vor-Episoden mathematisch strikt auszuschließen.
* **Ergebnis:** Reduktion des Speicherbedarfs um **Faktor 1.5** (von 23.2 GB auf 15.5 GB) bei **bitidentischen Batches**.

---

### 7.4 Korrekte Behandlung von Episoden-Timeouts (`handle_timeout_termination`)

Erreicht ein Agent das Zeitschrittlimit von 2000 Frames, liefert Gymnasium `truncated = True`.
* **Klassischer Fehler:** Wird dieser Übergang wie ein regulärer Crash (`terminated = True`) behandelt, setzt der Bellman-Operator den Folgezustandswert auf 0:
  $$y_t = r_t + \gamma \cdot 0$$
  Der Critic lernt fälschlicherweise, dass der Zustand bei Frame 1999 wertlos ist.
* **Korrektur:** Mit `handle_timeout_termination=True` wird der Wert des Folgezustands korrekt gebootstrappt:
  $$y_t = r_t + \gamma \cdot V(s_{t+1})$$

---

### 7.5 Tensorboard-Diagnose: Critic-Divergenz vs. stochastische Ausreißer

Die Analyse der Tensorboard-Logs deckte zwei fundamentale Phänomene auf:
1. **Lauf `1432` (Alte Physik): Echte Critic-Divergenz**  
   Unmittelbar nach dem Höchstwert explodierte der Critic-Loss von $22.2$ auf $26.670$ (Faktor 1200). Das Q-Netzwerk überschätzte Werte maßlos, wodurch die Policy destabilisiert wurde.
2. **Lauf `2210` (Neue Physik): Scheinbarer Peak war ein Ausreißer**  
   Während die deterministische Evaluation einen Peak von 875 Punkten meldete, verharrte der stochastische Trainings-Reward (`rollout/ep_rew_mean`) konstant bei $-300$. Der Peak war kein echter Fähigkeitserwerb, sondern ein deterministischer Einzelfall-Glückstreffer.

> [!TIP]
> Wissenschaftliche Best Practice: Eine Spitzen-Auswertung (`best`) darf niemals isoliert betrachtet werden. Nur wenn `rollout/ep_rew_mean` im Tensorboard parallel ansteigt, liegt echtes Policy-Lernen vor.

---

### 7.6 Stolpersteine in Messung, Multi-Env-Skalierung und Rendering

1. **Rendering-Diskrepanz:**  
   Training lief mit `render_mode="hidden"`, Visualisierung mit `render_mode="human"`. Antialiasing-Unterschiede von lediglich **0.05% der Pixel** reichten aus, um die Lenkentscheidung des CNNs von $+0.30$ auf $-0.80$ umzupolen. Evaluationen müssen zwingend denselben Bildpfad wie das Training nutzen!
2. **Gradient Steps bei Multi-Envs:**  
   Bei $N=12$ parallelen Subprozess-Umgebungen muss `gradient_steps` proportional skaliert werden, da sonst pro gesammeltem Zeitschritt $N$-fach zu wenige Netzwerk-Updates erfolgen.
3. **`DummyVecEnv` Auto-Reset:**  
   Der automatische Reset bei Episodenende überschreibt den finalen Crash-Frame. Für Fehleranalysen muss der letzte Pufferzustand vor dem Reset abgegriffen werden.

---

## 8. Codebase-Architektur & Workflow im Notebook

### 8.1 Struktur des Repositories

```
carrera_rl/
│
├── Notebook_prototyping.ipynb     # Zentrales interaktives Arbeitsbuch (Pipeline, Training, Eval)
├── academic_background.md         # Wissenschaftliche Einordnung & Literaturbezug
├── goals.md                       # Projektziele & Anforderungskatalog
├── zeitleiste.md                  # Vollständiges Protokoll aller Trainingsläufe
├── rl_erkenntnisse.md             # Detaillierte mathematische & technische Detailanalysen
├── requirements.txt               # Python-Abhängigkeiten (Torch, Gymnasium, SB3 etc.)
│
├── configs/                       # Hyperparameter- und Umgebungskonfigurationen
├── data/                          # Generierte Demonstrationsdatensätze (BC-Pretraining)
├── models/                        # Gespeicherte Checkpoints, train_config.json, evaluations.npz
├── tensorboard_logs/              # Diagnose-Logs (Critic-Loss, Entropie, EpLen)
├── Quellen und Paper/             # Wissenschaftliche Literatur (DIRL, LBC, GT Sophy, VISTA)
│
└── src/
    ├── envs/
    │   └── carrera_2d_env.py      # Gymnasium-Umgebung mit 2D-Physik, Lidar & VirtualCamera
    ├── train/
    │   ├── lidar_experte.py       # PPO/SAC Trainingsskripte für den Lehrer-Agenten
    │   └── framestapel_buffer.py  # Speicheroptimierter Ringpuffer für Frame-Stacks
    ├── utils/
    │   ├── tracklimits_cv.py      # OpenCV-Fahrbahn- und Mittelliniengenerierung
    │   ├── virtual_camera.py      # Bildquantisierung (4 Graustufen) & Rendering
    │   ├── reward_func.py         # Belohnungsfunktionen (v1 bis v3_rundenzeit)
    │   └── benchmark.py           # GFLOPS-Schätzung, FPS-Messung & Rundenzeitanalyse
    └── eval/
        ├── vergleich.py           # Standardisierter Re-Evaluation-Runner für alle Modelle
        └── ablation_lidar.py      # Ablationsstudien zur Sensorik
```

---

### 8.2 Leitfaden durch das `Notebook_prototyping.ipynb`

Das Jupyter Notebook [Notebook_prototyping.ipynb](file:///home/mneumann/Desktop/EigeneDateien/FE%20Carrera%20RL/carrera_rl/Notebook_prototyping.ipynb) bildet die vollständige Entwicklungskette interaktiv ab:

```
Zellen  0 - 15:  [ Initialisierung & Streckenextraktion ]
                 - Laden des Streckenbildes, interaktives HSV-Tuning
                 - Berechnung der Mittellinie und Streckenbegrenzungen via OpenCV
                 - Manuelles Fahren mit WASD zur Validierung der Fahrphysik

Zellen 16 - 26:  [ Environment-Test & Virtuelle Kamera ]
                 - Initialisierung von Carrera2DEnv, Lidar-Visualisierung
                 - Konfiguration der VirtualCamera (Crop vs. Global, 4 Graustufen)

Zellen 27 - 49:  [ Baseline-Trainings & Modalitäten-Vergleich ]
                 - Training: Nur Raycasts vs. Single-Frame vs. Hybrid vs. Stacked Crop
                 - Benchmarking der Inferenz-GFLOPS und Frame-Raten

Zellen 50 - 63:  [ Die Teacher-Student Pipeline ]
                 - Generierung des Noisy-Lidar-Demonstrationsdatensatzes
                 - Supervised Pretraining des CNN-Backbones (MSE-Loss)
                 - Großes SAC-Feintuning auf gestapelten globalen Bildern
                 - Automatisierter Vergleich aller Trainingsruns

Zellen 64 - 70:  [ Diagnose & Forensische Beobachtung ]
                 - Beobachtung beliebiger historischer Modelle unter ihrer Original-Physik
                 - Kontrollversuch: SAC auf Lidar unter measured_v2
                 - Echtzeit-Telemetrieanzeige (Geschwindigkeit, Lenkung, Lidar-Balken, Agenten-Sicht)
```

---

## 9. Fazit & Roadmap für den Transfer in NVIDIA Isaac Sim

### 9.1 Zusammenfassung der 2D-Prototyping-Phase
* **Methodische Validierung:** Das Konzept der **Privilegierten Expertendestillation (Teacher-Student)** hat sich als hochgradig überlegen erwiesen. Der Lehrer lernt in Minuten; das Vortraining stabilisiert den visuellen Agenten von Beginn an.
* **Physikalische Sensitivität:** Die realitätsgetreue Kalibrierung des Fahrzeugmodells (`measured_v2`) deckte das **Subpixel-Wahrnehmungsproblem** auf, welches bei künstlich überhöhter Physik verborgen geblieben wäre.
* **Algorithmischer Nachweis:** Der Kontrollversuch bewies, dass **Soft Actor-Critic (SAC)** stabil konvergiert, sofern die Wahrnehmungsauflösung und temporale Abtastung an die Fahrzeugdynamik angepasst sind.

---

### 9.2 Handlungsempfehlungen & Roadmap für NVIDIA Isaac Sim / Isaac Lab

Für die anstehende Übertragung in die 3D-Simulation und das spätere Sim-to-Real-Deployment auf realen Carrera-Hybrid-Fahrzeugen ergeben sich folgende Kernempfehlungen:

```
+---------------------------------------------------------------------------------------------------+
|                        ROADMAP: TRANSFER IN ISAAC SIM & SIM-TO-REAL                               |
+---------------------------------------------------------------------------------------------------+
|                                                                                                   |
|  1. TEMPORALE STRIDING-PIPELINE IMPLEMENTIEREN:                                                   |
|     Aufbau der Frame-Stacking-Pipeline mit Striding [t, t-4, t-8] in Isaac Lab, um                |
|     Subpixel-Blindheit bei niedrigen Geschwindigkeiten von Beginn an auszuschließen.              |
|                                                                                                   |
|  2. ASYMMETRISCHER ACTOR-CRITIC IN 3D:                                                            |
|     Nutzung der vollen 3D-Ground-Truth-Zustände (exakte Position, Gierrate, Radschlupf)           |
|     für das Training des Critic-Netzwerks, während der Actor ausschließlich Pixel und             |
|     Telemetriedaten erhält (Asymmetric Actor-Critic / GT Sophy Blueprint).                        |
|                                                                                                   |
|  3. DOMAIN RANDOMIZATION GEGEN DEN SIM-TO-REAL-GAP:                                               |
|     Randomisierung von:                                                                           |
|     - Reibwerten (Fahrbahnhaftung, Schlupf in Kurven)                                             |
|     - Beleuchtungsverhältnissen und Reflexionen auf der Strecke                                   |
|     - Kamerarauschen, Latenzen und Unschärfen                                                     |
|     - Fahrzeugmasse und Schwerpunktlage                                                           |
|                                                                                                   |
|  4. ROBUSTE NOISE INJECTION ZUR DATENSYNTHESE:                                                    |
|     Generierung von Millionen synthetischer Recovery-Frames aus 3D-Kamerawinkeln über             |
|     den privilegierten 3D-Raycast-Lehrer zur optimalen Initialisierung des visuellen Modells.     |
+---------------------------------------------------------------------------------------------------+
```

---
*Dokumentation generiert auf Basis des Projektstands August 2026. Alle mathematischen Modelle, Parameter und Code-Referenzen sind im Repository [carrera_rl](file:///home/mneumann/Desktop/EigeneDateien/FE%20Carrera%20RL/carrera_rl) verifiziert.*
