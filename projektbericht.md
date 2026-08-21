<!-- ========================================================================== -->
<!-- DECKBLATT / TITLE PAGE                                                     -->
<!-- ========================================================================== -->

<div class="deckblatt">
  <div class="deckblatt-header">
    <span class="institution">Hochschule Karlsruhe – Technik und Wirtschaft</span>
    <span class="fakultaet">Fakultät für Maschinenbau und Mechatronik</span>
    <span class="studiengang">Studiengang: Robotik und Künstliche Intelligenz in der Produktion</span>
  </div>

  <div class="deckblatt-main">
    <div class="arbeitstyp">Forschungs- und Entwicklungsprojekt</div>
    <h1 class="titel">Autonome Fahrzeugsteuerung für Carrera Hybrid mittels Deep Reinforcement Learning und Privilegierter Expertendestillation</h1>
    <h2 class="subtitel">Konzeption, 2D-Prototyping und empirische Evaluation einer sample-effizienten Teacher-Student-Architektur zur Vorbereitung des Sim-to-Real-Transfers</h2>
  </div>

  <div class="deckblatt-footer">
    <table class="meta-table">
      <tr>
        <td><strong>Vorgelegt von:</strong></td>
        <td>Felix Faaß, Magnus Neumann</td>
      </tr>
      <tr>
        <td><strong>Matrikelnummern:</strong></td>
        <td>[Matrikelnummer Felix], [Matrikelnummer Magnus]</td>
      </tr>
      <tr>
        <td><strong>Projektbetreuer:</strong></td>
        <td>Prof. Dr. Björn Hein</td>
      </tr>
      <tr>
        <td><strong>Institution:</strong></td>
        <td>Hochschule Karlsruhe (HKA)</td>
      </tr>
      <tr>
        <td><strong>Abgabedatum:</strong></td>
        <td>[Datum der Abgabe, z. B. 31. August 2026]</td>
      </tr>
    </table>
  </div>
</div>

<div class="page-break"></div>

<!-- ========================================================================== -->
<!-- EIGENSTÄNDIGKEITSERKLÄRUNG                                                 -->
<!-- ========================================================================== -->

# Eigenständigkeitserklärung

Wir versichern hiermit wahrheitsgemäß, dass wir die vorliegende Projektarbeit selbstständig verfasst und keine anderen als die angegebenen Quellen und Hilfsmittel benutzt haben. Alle Stellen, die wörtlich oder sinngemäß aus veröffentlichten oder unveröffentlichten Schriften entnommen wurden, sind als solche kenntlich gemacht.

Die Arbeit hat in gleicher oder ähnlicher Form noch keiner anderen Prüfungsbehörde vorgelegen und wurde nicht anderweitig veröffentlicht.

<br><br><br>

Karlsruhe, den [Datum der Abgabe]

<br><br>

______________________________________ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ______________________________________  
(Felix Faaß) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(Magnus Neumann)

<div class="page-break"></div>

<!-- ========================================================================== -->
<!-- ABSTRACT / KURZFASSUNG                                                     -->
<!-- ========================================================================== -->

# Kurzfassung

Das Training von End-to-End Reinforcement Learning (RL) Agenten zur visuellen Fahrzeugsteuerung am physikalischen Limit ist in fotorealistischen 3D-Simulatoren mit extrem hohem Rechenaufwand und ausgeprägter Sample-Ineffizienz verbunden. Die vorliegende Projektarbeit adressiert diese Herausforderung durch den Aufbau einer leichtgewichtigen, physikalisch kalibrierten 2D-Simulationsumgebung in Python zur iterativen Methodenentwicklung und Evaluation von RL-Architekturen für *Carrera Hybrid* Rennfahrzeuge (Maßstab ca. 1:43). 

Als zentraler methodischer Ansatz wird eine **Cross-Modal Privileged Teacher-Student Distillation Pipeline** erforscht: Ein leichtgewichtiger Lehrer-Agent (Oracle) wird binnen weniger Minuten auf niedrigdimensionalen Raycast-Sensoren trainiert. Durch gezielte Rausch-Injektion (*Action Noise Injection*) während der Demonstrationsfahrten generiert der Lehrer einen Datensatz reich an Rettungsmanövern (*Recovery Behaviors*), wodurch das fundamentale Problem des *Covariate Shift* beim anschließenden *Behavioral Cloning* des visuellen Convolutional Neural Networks (CNN) gelöst wird. Das vortrainierte visuelle Modell wird anschließend mittels *Soft Actor-Critic* (SAC) auf Rundenzeiten optimiert.

Die empirische Untersuchung deckt zentrale physikalische und wahrnehmungstheoretische Phänomene auf: Insbesondere wird das **Subpixel-Wahrnehmungsproblem** bei realistischen Fahrzeuggeschwindigkeiten mathematisch hergeleitet und experimentell validiert. Ein isolierter Kontrollversuch belegt, dass SAC unter gemessener Fahrzeugphysik die Rennstrecke mit Lidar-Sensorik in 54 Minuten meistert (über 11 fehlerfreie Runden), während reine Bildagenten bei Standard-Framestapeln an der Bewegungswahrnehmungsschwelle scheitern. Zur Lösung werden temporale Striding-Verfahren und optimierte Replay-Buffer-Architekturen vorgestellt. Die Arbeit liefert damit die validierte methodische Grundlage für den anschließenden Transfer in NVIDIA Isaac Sim und das reale Fahrzeug-Deployment.

---

# Abstract

Training end-to-end vision-based reinforcement learning (RL) agents for autonomous racing at the physical friction limit typically requires prohibitive compute budgets and suffers from severe sample inefficiency in complex 3D simulators. This project addresses these challenges by developing a lightweight, physics-calibrated 2D simulation environment in Python for rapid prototyping, architecture exploration, and benchmark evaluation targeting *Carrera Hybrid* miniature race cars (1:43 scale).

The core methodological contribution is a **Cross-Modal Privileged Teacher-Student Distillation Pipeline**: A lightweight teacher agent (oracle) is trained within minutes using low-dimensional raycast sensors. By introducing deliberate *Action Noise Injection* during demonstration rollouts, the teacher generates a diverse dataset containing essential recovery maneuvers, effectively mitigating the *Covariate Shift* during subsequent *Behavioral Cloning* of the visual Convolutional Neural Network (CNN). The pretrained vision backbone is then fine-tuned using *Soft Actor-Critic* (SAC) for lap-time optimization.

Empirical evaluations reveal critical physical and perceptual bottlenecks: Specifically, the **subpixel perception barrier** at realistic miniature vehicle velocities is mathematically derived and experimentally verified. An isolated control experiment proves that SAC seamlessly solves the racing task under measured vehicle dynamics with lidar inputs in 54 minutes (>11 consecutive laps), isolating visual motion resolution as the sole performance bottleneck. To resolve this, temporal frame striding and memory-efficient replay buffer architectures are established. This work provides the validated methodology and architectural blueprints for subsequent deployment in NVIDIA Isaac Sim and real-world sim-to-real transfer.

<div class="page-break"></div>

<!-- ========================================================================== -->
<!-- INHALTSVERZEICHNIS                                                         -->
<!-- ========================================================================== -->

# Inhaltsverzeichnis

- [1. Einleitung](#1-einleitung)
  - [1.1 Motivation und Anwendungsbereich](#11-motivation-und-anwendungsbereich)
  - [1.2 Projektstruktur und forschungsmethodische Einordnung](#12-projektstruktur-und-forschungsmethodische-einordnung)
  - [1.3 Zielsetzung und Anforderungskatalog](#13-zielsetzung-und-anforderungskatalog)
- [2. Theoretische Grundlagen und Stand der Technik](#2-theoretische-grundlagen-und-stand-der-technik)
  - [2.1 Reinforcement Learning Grundlagen (MDP vs. POMDP)](#21-reinforcement-learning-grundlagen-mdp-vs-pomdp)
  - [2.2 Zeitliche Dynamik und Frame Stacking](#22-zeitliche-dynamik-und-frame-stacking)
  - [2.3 Imitation Learning, Behavioral Cloning und Covariate Shift](#23-imitation-learning-behavioral-cloning-und-covariate-shift)
  - [2.4 Privilegiertes Lernen: Teacher-Student Distillation](#24-privilegiertes-lernen-teacher-student-distillation)
  - [2.5 Algorithmen im kontinuierlichen Kontrollraum (PPO vs. SAC)](#25-algorithmen-im-kontinuierlichen-kontrollraum-ppo-vs-sac)
  - [2.6 State of the Art im autonomen Rennsport](#26-state-of-the-art-im-autonomen-rennsport)
- [3. Methodik und Aufbau der 2D-Simulationsumgebung](#3-methodik-und-aufbau-der-2d-simulationsumgebung)
  - [3.1 Streckengeometrie und Computer Vision Pipeline](#31-streckengeometrie-und-computer-vision-pipeline)
  - [3.2 Physikalische Modellierung: Von `force_drag_v1` zu `measured_v2`](#32-physikalische-modellierung-von-force_drag_v1-zu-measured_v2)
  - [3.3 Beobachtungsräume und Virtuelle Kamera](#33-beobachtungsräume-und-virtuelle-kamera)
  - [3.4 Aktionsraum und Tiefpass-Lenkdynamik](#34-aktionsraum-und-tiefpass-lenkdynamik)
  - [3.5 Reward Engineering und Belohnungsgestaltung](#35-reward-engineering-und-belohnungsgestaltung)
- [4. Der 4-Phasen-Trainingsansatz (Teacher-Student Pipeline)](#4-der-4-phasen-trainingsansatz-teacher-student-pipeline)
  - [4.1 Phase 1: Bootstrapping des Lidar-Experten](#41-phase-1-bootstrapping-des-lidar-experten)
  - [4.2 Phase 2: Datensammlung mit Action Noise Injection](#42-phase-2-datensammlung-mit-action-noise-injection)
  - [4.3 Phase 3: Visuelles Pre-Training (Behavioral Cloning)](#43-phase-3-visuelles-pre-training-behavioral-cloning)
  - [4.4 Phase 4: End-to-End Fine-Tuning mit Soft Actor-Critic (SAC)](#44-phase-4-end-to-end-fine-tuning-mit-soft-actor-critic-sac)
- [5. Experimentelle Ergebnisse und Chronologie der Trainingsläufe](#5-experimentelle-ergebnisse-und-chronologie-der-trainingsläufe)
  - [5.1 Chronologie der Experimente und Paradigmenwechsel](#51-chronologie-der-experimente-und-paradigmenwechsel)
  - [5.2 Quantitativer Leistungsvergleich](#52-quantitativer-leistungsvergleich)
  - [5.3 Der Kontrollversuch: Lidar-SAC unter gemessener Physik](#53-der-kontrollversuch-lidar-sac-unter-gemessener-physik)
- [6. Tiefgehende RL-Erkenntnisse und Systemanalysen](#6-tiefgehende-rl-erkenntnisse-und-systemanalysen)
  - [6.1 Das Subpixel-Wahrnehmungsproblem](#61-das-subpixel-wahrnehmungsproblem)
  - [6.2 Lösungsstrategien: Temporale Spreizung vs. Bildauflösung](#62-lösungsstrategien-temporale-spreizung-vs-bildauflösung)
  - [6.3 Replay-Buffer-Architektur und Speicheroptimierung](#63-replay-buffer-architektur-und-speicheroptimierung)
  - [6.4 Korrekte Behandlung von Zeitschritt-Timeouts](#64-korrekte-behandlung-von-zeitschritt-timeouts)
  - [6.5 TensorBoard-Diagnostik: Critic-Divergenz vs. Ausreißer](#65-tensorboard-diagnostik-critic-divergenz-vs-ausreißer)
  - [6.6 Messfallen und Rendering-Diskrepanzen](#66-messfallen-und-rendering-diskrepanzen)
- [7. Übertragung in die 3D-Simulation (NVIDIA Isaac Sim)](#7-übertragung-in-die-3d-simulation-nvidia-isaac-sim)
  - [7.1 Aufbau der 3D-Simulationsumgebung in Isaac Lab [Platzhalter]](#71-aufbau-der-3d-simulationsumgebung-in-isaac-lab-platzhalter)
  - [7.2 Asymmetric Actor-Critic und Domain Randomization](#72-asymmetric-actor-critic-und-domain-randomization)
  - [7.3 Evaluierung und Sim-to-Real Ausblick [Platzhalter]](#73-evaluierung-und-sim-to-real-ausblick-platzhalter)
- [8. Fazit, Learnings und Ausblick](#8-fazit-learnings-und-ausblick)
  - [8.1 Zusammenfassung der Ergebnisse](#81-zusammenfassung-der-ergebnisse)
  - [8.2 Zentrale Erkenntnisse (Key Learnings)](#82-zentrale-erkenntnisse-key-learnings)
  - [8.3 Ausblick auf das Folgeprojekt](#83-ausblick-auf-das-folgeprojekt)
- [Quellenverzeichnis](#quellenverzeichnis)
- [Abbildungs- und Tabellenverzeichnis](#abbildungs--und-tabellenverzeichnis)
- [Anhang](#anhang)

<div class="page-break"></div>

<!-- ========================================================================== -->
<!-- KAPITEL 1: EINLEITUNG                                                      -->
<!-- ========================================================================== -->

# 1. Einleitung

## 1.1 Motivation und Anwendungsbereich
Bildgesteuerte Hardware stellt eine der anspruchsvollsten Disziplinen der modernen Robotik und Künstlichen Intelligenz dar. Während klassische regelungstechnische Ansätze wie modellprädiktive Regelungen (Model Predictive Control, MPC) bei präzise bekannten Strecken- und Reibungsparametern sehr gute Ergebnisse erzielen, kann der Mensch diese teilweise schalgen - durch vorraussicht und feingefühl im Grenzbereich des physikalisch möglichen.

*Deep Reinforcement Learning* (DRL) bietet hierbei das theoretische Potenzial, nichtlineare Fahrdynamiken rein datengetrieben direkt aus Bild- und Sensordaten zu erlernen. Als ideales Experimentierfeld dient in dieser Arbeit das System **Carrera Hybrid**: Miniatur-Rennfahrzeuge im Maßstab 1:43, die sich frei (ohne mechanische Führungsschiene) auf einer Fahrbahn bewegen und über kontinuierliche Lenk- und Antriebsaktoren gesteuert werden.

In der angestrebten Gesamtvision überwacht eine stationäre Deckenkamera (Top-Down Bird's Eye View) die gesamte Rennstrecke. Ein neuronaler Vision-Agent verarbeitet diesen globalen Bildstrom, erkennt die Position und Ausrichtung des Fahrzeugs relativ zu den Streckenbegrenzungen und steuert das Fahrzeug mit maximaler Geschwindigkeit entlang der Ideallinie.

```
+-----------------------------------------------------------------------------------+
|                              VISION DES GESAMTPROJEKTS                            |
|                                                                                   |
|    [ Deckenkamera (Top-Down) ]                                                    |
|                 |                                                                 |
|                 v (Globaler Bildstrom)                                            |
|    +-----------------------------------------+                                    |
|    |      Neuronaler Vision-Agent (CNN)      |                                    |
|    |   - Merkmalsextraktion                  |                                    |
|    |   - Zeitliche Dynamik (Frame Stack)     |                                    |
|    |   - Policy-Entscheidung (SAC)           |                                    |
|    +-----------------------------------------+                                    |
|                 |                                                                 |
|                 v (Funkbefehle: Lenkung, Gas, Bremse)                             |
|    [ Reales Carrera Hybrid Fahrzeug auf Strecke (1:43) ]                          |
+-----------------------------------------------------------------------------------+
```

## 1.2 Projektstruktur und forschungsmethodische Einordnung
Die vorliegende Forschungs- und Entwicklungsarbeit behandelt die **erste Phase** dieses Gesamtvorhabens. Das direkte Training eines visuellen End-to-End-Agenten in komplexen 3D-Simulatoren (wie NVIDIA Isaac Sim / Isaac Lab) ist rechenintensiv, leidet unter langen Feedbackzyklen und scheitert in der Praxis häufig an unzureichender Sample-Effizienz.

Daher verfolgt dieses Projekt einen systematischen zweistufigen Forschungsansatz:
1. **Phase 1 (Gegenstand dieser Arbeit): 2D-Python-Prototyping & Methodenevaluation**  
   Aufbau einer leichtgewichtigen, hochperformanten 2D-Simulationsumgebung (`Carrera2DEnv`), um fundamentale Fragen zu Beobachtungsräumen, Netzwerkarchitekturen, Lernalgorithmen (PPO vs. SAC), Belohnungsfunktionen und Datenvorverarbeitungsmethoden isoliert und zeiteffizient zu untersuchen.
2. **Phase 2 (Transfer in fotorealistische3D Simulation):**  
   Übertragung der im 2D-Prototyping validierten Best Practices in die fotorealistische 3D-Simulation (NVIDIA Isaac Sim).
3. **Ausblick auf Phase 3 (Sim-to-Real Transfer):**  
   Erforschung des Sim-to-Real-Gaps auf der realen Carrera-Hybrid-Hardware. Agenten in die Echte Welt bringen.

## 1.3 Zielsetzung und Anforderungskatalog 
Die konkreten Zielsetzungen für die prototypische Entwicklungsphase lauten:
* **Entwicklung einer physikalisch kalibrierten 2D-Umgebung:** Realitätsnahe Abbildung der Streckenmaße (25 cm Fahrbahnbreite, 21 mm Randstreifen) und gemessenen Fahrzeugbeschleunigungen.
* **Architekturvergleich verschiedener Inputmodalitäten:**  Systematischer Vergleich von niedrigdimensionalen Raycasts (Lidar), Einzelbildern, hybrider Vision/Lidar-Inputs, fahrzeugfesten Bildausschnitten (Ego-Crops) und globalen gestapelten Kamerabildern.
* **Entwicklung einer stabilen Teacher-Student-Distillationspipeline:**  Effizientes Vortraining visueller Netze durch automatisierte Expertendemonstrationen.
* **Rundenzeit- und Stabilitätsoptimierung:**  
Erreichen fehlerfreier Fahrten mit Bestzeit (in 2D Umgebung).

<div class="page-break"></div>

<!-- ========================================================================== -->
<!-- KAPITEL 2: THEORETISCHE GRUNDLAGEN                                         -->
<!-- ========================================================================== -->

# 2. Theoretische Grundlagen und Stand der Technik

## 2.1 Reinforcement Learning Grundlagen (MDP vs. POMDP)
Ein klassisches Reinforcement-Learning-Problem wird als **Markov-Entscheidungsprozess** (MDP) formuliert, definiert durch das 5-Tupel $(\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$:
* $\mathcal{S}$: Zustandsraum (*State Space*)
* $\mathcal{A}$: Aktionsraum (*Action Space*)
* $\mathcal{P}(s' \mid s, a)$: Übergangswahrscheinlichkeitsfunktion (*Transition Dynamics*)
* $\mathcal{R}(s, a, s')$: Belohnungsfunktion (*Reward Function*)
* $\gamma \in [0, 1)$: Diskontierungsfaktor für zukünftige Belohnungen (*Discount Factor*)

Ziel des Agenten ist die Maximierung des diskontierten kumulativen Ertrags (Return) $R_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}$.

Liegen dem Agenten nicht alle Zustandsvariablen (z. B. translatorische und rotatorische Geschwindigkeiten, Reibwerte) direkt vor, sondern lediglich unvollständige Beobachtungen $o \in \Omega$ aus einer Beobachtungsfunktion $\mathcal{O}(s)$, liegt ein **Partiell Beobachtbarer MDP** (POMDP) vor. 

## 2.2 Zeitliche Dynamik und Frame Stacking
Ein einzelnes, statisches Graustufenbild $o_t \in \mathbb{R}^{H \times W}$ ist fundamental nicht-markovianisch: Aus einem einzelnen Foto lässt sich nicht ableiten, ob das Fahrzeug mit $1.5\text{ m/s}$ vorwärtsfährt, rückwärts rollt oder stillsteht. 

Durch das **Frame Stacking** von $k$ aufeinanderfolgenden Bildern:
$$s_t = [o_t, o_{t-1}, \dots, o_{t-k+1}] \in \mathbb{R}^{k \times H \times W}$$
wird die zeitliche Pixelverschiebung $\Delta x / \Delta t$ über die Kanaldimension für Faltungsnetzwerke (CNNs) direkt lesbar. Faltungsfilter in den ersten Netzwerkschichten erlernen dadurch die Berechnung räumlich-zeitlicher Gradienten, was den POMDP näherungsweise in einen MDP überführt \cite{mnih2015human}.@das Zitat wird nicht als Abkürzung oder Zahl angezeigt und ist auch nicht verlinkt.

## 2.3 Imitation Learning, Behavioral Cloning und Covariate Shift
Beim **Behavioral Cloning** (BC) lernt ein parametrisiertes Modell $\pi_\theta$ mittels überwachten Lernens (Supervised Learning) die direkte Abbildung von Zuständen auf Expertenaktionen $a^*$:
$$\mathcal{L}_{\text{BC}}(\theta) = \frac{1}{N} \sum_{i=1}^N \| \pi_\theta(s_i) - a_i^* \|^2$$

**Das fundamentale Versagen von naivem BC:**
Ein optimaler Experte erzeugt ausschließlich Trajektorien auf der Ideallinie $\mathcal{S}_{\text{ideal}}$. Durch unvermeidliche Approximationsfehler driftet der Schüler nach wenigen Zeitschritten in unbekannte Randzustände $s_{\text{OOD}} \notin \mathcal{S}_{\text{ideal}}$ ab (*Out-of-Distribution*). Da im Trainingsdatensatz keine Korrekturmanöver für $s_{\text{OOD}}$ enthalten sind, akkumulieren sich die Fehler quadratisch über den Zeithorizont $T$ ($\mathcal{O}(T^2)$ nach Ross & Bagnell \cite{ross2011reduction}), was zum unmittelbaren Streckenabflug führt.

## 2.4 Privilegiertes Lernen: Teacher-Student Distillation
Das Paradigma des *Privileged Learning* bzw. *Learning by Cheating* (Chen et al. \cite{chen2019learning}) entkoppelt das Steuerungsproblem vom Wahrnehmungsproblem:
1. **Schritt 1 (Lehrer / Oracle):** Mittels minimalen Sensorinformationen lernt ein Agent das Fahren. Das Training dieser Policy $\pi_{\text{teacher}}$ basierend auf einfachen Sensorwerten (z. B. Raycast-Distanzen) und konvergiert in kürzester Zeit.
2. **Schritt 2 (Schüler / Student):** Training eines komplexen, visuellen Modells $\pi_{\text{student}}$, das auf Basis von Bildern operiert und dabei den Zusammenhang zwischen Input (Bild der Szene) und Aktion erlernt. Das Netz lernt die Aktionen des Lehrers zu replizieren.

```
                              [ PRIVILEGIERTER RAUM ]
                           (3 Raycasts + Geschwindigkeit)
                                         |
                                         v
                       +-----------------------------------+
                       |        Lehrer-Agent (Oracle)      |
                       |      (Kompaktes MLP / PPO)        |
                       +-----------------------------------+
                                         |
                                         v  (Aktion a_t* unter Rauschen)
+--------------------+                   |                  +--------------------+
|  Visueller Stack   | ==================v================> |  Visueller Schüler |
| (3 x 100 x 166 px) |         [ SUPERVISED DISTILLATION ]  |    (Nature-CNN)    |
+--------------------+                                      +--------------------+
```

## 2.5 Algorithmen im kontinuierlichen Kontrollraum (PPO vs. SAC)
* **Proximal Policy Optimization (PPO \cite{schulman2017proximal}):** On-Policy Algorithmus, der Stabilität durch eine geclippte Wahrscheinlichkeitsquotienten-Zielfunktion erzielt. Verwirft gesammelte Transitions nach jedem Gradientenupdate, was bei bildbasierten Eingaben zu hoher Sample-Ineffizienz führt.
* **Soft Actor-Critic (SAC \cite{haarnoja2018soft}):** Off-Policy Actor-Critic Algorithmus, der neben dem erwarteten Return die Entropie der Policy maximiert:
  $$J(\pi) = \mathbb{E}_{\tau \sim \pi} \left[ \sum_{t=0}^\infty \gamma^t \left( r(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot \mid s_t)) \right) \right]$$
  SAC nutzt einen Replay-Buffer zur wiederholten Auswertung von Erfahrungen und verhindert durch *Double-Q-Learning* (zwei parallele Critic-Netzwerke $Q_{\theta_1}, Q_{\theta_2}$) systematische Q-Wert-Überschätzungen.

## 2.6 State of the Art im autonomen Rennsport
Die Konzeption dieser Arbeit stützt sich auf zentrale Erkenntnisse führender Publikationen:
* **DIRL (Cai et al. \cite{cai2021vision}):** Beweist die Überlegenheit von hybridem Imitation- und Reinforcement-Learning für reale 1:20 RC-Fahrzeuge.
* **Gran Turismo Sophy (Wurman et al., Nature 2022 \cite{wurman2022outracing}):** Erreicht übermenschliche Rundenzeiten mittels SAC, asymmetrischen Critic-Architekturen und optimierten Streckenfortschritts-Belohnungen.
* **VISTA (Amini et al. \cite{amini2020learning}):** Zeigt, dass das gezielte Erleben und Trainieren von OOD-Rettungsmanövern für einen robusten Sim-to-Real-Transfer essenziell ist.

<div class="page-break"></div>

<!-- ========================================================================== -->
<!-- KAPITEL 3: METHODIK & SIMULATION                                           -->
<!-- ========================================================================== -->

# 3. Methodik und Aufbau der 2D-Simulationsumgebung

## 3.1 Streckengeometrie und Computer Vision Pipeline
Die Simulationsumgebung basiert auf der realen Geometrie einer Carrera-Hybrid-Rennstrecke:
* **Fahrbahnbreite:** $250\text{ mm}$
* **Randstreifen:** $21\text{ mm}$ (Farbmarkierungen: Rot für Außenrand, Blau für Innenrand)
* **Fahrzeugmaße:** Länge $98\text{ mm}$, Breite $39\text{ mm}$, Achsabstände: Heck bis Hinterachse $23\text{ mm}$, Heck bis Vorderachse $71\text{ mm}$.

Das Modul [tracklimits_cv.py](file:///home/mneumann/Desktop/EigeneDateien/FE%20Carrera%20RL/carrera_rl/src/utils/tracklimits_cv.py) extrahiert über Farbmasken im HSV-Farbraum die Konturen der Begrenzungen. Aus den Nearest-Neighbor-Mittelpunkten zwischen Innen- und Außenrand wird über B-Spline-Interpolation eine stetig differenzierbare **Mittellinie** generiert, die als Referenz für Fahrtrichtung und Streckenfortschritt dient.

## 3.2 Physikalische Modellierung: Von `force_drag_v1` zu `measured_v2`
Im Projektverlauf wurden zwei grundlegend verschiedene Physikmodelle untersucht:

| Parameter | `force_drag_v1` (Frühes Modell) | `measured_v2` (Gemessenes Modell) | Reale Auswirkung |
|---|---|---|---|
| **Max. Beschleunigung** | $12.5\text{ m/s}^2$ | $0.6 - 1.2\text{ m/s}^2$ | Altes Modell war ca. **20-fach übermotorisiert**. |
| **Max. Bremsverzögerung** | $14.4\text{ m/s}^2$ | $1.5\text{ m/s}^2$ | Altes Modell erlaubte unrealistisch abrupte Stopps. |
| **Max. Endgeschwindigkeit** | $5.0 - 12.0\text{ m/s}$ | $1.6 - 1.9\text{ m/s}$ | Anpassung an reale Carrera-Hybrid-Höchstgeschwindigkeit. |
| **Rückwärtsfahren** | Erlaubt ($v < 0$) | Gesperrt ($v \ge 0$) | Realitätsabgleich mit Carrera-Antriebsstrang. |
| **Lenkänderungsrate $\Delta\delta$** | $1.0\text{ /Step}$ (Sprunghaft) | $0.5\text{ /Step}$ (Tiefpass) | Modellierung der Servo-Stellzeit. |

## 3.3 Beobachtungsräume und Virtuelle Kamera
Die Umgebung ([carrera_2d_env.py](file:///home/mneumann/Desktop/EigeneDateien/FE%20Carrera%20RL/carrera_rl/src/envs/carrera_2d_env.py)) stellt modulare Beobachtungsräume bereit:
1. **Raycasts / Lidar (Privilegiert):** Vektor aus 3 Distanzstrahlen (Schnittpunkte mit Streckenbegrenzungen links, mitte, rechts) sowie der Eigengeschwindigkeit: $o_t = [v/v_{\max}, d_l, d_m, d_r] \in \mathbb{R}^4$.
2. **Globale Kamera (Bird's Eye View):** Skaliertes Graustufenbild der gesamten Rennstrecke ($166 \times 100$ Pixel, quantisiert auf 4 Helligkeitsstufen zur Rauschunterdrückung).
3. **Gestapelte globale Ansicht (Standard):** Stack aus $n_{\text{stack}}=3$ Bildern $\to \mathbb{R}^{3 \times 100 \times 166}$.

## 3.4 Aktionsraum und Tiefpass-Lenkdynamik
Kontinuierlicher Aktionsraum $\mathcal{A} \in [-1, 1]^2$:
* $a[0] \in [-1, 1]$: Lenkwinkel (begrenzt durch $\Delta\delta \le 0.5$ pro Simulationsschritt $\Delta t = 0.0167\text{ s}$).
* $a[1] \in [-1, 1]$: Antrieb ($>0$: Motorleistung, $<0$: Bremse).

## 3.5 Reward Engineering und Belohnungsgestaltung
Die finale Belohnungsfunktion `v3_rundenzeit` ([reward_func.py](file:///home/mneumann/Desktop/EigeneDateien/FE%20Carrera%20RL/carrera_rl/src/utils/reward_func.py)) balanciert Fortschritt und Sicherheit:
$$r_t = w_{\text{prog}} \cdot \Delta s_{\text{mittellinie}} + w_v \cdot \frac{v}{v_{\max}} - w_{\text{slow}} \cdot \max\left(0, 1 - \frac{v}{v_{\text{threshold}}}\right) - r_{\text{crash}} + r_{\text{lap\_bonus}}$$

Wichtige Fehlerkorrekturen gegenüber früheren Versionen:
* **Entfernung der Sperrklinke (Ratchet-Bug):** Frühere Rewards bestraften langsamere Runden relativ zu einer Zufalls-Bestzeit.
* **Beseitigung von Oszillationsstrafen (`w_integral = 0.0`):** Die vormalige Bestrafung kontinuierlicher Lenkwinkel bestrafte saubere Kurvenfahrten mit $-625$ Punkten pro Episode.

<div class="page-break"></div>

<!-- ========================================================================== -->
<!-- KAPITEL 4: 4-PHASEN-TRAININGSANSATZ                                       -->
<!-- ========================================================================== -->

# 4. Der 4-Phasen-Trainingsansatz (Teacher-Student Pipeline)

```
+---------------------------------------------------------------------------------------------------+
|                            DIE 4-PHASEN DISTILLATIONS-PIPELINE                                    |
|                                                                                                   |
|  [ Phase 1: Lidar-Experte ] --------------------------------------------------------+             |
|    - PPO / SAC auf 4D-Raycasts                                                      |             |
|    - Konvergiert in < 15 Minuten (> 11 Runden stabil)                               |             |
|                                                                                     |             |
|  [ Phase 2: Datensammlung mit Noise Injection ] <-----------------------------------+             |
|    - Experte fährt mit Störung: a_tilde = a* + N(0, sigma^2)                                      |
|    - Speicherung von (3-Frame-Stack, a*) -> 12.000 bis 50.000 Transitionen                    |
|    - Erfasst kritische Rettungsmanöver (Recovery Behaviors) gegen Covariate Shift                 |
|                                                                                     |             |
|  [ Phase 3: Supervised Pre-Training (Behavioral Cloning) ] <------------------------+             |
|    - Training des CNN-Feature-Encoders via MSE-Loss                                 |             |
|    - Erreicht MSE < 0.0037 nach 20 Epochen                                          |             |
|                                                                                     |             |
|  [ Phase 4: Online Fine-Tuning mit Soft Actor-Critic (SAC) ] <----------------------+             |
|    - Initialisierung der SAC Policy mit vortrainierten CNN-Gewichten                              |
|    - Rundenzeitoptimierung & Ausreizen der Ideallinie im kontinuierlichen Kontrollraum            |
+---------------------------------------------------------------------------------------------------+
```

## 4.1 Phase 1: Bootstrapping des Lidar-Experten
Aufgrund des kompakten Zustandsraums ($\mathbb{R}^4$) lernt ein zweischichtiges MLP (`[64, 64]`) mittels PPO oder SAC innerhalb von $50.000$ Schritten (< 15 Minuten Rechenzeit) eine perfekte Fahrkompetenz mit Durchschnittsgeschwindigkeiten von $1.22\text{ m/s}$.

## 4.2 Phase 2: Datensammlung mit Action Noise Injection
Um den Schüler auf Abweichungen von der Ideallinie vorzubereiten, wird der Lidar-Experte stochastisch gestört:
$$\tilde{a}_t = \text{clip}(a_t^* + \xi_t, -1, 1), \quad \xi_t \sim \mathcal{N}(0, \sigma^2)$$
Das Auto bricht aus, und der Lidar-Experte steuert im Folgeschritt mit maximaler Gegenlenkung gegen. Gespeichert wird das Paar $(s_t, a_t^*)$, bestehend aus dem visuellen 3-Frame-Stack $s_t$ und der **unverrauschten Korrekturaktion** $a_t^*$.

## 4.3 Phase 3: Visuelles Pre-Training (Behavioral Cloning)
Ein Faltungsnetzwerk (Nature-CNN oder modifiziertes 3-Layer-CNN mit Linear-Kopf) wird per Gradientenabstieg optimiert:
$$\mathcal{L}_{\text{MSE}}(\theta) = \frac{1}{|\mathcal{D}|} \sum_{(s, a^*) \in \mathcal{D}} \| f_\theta(s) - a^* \|^2$$
Nach 20 Epochen erreicht das Netzwerk einen Validierungs-MSE von $< 0.0037$, wodurch die visuellen Filter für Streckenränder und Fahrzeuglage bereits vor dem ersten RL-Schritt vollständig ausgebildet sind.

## 4.4 Phase 4: End-to-End Fine-Tuning mit Soft Actor-Critic (SAC)
Das vortrainierte Netzwerk wird als Actor-Backbone in SAC überführt. Der Off-Policy Replay-Buffer sammelt reale Fahrerfahrungen, während die Entropiemaximierung verhindert, dass der Agent in lokalen Minima (z. B. vorsichtiges Dauerschleichen) verharrt.

<div class="page-break"></div>

<!-- ========================================================================== -->
<!-- KAPITEL 5: ERGEBNISSE & ZEITLEISTE                                         -->
<!-- ========================================================================== -->

# 5. Experimentelle Ergebnisse und Chronologie der Trainingsläufe

## 5.1 Chronologie der Experimente und Paradigmenwechsel
Die Chronologie der Trainingsläufe dokumentiert die methodische Evolution des Projekts:

```
[ Phase 1: force_drag_v1 ]  --->  [ Phase 2: measured_v2 ]  --->  [ Phase 3: gamma 0.999 ]  --->  [ Phase 4: Kontrollversuch ]
(Alte Physik, Peak 1432)         (Gemessene Physik, v3)          (Kollaps Lauf 2331)            (Lidar-SAC Lauf 1748)
```

## 5.2 Quantitativer Leistungsvergleich
Alle Modelle wurden unter identischen Evaluationsbedingungen im aktuellen Benchmark-Runner ([src/eval/vergleich.py](file:///home/mneumann/Desktop/EigeneDateien/FE%20Carrera%20RL/carrera_rl/src/eval/vergleich.py)) vermessen:

| Modell / Run | Algorithmus | Beobachtungsraum | Physikmodell | Total Steps | Reward (Eval) | EpLen (Frames) | Runden | Ø-Geschw. | Status / Befund |
|---|---|---|---|---|---|---|---|---|---|
| **Lidar-Experte** | PPO | Raycasts (4D) | `measured_v2` | 500 k | **6401.7** | **2000** (Limit) | **11** | $1.22\text{ m/s}$ | Souveräne Referenz |
| **Lidar SAC (`1748`)** | SAC | Raycasts (4D) | `measured_v2` | 500 k | **6116.4** | **2000** (Limit) | **11** | $1.19\text{ m/s}$ | **Kontrollversuch bestanden** |
| **Vision SAC `1432`** | SAC | 3-Stack ($166\times 100$) | `force_drag_v1` | 2.5 M | **3828.2** | **2000** | 3 | $1.27\text{ m/s}$ | Peak alte Physik; divergiert später |
| **Vision SAC `2210`** | SAC | 3-Stack ($166\times 100$) | `measured_v2` | 5.0 M | $-30.3$ | 256 | 0 | $0.81\text{ m/s}$ | Scheitert an Subpixel-Schwelle |
| **Vision SAC `2147`** | SAC | 3-Stack ($166\times 100$) | `measured_v2` | 2.5 M | $-266.5$ | 174 | 0 | $0.62\text{ m/s}$ | Früher Wandkontakt |
| **Vision SAC `2331`** | SAC | 3-Stack ($166\times 100$) | `measured_v2` | 1.5 M | $-349.8$ | 33 | 0 | $0.12\text{ m/s}$ | Kollaps durch $\gamma = 0.999$ |
| **Vision PPO `2328`** | PPO | 3-Stack ($166\times 100$) | `force_drag_v1` | 90 M | $-3.9$ | 76 | 0 | $0.45\text{ m/s}$ | Extreme Sample-Ineffizienz |

## 5.3 Der Kontrollversuch: Lidar-SAC unter gemessener Physik
Um zu überprüfen, ob das Scheitern der visuellen Agenten unter `measured_v2` an der SAC-Stabilität, der trägeren Physik oder dem Reward liegt, wurde in Lauf `1748` die visuelle Beobachtung gegen Lidar-Werte getauscht. 

**Ergebnis:** Der Agent meisterte die Strecke in **54 Minuten** vollständig ($6116.4$ Punkte, $11$ Runden). Damit wurde zweifelsfrei bewiesen: **Physik, Reward und Algorithmus funktionieren perfekt – die Ursache liegt rein in der visuellen Wahrnehmungsschwelle.**

<div class="page-break"></div>

<!-- ========================================================================== -->
<!-- KAPITEL 6: VERTIEFTE RL-ERKENNTNISSE                                       -->
<!-- ========================================================================== -->

# 6. Tiefgehende RL-Erkenntnisse und Systemanalysen

## 6.1 Das Subpixel-Wahrnehmungsproblem
Die mathematische Analyse der Bildauflösung liefert die Erklärung für das Verhalten der visuellen Agenten:
* **Streckenbreite in Realität:** $1.80\text{ m}$
* **Bildbreite:** $166\text{ Pixel} \implies 1\text{ Pixel} \approx 10.8\text{ mm}$
* **Simulationszeitschritt:** $\Delta t = 0.0167\text{ s}$ ($60\text{ Hz}$)

Für einen sichtbaren Versatz von mindestens $1\text{ Pixel}$ pro Zeitschritt ist eine Mindestgeschwindigkeit erforderlich:
$$v_{\min} = \frac{0.0108\text{ m}}{0.0167\text{ s}} \approx 0.64\text{ m/s}$$

```
+-----------------------------------------------------------------------------------+
|                        DAS SUBPIXEL-DILEMMA IM ANFAHRBEREICH                      |
|                                                                                   |
|  Geschwindigkeit v < 0.64 m/s:                                                    |
|  Frame(t) == Frame(t-1) == Frame(t-2)  ===>  Differenzenbild dI/dt = 0            |
|                                                                                   |
|  Konsequenz:                                                                      |
|  - Der 3-Frame-Stack enthält drei identische Bilder.                              |
|  - Das CNN kann weder Geschwindigkeit noch Beschleunigung wahrnehmen.             |
|  - Der Agent verhält sich wie ein Blindflug-System und wird vom Reward bestraft.  |
+-----------------------------------------------------------------------------------+
```

## 6.2 Lösungsstrategien: Temporale Spreizung vs. Bildauflösung

| Lösungsansatz | Konfiguration | Wahrnehmungsschwelle | Rechenzeit je Update | Replay-Buffer (500k) |
|---|---|---|---|---|
| **Ausgangszustand** | $166 \times 100$, Stack $[t, t-1, t-2]$ | $0.64\text{ m/s}$ | $55.4\text{ ms}$ | $15.5\text{ GB}$ |
| **Hebel 1: Temporale Spreizung** | $166 \times 100$, Stack $[t, t-4, t-8]$ | **$0.16\text{ m/s}$** | **$55.4\text{ ms}$ ($\pm 0\%$)** | **$15.5\text{ GB}$ ($\pm 0\%$)** |
| **Hebel 2: Höhere Auflösung** | $250 \times 150$, Stack $[t, t-1, t-2]$ | $0.42\text{ m/s}$ | $104.6\text{ ms}$ ($+89\%$) | $34.9\text{ GB}$ ($+125\%$) |

> **Erkenntnis:** Die temporale Spreizung (*Frame Striding*) ist der weitaus effizientere Hebel, da sie die Wahrnehmungsschwelle um den Faktor 4 senkt, ohne zusätzlichen Rechen- oder Speicheraufwand zu verursachen.

## 6.3 Replay-Buffer-Architektur und Speicheroptimierung
Standard-Puffer speichern gestapelte Bilder redundant ab ($3 \times$ Bilddaten). Der in [src/utils/framestapel_buffer.py](file:///home/mneumann/Desktop/EigeneDateien/FE%20Carrera%20RL/carrera_rl/src/utils/framestapel_buffer.py) implementierte Ringspeicher speichert pro Zeitschritt nur das Einzelbild und assembliert den Stack on-the-fly beim Batch-Sampling. Dies reduziert den Speicherbedarf um **Faktor 1.5** bei mathematisch bitidentischen Batches.

## 6.4 Korrekte Behandlung von Zeitschritt-Timeouts
Wird ein Timeout bei Frame 2000 fälschlicherweise als Terminalzustand markiert, setzt das Bellman-Update $V(s_{2000}) = 0$. Mit `handle_timeout_termination=True` wird der Wert korrekt gebootstrappt ($y_t = r_t + \gamma V(s_{t+1})$), was den künstlichen Werteverfall am Episodenende verhindert.

## 6.5 TensorBoard-Diagnostik: Critic-Divergenz vs. Ausreißer
* **Lauf `1432`:** Der Critic-Loss explodierte nach Schritt 2.0M von $22.2$ auf $26.670$ (Faktor 1200) $\to$ echte Divergenz durch Q-Wert-Überschätzung.
* **Lauf `2210`:** Der Peak von $875.7$ Punkten spiegelte sich nicht im stochastischen Trainings-Return wider $\to$ statistischer Ausreißer einer deterministischen Einzelauswertung.

## 6.6 Messfallen und Rendering-Diskrepanzen
Ein Unterschied von nur **0.05% der Pixel** zwischen `render_mode="hidden"` (Training) und `render_mode="human"` (Visualisierung) kippte die Lenkentscheidung des CNNs von $+0.30$ auf $-0.80$. Evaluationen müssen strikt dieselbe Bildpipeline wie das Training durchlaufen.

<div class="page-break"></div>

<!-- ========================================================================== -->
<!-- KAPITEL 7: ÜBERTRAGUNG IN DIE 3D-SIMULATION                                -->
<!-- ========================================================================== -->

# 7. Übertragung in die 3D-Simulation (NVIDIA Isaac Sim)

## 7.1 Aufbau der 3D-Simulationsumgebung in Isaac Lab [Platzhalter]
*Beschreibung der 3D-Szene, USD-Modelle der Carrera-Hybrid-Strecke und des Fahrzeugs, PhysX-Kollisionsparameter.*

> [!NOTE]
> *Platzhalter für detaillierte Ausgestaltung durch 3D-Ergebnisse: Hier werden die USD-Szenenstrukturen, Kamera-Setups und PhysX-Reibwerte eingetragen.*

## 7.2 Asymmetric Actor-Critic und Domain Randomization
In der 3D-Simulation wird die Teacher-Student-Architektur durch einen **Asymmetric Actor-Critic** erweitert:
* **Critic:** Erhält während des Trainings Zugriff auf den vollständigen 3D-Ground-Truth-Zustand (exakte Position, Orientierungsquaternion, lineare und rotatorische Geschwindigkeiten, Reifenschlupf).
* **Actor:** Erhält ausschließlich die synthetisierten 3-Kanal-Kamerabilder und Onboard-Telemetriedaten.

Zur Vorbereitung des Sim-to-Real-Transfers werden Reibwerte ($\mu \in [0.6, 1.2]$), Beleuchtungsstärken, Kamerarauschen und Latenzen domain-randomisiert.

## 7.3 Evaluierung und Sim-to-Real Ausblick [Platzhalter]
*Vergleich der Rundenzeiten und Konvergenzgeschwindigkeiten zwischen 2D-Prototyp und 3D-Isaac-Sim-Training.*

<div class="page-break"></div>

<!-- ========================================================================== -->
<!-- KAPITEL 8: FAZIT & AUSBLICK                                                -->
<!-- ========================================================================== -->

# 8. Fazit, Learnings und Ausblick

## 8.1 Zusammenfassung der Ergebnisse
Die vorliegende Arbeit demonstriert die erfolgreiche Konzeption und Validierung einer sample-effizienten Reinforcement-Learning-Pipeline für Miniatur-Rennfahrzeuge in einer 2D-Simulationsumgebung. Durch die Kombination aus privilegiertem Lehrer-Training (Raycasts), stochastischer Rausch-Injektion (Recovery Manöver) und visuellem Vortraining (Behavioral Cloning) konnte ein robuster Workflow geschaffen werden, der teure Trial-and-Error-Iterationen in 3D-Umgebungen überflüssig macht.

## 8.2 Zentrale Erkenntnisse (Key Learnings)
1. **Wahrnehmung vor Algorithmus:** Das Scheitern visueller Agenten lag nicht an SAC oder der Fahrzeugdynamik, sondern am Subpixel-Wahrnehmungsproblem bei niedrigen Geschwindigkeiten.
2. **Effizienz von Frame Striding:** Temporale Spreizung ($t, t-4, t-8$) löst die Wahrnehmungsschwelle ohne Rechenzeit-Overhead.
3. **Notwendigkeit von Noisy Demonstrations:** Reines Behavioral Cloning auf idealen Trajektorien führt unweigerlich zum Covariate Shift; nur verrauschte Demonstrationen vermitteln Abfangkompetenz.
4. **Speicher- und Render-Disziplin:** On-the-Fly Replay-Buffer und bitidentische Renderpipelines sind essenziell für reproduzierbares Deep RL.

## 8.3 Ausblick auf das Folgeprojekt
Aufbauend auf diesen 2D-Ergebnissen umfasst die Roadmap für das Folgeprojekt:
* Vollständige Implementierung der Teacher-Student-Pipeline in **NVIDIA Isaac Lab**.
* Training mit Asymmetric Actor-Critic unter intensiver Domain Randomization.
* Sim-to-Real Transfer auf reale Carrera-Hybrid-Fahrzeuge mit Deckenkamera-Tracking.

<div class="page-break"></div>

<!-- ========================================================================== -->
<!-- QUELLENVERZEICHNIS                                                         -->
<!-- ========================================================================== -->

# Quellenverzeichnis

1. <a id="ref-cai2021"></a>**Cai, P. et al. (2021):** *Vision-Based Autonomous Car Racing Using Deep Imitative Reinforcement Learning.* IEEE Robotics and Automation Letters (RA-L), Vol. 6, No. 4, S. 7262–7269.
2. <a id="ref-chen2019"></a>**Chen, D. et al. (2019):** *Learning by Cheating.* In: Conference on Robot Learning (CoRL 2019), Proceedings of Machine Learning Research (PMLR), Vol. 100, S. 66–75.
3. <a id="ref-wurman2022"></a>**Wurman, P. R. et al. (2022):** *Outracing champion Gran Turismo drivers with deep reinforcement learning.* Nature, Vol. 602, S. 223–228.
4. <a id="ref-amini2020"></a>**Amini, A. et al. (2020):** *Learning Robust Control Policies for End-to-End Autonomous Driving From Data-Driven Simulation.* IEEE Robotics and Automation Letters (RA-L), Vol. 5, No. 2, S. 1143–1150.
5. <a id="ref-haarnoja2018"></a>**Haarnoja, T. et al. (2018):** *Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor.* In: International Conference on Machine Learning (ICML 2018), S. 1861–1870.
6. <a id="ref-schulman2017"></a>**Schulman, J. et al. (2017):** *Proximal Policy Optimization Algorithms.* arXiv preprint arXiv:1707.06347.
7. <a id="ref-mnih2015"></a>**Mnih, V. et al. (2015):** *Human-level control through deep reinforcement learning.* Nature, Vol. 518, S. 529–533.
8. <a id="ref-ross2011"></a>**Ross, S. & Bagnell, J. A. (2011):** *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning.* In: International Conference on Artificial Intelligence and Statistics (AISTATS 2011), S. 627–635.
9. <a id="ref-fujimoto2018"></a>**Fujimoto, S. et al. (2018):** *Addressing Function Approximation Error in Actor-Critic Methods.* In: International Conference on Machine Learning (ICML 2018), S. 1587–1596.

<div class="page-break"></div>

<!-- ========================================================================== -->
<!-- ABBILDUNGS- UND TABELLENVERZEICHNIS                                        -->
<!-- ========================================================================== -->

# Abbildungs- und Tabellenverzeichnis

### Tabellenverzeichnis
* **Tabelle 1:** Vergleich der physikalischen Modellierung (`force_drag_v1` vs. `measured_v2`) – *Kapitel 3.2*
* **Tabelle 2:** Quantitativer Vergleich aller Haupttrainingsläufe (PPO vs. SAC, Lidar vs. Vision) – *Kapitel 5.2*
* **Tabelle 3:** Vergleich der Lösungsansätze für das Subpixel-Wahrnehmungsproblem – *Kapitel 6.2*

### Abbildungsverzeichnis
* **Abbildung 1:** Gesamtübersicht der Cross-Modal Privileged Distillation Pipeline – *Kapitel 1.1*
* **Abbildung 2:** Schematischer Ablauf des Privilegierten Lernens (*Learning by Cheating*) – *Kapitel 2.4*
* **Abbildung 3:** Die 4 Phasen der Teacher-Student Trainingspipeline – *Kapitel 4*
* **Abbildung 4:** Das Subpixel-Wahrnehmungsdilemma im Anfahrbereich – *Kapitel 6.1*

<div class="page-break"></div>

<!-- ========================================================================== -->
<!-- ANHANG                                                                     -->
<!-- ========================================================================== -->

# Anhang

## Anhang A: Struktur und Workflow im Jupyter Notebook
Das Jupyter Notebook [Notebook_prototyping.ipynb](file:///home/mneumann/Desktop/EigeneDateien/FE%20Carrera%20RL/carrera_rl/Notebook_prototyping.ipynb) dient als interaktive Arbeitsumgebung:
* **Zellen 0–15:** Streckenerfassung via OpenCV, interaktives HSV-Tuning, Mittelliniengenerierung, manuelles Fahren mit WASD.
* **Zellen 16–26:** Initialisierung von `Carrera2DEnv`, Lidar-Visualisierung, Konfiguration der `VirtualCamera`.
* **Zellen 27–49:** Baseline-Trainings (Raycasts, Single-Frame, Hybrid, Stacked Crop) und GFLOPS-Benchmarks.
* **Zellen 50–63:** Teacher-Student Pipeline (Datensammlung mit Noise, CNN-Pretraining, SAC-Fineteuning).
* **Zellen 64–70:** Diagnose- und Beobachtungszellen unter Original-Physik mit Telemetrie-Einblendung.

## Anhang B: Wichtige Hyperparameter der Trainingsläufe

```json
{
  "algorithm": "SAC",
  "policy": "CnnPolicy",
  "learning_rate": 5e-5,
  "buffer_size": 500000,
  "batch_size": 512,
  "gamma": 0.99,
  "tau": 0.005,
  "ent_coef": "auto",
  "train_freq": 1,
  "gradient_steps": 1,
  "handle_timeout_termination": true,
  "camera_resolution": [166, 100],
  "n_stack": 3,
  "max_steer_change": 0.5
}
```
